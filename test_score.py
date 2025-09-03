# -*- coding: utf-8 -*-
"""
Score-based cross-fold majority evaluation.

Procedure:
 1. Define n (N_FIRST_SAMPLE). Randomly sample n images from interval_sample_non_overlap.csv.
 2. Define SECOND set = all images listed in interval_sample.csv.
 3. Build pairs: each of n FIRST images paired with every SECOND image => n * len(SECOND) pairs.
    For each pair compute ground-truth tri-class label using score difference & config.THETA.
 4. For each fold adapter (fold1..foldK) in the latest run directory:
      - Load model (quantized base + PEFT adapter) in eval mode.
      - Classify every pair (image1=sampled image, image2=second image) to predict one of {First., Second., Similar.}.
      - Store per-fold prediction.
      - Free model to release GPU memory.
 5. For each pair, perform majority vote over fold predictions (mode). If tie, fall back to order
    of config.ANSWER_SET (First. > Second. > Similar.) when counts equal.
 6. Compute overall accuracy of majority prediction vs ground-truth label. Report stats.
 7. Write CSV with columns: img1,img2,score1,score2,gt_label,pred_fold1,...,pred_foldK,maj_pred,correct

This script reuses some logic adapted from test_acc.py for prompt construction and generation.
"""
from __future__ import annotations
import csv
import os
import random
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import time
from datetime import datetime
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel
from configs.config import config
from model.model import build_inference_base
from data.data_utils import label_from_scores
from PIL import Image  # added

# ---------------------------- Configuration ----------------------------
N_FIRST_SAMPLE = 100  # <-- define n here; change as needed
NON_OVERLAP_CSV = Path('data/interval_sample_non_overlap.csv')
SECOND_SET_CSV = Path('data/interval_sample.csv')
OUT_CSV_NAME = f'score_eval_pairs_n{N_FIRST_SAMPLE}.csv'
# ----------------------------------------------------------------------

CKPT_ROOT = Path(__file__).resolve().parent / config.OUTPUT_DIR
IMAGE_DIR = Path(config.IMAGE_DIR)
SEED = config.SEED
RNG = random.Random(SEED)

# ---------------------------- Data structures -------------------------
@dataclass
class ScorePair:
    img1: str
    img2: str
    score1: float
    score2: float
    gt_label: str
    preds: List[str] = field(default_factory=list)  # per-fold predictions
    def majority_pred(self) -> str:
        if not self.preds:
            return ''
        cnt = Counter(self.preds)
        f = cnt.get(config.ANSWER_FIRST, 0)
        s = cnt.get(config.ANSWER_SECOND, 0)
        sim = cnt.get(config.ANSWER_SIMILAR, 0)
        # New rule: if any class appears >=3 times among the 5 folds, select it directly.
        if f >= 3 or s >= 3 or sim >= 3:
            if f >= 3:
                return config.ANSWER_FIRST
            if s >= 3:
                return config.ANSWER_SECOND
            return config.ANSWER_SIMILAR
        # Special tie patterns for 5 votes (2,2,1):
        if f == 2 and s == 2 and sim == 1:
            return config.ANSWER_SIMILAR
        # Additional symmetric tie patterns (2,1,2) and (1,2,2)
        if f == 2 and sim == 2 and s == 1:
            return config.ANSWER_FIRST
        if s == 2 and sim == 2 and f == 1:
            return config.ANSWER_SECOND
        # Standard majority / tie-break order First > Second > Similar
        max_c = max(f, s, sim)
        for lab in (config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR):
            if cnt.get(lab, 0) == max_c:
                return lab
        return config.ANSWER_SIMILAR  # fallback
    def correct(self) -> bool:
        return self.majority_pred() == self.gt_label

# ---------------------------- Utility functions -----------------------

def _latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run dirs in {root}")
    dirs.sort()
    return dirs[-1]

def _read_simple_csv(path: Path) -> List[Tuple[str,float]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r['img'], float(r['score'])))
    return rows

def _tokens_per_image(processor) -> int:
    patch_size = getattr(getattr(processor, 'image_processor', {}), 'patch_size', 14)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    if isinstance(patch_size, dict):
        patch_size = patch_size.get('height', 14)
    size_cfg = getattr(getattr(processor, 'image_processor', {}), 'size', {})
    if isinstance(size_cfg, dict):
        h = size_cfg.get('height') or size_cfg.get('shortest_edge') or config.IMAGE_SIZE
        w = size_cfg.get('width') or size_cfg.get('shortest_edge') or config.IMAGE_SIZE
    else:
        h = w = config.IMAGE_SIZE
    return (h // patch_size) * (w // patch_size)

def _build_prompt_prefix(tokens_per_image: int) -> str:
    placeholder = '<image>' * (tokens_per_image * 2)
    return placeholder + ', ' + config.QUESTION_TEXT + ' '

@torch.no_grad()
def _parse_answer(text: str) -> str:
    t = text.strip().lower()
    if 'first' in t:
        return config.ANSWER_FIRST
    if 'second' in t:
        return config.ANSWER_SECOND
    if 'similar' in t or 'same' in t:
        return config.ANSWER_SIMILAR
    if '1' in t and '2' not in t:
        return config.ANSWER_FIRST
    if '2' in t and '1' not in t:
        return config.ANSWER_SECOND
    return config.ANSWER_SIMILAR

@torch.no_grad()
def _classify_pair(model, tokenizer, prompt_ids, prompt_attn, pixel_pair) -> str:
    gen_ids = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_attn,
        pixel_values=pixel_pair,
        max_new_tokens=3,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_only = gen_ids[0, prompt_ids.size(1):]
    text = tokenizer.decode(gen_only, skip_special_tokens=True)
    return _parse_answer(text)

def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    cand = IMAGE_DIR / base
    if cand.exists():
        return str(cand)
    return path

def free_model(model):
    try:
        model.to('cpu')
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

# ---------------------------- Main logic ------------------------------

def build_pairs() -> List[ScorePair]:
    first_rows = _read_simple_csv(NON_OVERLAP_CSV)
    second_rows = _read_simple_csv(SECOND_SET_CSV)
    if N_FIRST_SAMPLE > len(first_rows):
        raise ValueError(f"Requested n={N_FIRST_SAMPLE} exceeds available first-set size={len(first_rows)}")
    sampled = RNG.sample(first_rows, N_FIRST_SAMPLE)
    pairs: List[ScorePair] = []
    for img1, s1 in sampled:
        for img2, s2 in second_rows:
            gt = label_from_scores(s1, s2)
            pairs.append(ScorePair(img1=img1, img2=img2, score1=s1, score2=s2, gt_label=gt))
    return pairs

def evaluate_pairs_across_folds(pairs: List[ScorePair]):
    run_dir = _latest_run_dir(CKPT_ROOT)
    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('fold')])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories under {run_dir}")
    print(f"Found {len(fold_dirs)} folds: {[p.name for p in fold_dirs]}")
    from utils.transform import basic_image_loader
    pil_loader = basic_image_loader()
    image_cache: Dict[str, Image.Image] = {}
    pixel_cache: Dict[str, torch.Tensor] = {}

    for fold_idx, fold_path in enumerate(fold_dirs, start=1):
        print(f"[Fold {fold_idx}] Loading adapter from {fold_path}")
        processor = AutoProcessor.from_pretrained(fold_path)
        tok = processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = build_inference_base()
        model = PeftModel.from_pretrained(base, fold_path, is_trainable=False)
        model.eval()
        device = next(model.parameters()).device
        # Pre-tokenize prompt once
        tpi = _tokens_per_image(processor)
        prompt_text = _build_prompt_prefix(tpi)
        prompt_tok = tok(prompt_text, add_special_tokens=True, return_tensors='pt')
        prompt_ids = prompt_tok.input_ids.to(device)
        prompt_attn = prompt_tok.attention_mask.to(device)
        prog = tqdm(pairs, desc=f"fold{fold_idx}", dynamic_ncols=True, unit='pair')
        for sp in prog:
            path1 = _resolve(sp.img1)
            path2 = _resolve(sp.img2)
            if path1 not in image_cache:
                image_cache[path1] = pil_loader(path1)
            if path2 not in image_cache:
                image_cache[path2] = pil_loader(path2)
            if path1 not in pixel_cache:
                pv1 = processor.image_processor([image_cache[path1]], return_tensors='pt').pixel_values[0]
                pixel_cache[path1] = pv1.cpu()
            if path2 not in pixel_cache:
                pv2 = processor.image_processor([image_cache[path2]], return_tensors='pt').pixel_values[0]
                pixel_cache[path2] = pv2.cpu()
            pixel_pair = torch.stack([pixel_cache[path1], pixel_cache[path2]], dim=0).to(device, dtype=torch.bfloat16)
            pred = _classify_pair(model, tok, prompt_ids, prompt_attn, pixel_pair)
            sp.preds.append(pred)
        # Free model before next
        del processor
        free_model(model)
    return fold_dirs

def summarize(pairs: List[ScorePair], fold_dirs: List[Path]):
    total = len(pairs)
    correct = sum(1 for p in pairs if p.correct())
    acc = correct / total if total else 0.0
    print(f"[Summary] n_pairs={total} accuracy={acc:.6f} ({correct}/{total})")
    # Per-class
    cls_tot = Counter(p.gt_label for p in pairs)
    cls_cor = Counter(p.gt_label for p in pairs if p.correct())
    for lab in config.ANSWER_SET:
        if cls_tot[lab]:
            print(f"  {lab:<7} {cls_cor[lab]}/{cls_tot[lab]} = {cls_cor[lab]/cls_tot[lab]:.6f}")
    # Write CSV
    run_dir = _latest_run_dir(CKPT_ROOT)
    out_path = run_dir / OUT_CSV_NAME
    fold_names = [p.name for p in fold_dirs]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['img1','img2','score1','score2','gt_label'] + [f'pred_{fn}' for fn in fold_names] + ['maj_pred','correct']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sp in pairs:
            row = {
                'img1': sp.img1,
                'img2': sp.img2,
                'score1': f"{sp.score1:.6f}",
                'score2': f"{sp.score2:.6f}",
                'gt_label': sp.gt_label,
                'maj_pred': sp.majority_pred(),
                'correct': int(sp.correct()),
            }
            for fn, pred in zip(fold_names, sp.preds):
                row[f'pred_{fn}'] = pred
            writer.writerow(row)
    print(f"[Write] Saved detailed results to {out_path}")

# ---------------------------- Entry point -----------------------------

def main():
    start = time.perf_counter()
    print(f"[Time] Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sampling n={N_FIRST_SAMPLE} from {NON_OVERLAP_CSV} and pairing with all in {SECOND_SET_CSV}")
    pairs = build_pairs()
    print(f"Constructed {len(pairs)} pairs (expected {N_FIRST_SAMPLE} * SECOND size)")
    fold_dirs = evaluate_pairs_across_folds(pairs)
    summarize(pairs, fold_dirs)
    dur = time.perf_counter() - start
    print(f"[Time] End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | total={dur/60:.2f} min")

if __name__ == '__main__':
    main()
