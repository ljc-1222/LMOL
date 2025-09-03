# -*- coding: utf-8 -*-
"""
Load each fold adapter checkpoint and evaluate via a single generation per pair.
No loss / log-likelihood computation: the model is prompted once and its short
text output is parsed into one of {First., Second., Similar.}.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
import time
import gc
from functools import lru_cache
import random

import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from configs.config import config
from model.model import build_inference_base
from data.data_utils import read_pairs_csv, PairRecord, write_pairs_csv

# Derive checkpoint root from config
CKPT_ROOT = Path(__file__).resolve().parent / config.OUTPUT_DIR
DATA_PAIRS_DIR = Path(__file__).resolve().parent / "data" / "pairs"

# Number of pairs to sample per fold for evaluation. Set to 0 or None to use all pairs.
N_EVAL_SAMPLE = 1000 # <-- set this to desired n (e.g., 10000) to subsample

def _latest_run_dir(root: Path) -> Path:
    """Return the most recent timestamp-named run directory under root."""
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    cand = [p for p in root.iterdir() if p.is_dir()]
    if not cand:
        raise FileNotFoundError(f"No run directories inside {root}")
    # Sort lexicographically (timestamps yyyymmdd_HHMMSS) -> latest last
    cand.sort()
    return cand[-1]

# Remove find_eval_csv; replace with helper using config.EVAL_PAIRS_CSVS
def get_eval_csv_for_fold(fold_idx: int) -> Path:
    try:
        rel = config.EVAL_PAIRS_CSVS[fold_idx - 1]
    except IndexError:
        raise IndexError(f"No eval CSV configured for fold {fold_idx}")
    path = Path(__file__).resolve().parent / rel
    if not path.exists():
        raise FileNotFoundError(f"Eval CSV not found for fold {fold_idx}: {path}")
    return path

def load_single_fold(fold_path: Path) -> Tuple[PeftModel, AutoProcessor]:
    print(f"[Load] {fold_path.name} from {fold_path}")
    processor = AutoProcessor.from_pretrained(fold_path)
    tok = processor.tokenizer
    if (tok.pad_token is None):
        tok.pad_token = tok.eos_token
    base = build_inference_base()
    model = PeftModel.from_pretrained(base, fold_path, is_trainable=False)
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  params: total={total:,} trainable={trainable:,} ({trainable/total:.6f})")
    return model, processor

# ---------------------- Evaluation ----------------------

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
    placeholder = "<image>" * (tokens_per_image * 2)
    return placeholder + ", " + config.QUESTION_TEXT

@torch.no_grad()
def _parse_answer(text: str) -> str:
    t = text.strip().lower()
    if "first" in t:
        return config.ANSWER_FIRST
    if "second" in t:
        return config.ANSWER_SECOND
    if "similar" in t or "same" in t:
        return config.ANSWER_SIMILAR
    # fallback heuristic: if contains '1' choose First, '2' choose Second
    if '1' in t and '2' not in t:
        return config.ANSWER_FIRST
    if '2' in t and '1' not in t:
        return config.ANSWER_SECOND
    return config.ANSWER_SIMILAR  # conservative default

@torch.no_grad()
def classify_pair(model, tokenizer, prompt_ids, prompt_attn, pixel_pair) -> str:
    """Fast classify using pre-tokenized prompt and cached pixel tensors.
    pixel_pair: Tensor [2,3,H,W]
    """
    gen_ids = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_attn,
        pixel_values=pixel_pair,  # shape [2,3,H,W]
        max_new_tokens=3,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_only = gen_ids[0, prompt_ids.size(1):]
    text = tokenizer.decode(gen_only, skip_special_tokens=True)
    return _parse_answer(text)

@torch.no_grad()
def evaluate_fold(fold_name: str, model, processor, eval_csv: Path, fold_path: Path) -> Tuple[int, int]:
    start_ts = time.perf_counter()
    start_clock = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[Time] {fold_name} eval start: {start_clock}")
    pair_records = read_pairs_csv(str(eval_csv))
    # Optional random subsample for speed
    if N_EVAL_SAMPLE and N_EVAL_SAMPLE > 0 and len(pair_records) > N_EVAL_SAMPLE:
        rng = random.Random(config.SEED + hash(fold_name) % 100000)
        pair_records = rng.sample(pair_records, N_EVAL_SAMPLE)
    total = len(pair_records)
    correct = 0
    pred_records: List[PairRecord] = []

    from utils.transform import basic_image_loader
    pil_loader = basic_image_loader()

    # Pre-tokenize prompt once
    tpi = _tokens_per_image(processor)
    prompt_text = _build_prompt_prefix(tpi) + " "
    tokenizer = processor.tokenizer
    prompt_tok = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")
    prompt_ids = prompt_tok.input_ids.to(next(model.parameters()).device)
    prompt_attn = prompt_tok.attention_mask.to(next(model.parameters()).device)

    # Image / pixel caches
    image_cache: dict[str, 'Image.Image'] = {}
    pixel_cache: dict[str, torch.Tensor] = {}  # single image tensor [3,H,W] (cpu)

    def _resolve(p: str) -> str:
        if os.path.exists(p):
            return p
        base = os.path.basename(p)
        cand = os.path.join(config.IMAGE_DIR, base)
        if os.path.exists(cand):
            return cand
        return p

    def get_pil(path: str):
        if path not in image_cache:
            image_cache[path] = pil_loader(path)
        return image_cache[path]

    def get_pixel(path: str):
        if path not in pixel_cache:
            img = get_pil(path)
            # processor.image_processor expects list; take first element result
            pv = processor.image_processor([img], return_tensors="pt").pixel_values[0]  # [3,H,W]
            pixel_cache[path] = pv.cpu()
        return pixel_cache[path]

    prog = tqdm(pair_records, desc=f"{fold_name} eval", dynamic_ncols=True, unit="pair")
    device = next(model.parameters()).device
    for idx, pr in enumerate(prog):
        p1 = _resolve(pr.img1)
        p2 = _resolve(pr.img2)
        t1 = get_pixel(p1)
        t2 = get_pixel(p2)
        pixel_pair = torch.stack([t1, t2], dim=0).to(device, dtype=torch.bfloat16)  # [2,3,H,W]
        pred_label = classify_pair(model, tokenizer, prompt_ids, prompt_attn, pixel_pair)
        if pred_label == pr.label:
            correct += 1
        pred_records.append(PairRecord(img1=pr.img1, img2=pr.img2, score1=pr.score1, score2=pr.score2, label=pred_label))
        if (idx + 1) % 200 == 0:
            prog.set_postfix(acc=f"{correct/(idx+1):.4f}")
    acc = correct / total if total else 0.0
    dur = time.perf_counter() - start_ts
    out_csv = fold_path / f"pred_{eval_csv.name}"
    write_pairs_csv(str(out_csv), pred_records)
    print(f"[Result] {fold_name}: accuracy={acc:.6f} ({correct}/{total}) | time={dur/60:.2f} min | wrote={out_csv}")
    print(f"[Cache] images={len(image_cache)} pixel_tensors={len(pixel_cache)}")
    return correct, total

def free_model(model):
    """Aggressively release a loaded model to free GPU + host RAM."""
    try:
        if hasattr(model, 'peft_config'):  # PEFT wrapper
            # Move to CPU first (safer for some allocators)
            try:
                model.to('cpu')
            except Exception:
                pass
        # Drop large submodules references explicitly
        for attr in ['model', 'base_model', 'base_model.model']:
            if hasattr(model, attr):
                try:
                    setattr(model, attr, None)
                except Exception:
                    pass
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

def main():
    run_start = time.perf_counter()
    print(f"[Time] Script start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_dir = _latest_run_dir(CKPT_ROOT)
    print(f"Using latest run directory: {run_dir}")

    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("fold")])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories in {run_dir}")

    grand_correct = 0
    grand_total = 0

    for fold_idx, fold_path in enumerate(fold_dirs, start=1):
        eval_csv = get_eval_csv_for_fold(fold_idx)
        print(f"Evaluating {fold_path.name} on {eval_csv}")
        model, processor = load_single_fold(fold_path)
        c, t = evaluate_fold(fold_path.name, model, processor, eval_csv, fold_path)
        grand_correct += c
        grand_total += t
        # Release memory before next fold
        del processor
        free_model(model)
    if grand_total:
        print(f"[Overall] accuracy={grand_correct/grand_total:.6f} ({grand_correct}/{grand_total})")
    total_dur = time.perf_counter() - run_start
    print(f"[Time] Script end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | total={total_dur/60:.2f} min")

if __name__ == "__main__":
    main()
