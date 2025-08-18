from __future__ import annotations
import csv
import shutil
import sys
from pathlib import Path
from collections import Counter

# ---------- 0. Remove unused awq (causes deprecation splash) ----------
try:
    import importlib.util as _iu
    if _iu.find_spec("awq") or _iu.find_spec("autoawq"):
        import subprocess, sys as _sys
        subprocess.run(
            [_sys.executable, "-m", "pip", "uninstall", "-y", "awq", "autoawq"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
except Exception:
    pass  # best effort

import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from configs import config
from data.data_utils import label_reading, normalize_filename

# ---------- 1. Paths (edit if needed) ----------------------------------
CKPT_DIR      = Path("model/checkpoints/Qwen2.5-VL-3B/20250723_124027").resolve()
TARGET_IMG    = Path("data/raw/SCUT-FBP5500/Images/AF346.jpg").resolve()
REFERENCE_CSV = Path("data/test_images.csv").resolve()
REF_ROOT      = Path("data/raw/SCUT-FBP5500/Images").resolve()
BATCH_SIZE    = 4
SIM_THRESHOLD = 0.2   


# ---------- 2. Pre‑fix old video preprocessor file ---------------------
def _fix_video_preprocessor(folder: Path):
    pp = folder / "preprocessor.json"
    vp = folder / "video_preprocessor.json"
    if pp.exists() and not vp.exists():
        shutil.copy(pp, vp)

_fix_video_preprocessor(CKPT_DIR)
_fix_video_preprocessor(Path.home() / ".cache/huggingface/hub")


# ---------- 3. Load model / tokenizer / processor ----------------------
BASE = config.MODEL_ID
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, str(CKPT_DIR))
model.eval()

tok  = AutoTokenizer.from_pretrained(BASE,  trust_remote_code=True)
proc = AutoProcessor.   from_pretrained(BASE, trust_remote_code=True, use_fast=True)

# ---------- 5. Load scores (rounded 0.1) -------------------------------
score_dict = {normalize_filename(f): s for f, s in label_reading()}

def get_score(path: Path) -> float:
    name = normalize_filename(path.name)
    if name not in score_dict:
        sys.exit(f"No score found for {name}")
    return score_dict[name]


# ---------- 6. Build chat message --------------------------------------
def make_msg(uri1: str, uri2: str):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": uri1},
            {"type": "image", "image": uri2},
            {"type": "text",  "text": config.QUESTION_TEXT},
        ],
    }]


# ---------- 7. Batch inference -----------------------------------------
def infer_batch(pairs):
    msgs = [make_msg(a, b) for a, b in pairs]
    prompts = [
        proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in msgs
    ]
    vision = [process_vision_info(m)[0] for m in msgs]

    inputs = proc(text=prompts, images=vision,
                  return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
        )

    out_texts = []
    for i in range(len(pairs)):
        new_ids = outs[i][len(inputs.input_ids[i]):]
        out_texts.append(tok.decode(new_ids, skip_special_tokens=True).strip())
    return out_texts


# ---------- 8. Prepare reference list ----------------------------------
if not REFERENCE_CSV.is_file():
    sys.exit(f"CSV not found: {REFERENCE_CSV}")
with REFERENCE_CSV.open(newline="") as f:
    reader = csv.DictReader(f)
    col = (set(reader.fieldnames or []) &
           {"Image", "image", "filename", "file"}).pop()
    reference_files = [row[col] for row in reader]

target_uri = TARGET_IMG.as_uri()
target_score = get_score(TARGET_IMG)

counter   = Counter()
correct   = 0
total_cmp = 0
canon     = {k: v.rstrip(".") for k, v in config.ANSWER_MAP.items()}


# ---------- 9. Main loop with tqdm -------------------------------------
for idx in tqdm(range(0, len(reference_files), BATCH_SIZE),
                desc="Comparing", unit="img", colour="cyan"):
    batch_names = reference_files[idx: idx + BATCH_SIZE]
    uri_pairs, ref_paths = [], []
    for name in batch_names:
        p = Path(name)
        if not p.is_file():
            p = REF_ROOT / name
        if not p.is_file():
            sys.exit(f"Reference image missing: {name}")
        uri_pairs.append((target_uri, p.resolve().as_uri()))
        ref_paths.append(p)

    preds = infer_batch(uri_pairs)
    for pred_txt, ref_path in zip(preds, ref_paths):
        pred = next((canon[k] for k in canon if pred_txt.lower().startswith(k)), "Other")
        counter[pred] += 1

        # ----- ground‑truth via score difference -----
        ref_score = get_score(ref_path)
        diff = abs(target_score - ref_score)
        if diff <= SIM_THRESHOLD:
            gt = "Similar"
        elif target_score > ref_score:
            gt = "First"
        else:
            gt = "Second"

        if pred == gt:
            correct += 1
        total_cmp += 1


# ---------- 10. Summary -------------------------------------------------
print("\n----- Summary -----")
print(f"Total comparisons: {total_cmp}")
for lbl in ["First", "Second", "Similar", "Other"]:
    if counter[lbl]:
        print(f"{lbl}: {counter[lbl]}")

acc = correct / total_cmp * 100 if total_cmp else 0
print(f"\nAccuracy (threshold {SIM_THRESHOLD}): {acc:.2f}%")
