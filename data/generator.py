# -*- coding: utf-8 -*-
"""
LMOL Dataset Generator Module

This module implements the complete dataset generation pipeline for LMOL training.
It creates balanced, tri-class pairwise datasets from the SCUT-FBP5500 facial
attractiveness dataset using sophisticated sampling strategies.

Key Features:
- Tri-class balanced sampling (First/Second/Similar)
- Near-boundary sampling for Similar class
- K-fold cross-validation support
- Comprehensive evaluation dataset generation
- Robust error handling and validation

Sampling Strategy:
- First/Second classes: Standard random sampling
- Similar class: Mixed sampling with near-boundary emphasis
- Near-boundary sampling: Focuses on score differences near THETA
- Balanced output: Equal samples per class for training stability

The generator creates both training and evaluation datasets with proper
class balance and comprehensive coverage of the score space.
"""

import csv
import os
import random
from typing import Dict, List, Tuple
from configs.config import config

CSV_FIELDS = ["img1", "img2", "score1", "score2", "label"]

# ----------------------------
# I/O helpers
# ----------------------------
def read_labels_file(labels_path: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                score = float(parts[1])
            except ValueError:
                continue
            scores[name] = score
    if not scores:
        raise ValueError("No valid entries found in labels file.")
    return scores

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_pairs_csv(path: str, rows: List[Tuple[str, str, float, float, str]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for img1, img2, s1, s2, lab in rows:
            writer.writerow({
                "img1": img1,
                "img2": img2,
                "score1": f"{s1:.6f}",
                "score2": f"{s2:.6f}",
                "label": lab
            })

# ----------------------------
# Core tri-class logic
# ----------------------------
def label_from_scores(s1: float, s2: float) -> str:
    if abs(s1 - s2) <= config.THETA:
        return config.ANSWER_SIMILAR
    return config.ANSWER_FIRST if s1 > s2 else config.ANSWER_SECOND

def build_offline_balanced_pairs(
    images: List[str],
    scores: Dict[str, float],
    per_class: int,
    rng: random.Random,
) -> List[Tuple[str, str, float, float, str]]:
    """Tri-class balanced sampling: collect per_class samples for each label.
    For Similar class, uses near-boundary sampling with probability config.NEAR_RATIO.
    """
    n = len(images)
    if n < 2:
        raise ValueError("Need at least two images to build pairs.")
    
    buckets = {
        config.ANSWER_FIRST: [],
        config.ANSWER_SECOND: [],
        config.ANSWER_SIMILAR: [],
    }
    
    TRIAL_CAP = 10_000_000  # Increased to ensure enough time to find near-band pairs
    trials = 0
    
    # For Similar class near-boundary sampling
    near_trials = 0
    NEAR_TRIAL_CAP = 500_000  # Cap for near-boundary trials before fallback
    
    while (any(len(buckets[k]) < per_class for k in buckets)) and trials < TRIAL_CAP:
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            trials += 1
            continue
            
        f1, f2 = images[i], images[j]
        s1, s2 = scores[f1], scores[f2]
        score_diff = abs(s1 - s2)
        
        # Determine the label
        if score_diff <= config.THETA:
            # This is a potential Similar pair
            
            # For Similar class, check if we need near-boundary sampling
            if len(buckets[config.ANSWER_SIMILAR]) < per_class:
                # Decide whether to use near-boundary sampling based on NEAR_RATIO
                use_near_band = rng.random() < config.NEAR_RATIO
                
                if use_near_band:
                    # Only accept if within the tighter NEAR_BAND
                    if score_diff <= config.NEAR_BAND:
                        buckets[config.ANSWER_SIMILAR].append((f1, f2, s1, s2, config.ANSWER_SIMILAR))
                    else:
                        # Skip this pair, not in near band, but count as a near trial
                        near_trials += 1
                        if near_trials >= NEAR_TRIAL_CAP:
                            # Fallback to standard Similar rule if we've tried too many times
                            print(f"Warning: Reached near-band trial cap ({NEAR_TRIAL_CAP}). Falling back to standard Similar rule.")
                            buckets[config.ANSWER_SIMILAR].append((f1, f2, s1, s2, config.ANSWER_SIMILAR))
                else:
                    # Standard Similar rule
                    buckets[config.ANSWER_SIMILAR].append((f1, f2, s1, s2, config.ANSWER_SIMILAR))
        elif s1 > s2:
            # First class
            if len(buckets[config.ANSWER_FIRST]) < per_class:
                buckets[config.ANSWER_FIRST].append((f1, f2, s1, s2, config.ANSWER_FIRST))
        else:  # s2 > s1
            # Second class
            if len(buckets[config.ANSWER_SECOND]) < per_class:
                buckets[config.ANSWER_SECOND].append((f1, f2, s1, s2, config.ANSWER_SECOND))
                
        trials += 1
    
    if any(len(buckets[k]) < per_class for k in buckets):
        raise RuntimeError(
            f"Failed to construct balanced tri-class pairs. Got counts: "
            f"First={len(buckets[config.ANSWER_FIRST])}, "
            f"Second={len(buckets[config.ANSWER_SECOND])}, "
            f"Similar={len(buckets[config.ANSWER_SIMILAR])}. "
            f"Consider reducing per_class."
        )
    
    # Combine all buckets into a single list
    result = []
    for bucket in buckets.values():
        result.extend(bucket[:per_class])  # Ensure exactly per_class samples per label
    
    return result

# ----------------------------
# K-fold utilities
# ----------------------------
def kfold_split_images(img_names: List[str], k: int, rng: random.Random) -> List[List[str]]:
    pool = list(img_names)
    rng.shuffle(pool)
    folds: List[List[str]] = [[] for _ in range(k)]
    for idx, name in enumerate(pool):
        folds[idx % k].append(name)
    return folds

def prepend_dir(rows: List[Tuple[str, str, float, float, str]], image_dir: str) -> List[Tuple[str, str, float, float, str]]:
    out: List[Tuple[str, str, float, float, str]] = []
    for f1, f2, s1, s2, lab in rows:
        p1 = os.path.join(image_dir, f1) if image_dir else f1
        p2 = os.path.join(image_dir, f2) if image_dir else f2
        out.append((p1, p2, s1, s2, lab))
    return out

# ----------------------------
# Eval: all-pairs helpers
# ----------------------------
def n_choose_2(n: int) -> int:
    if n < 2:
        return 0
    return n * (n - 1) // 2

def write_eval_all_pairs_csv(
    path: str,
    images: List[str],
    scores: Dict[str, float],
    image_dir: str,
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        join = os.path.join
        n = len(images)
        for i in range(n):
            f1 = images[i]; s1 = scores[f1]; p1 = join(image_dir, f1) if image_dir else f1
            for j in range(i + 1, n):
                f2 = images[j]; s2 = scores[f2]; p2 = join(image_dir, f2) if image_dir else f2
                lab = label_from_scores(s1, s2)
                writer.writerow({
                    "img1": p1,
                    "img2": p2,
                    "score1": f"{s1:.6f}",
                    "score2": f"{s2:.6f}",
                    "label": lab
                })

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    random.seed(config.SEED)
    scores = read_labels_file(config.LABELS_PATH)
    img_names = sorted(scores.keys())
    ensure_dir(config.PAIRS_OUT_DIR)
    
    base_rng = random.Random(config.SEED)
    folds = kfold_split_images(img_names, config.KFOLDS, base_rng)
    
    for f_idx in range(config.KFOLDS):
        fold_id = f_idx + 1
        eval_imgs  = folds[f_idx]
        train_imgs = [x for i, fold in enumerate(folds) if i != f_idx for x in fold]
        
        rng_train = random.Random(config.SEED + 100 * fold_id)
        train_rows = build_offline_balanced_pairs(
            images=train_imgs,
            scores=scores,
            per_class=config.TRAIN_PER_CLASS,
            rng=rng_train,
        )
        train_rows = prepend_dir(train_rows, config.IMAGE_DIR)
        
        eval_total  = n_choose_2(len(eval_imgs))
        train_total = 3 * config.TRAIN_PER_CLASS  # 3 classes
        
        train_csv = os.path.join(config.PAIRS_OUT_DIR, f"train_fold{fold_id}_{train_total}.csv")
        eval_csv  = os.path.join(config.PAIRS_OUT_DIR, f"eval_fold{fold_id}_{eval_total}.csv")
        
        write_pairs_csv(train_csv, train_rows)
        write_eval_all_pairs_csv(
            path=eval_csv,
            images=eval_imgs,
            scores=scores,
            image_dir=config.IMAGE_DIR,
        )
        
        print(f"[Fold {fold_id}] Wrote: {train_csv} and {eval_csv}")
    
    print("Done. Update config.TRAIN_PAIRS_CSVS / EVAL_PAIRS_CSVS if paths differ.")

if __name__ == "__main__":
    main()
