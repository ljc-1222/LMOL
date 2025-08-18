# -*- coding: utf-8 -*-

"""
dataset_generator.py

This script generates offline, class-balanced pairwise datasets for LMOL/UOL-style
order learning from a plain labels file. No argument parser is used; all settings
are defined as constants below.

Strict alignment with UOL/LMOL and your project's IO format:
- Threshold theta=0.2 to define "Similar." (≈).
- Exactly three label strings: "First.", "Second.", "Similar."
- CSV schema exactly: img1,img2,score1,score2,label
- Output files per fold (K-fold on images):
    ./pairs/train_fold{fold}_{N}.csv  (N = 3 * TRAIN_PER_CLASS)
    ./pairs/eval_fold{fold}_{M}.csv   (M = 3 * EVAL_PER_CLASS)
- Train pairs are sampled only from train images of the fold, eval pairs only from
  eval images of the fold (no leakage).
"""

import csv
import os
import random
from typing import Dict, List, Tuple

# ----------------------------
# Hard-coded settings
# ----------------------------
LABELS_PATH = "./labels.txt"     # a text file with lines "<filename> <score>"
IMAGE_DIR   = "./images"         # prefix to prepend to filenames in CSV; "" means filenames only
OUT_DIR     = "./pairs"          # where CSVs will be written

KFOLDS = 5                       # number of folds
SEED   = 42                      # global seed for shuffling and sampling

# Per-class pair counts (total = 3 * per_class)
TRAIN_PER_CLASS = 30_000         # 30k per class -> 90k total per fold (LMOL)
EVAL_PER_CLASS  = 1_000          # 1k per class  -> 3k total per fold

# Tri-class threshold as in UOL/LMOL: |s1 - s2| <= THETA => "Similar."
THETA = 0.2

# Label literals (must be exact)
ANSWER_FIRST   = "First."
ANSWER_SECOND  = "Second."
ANSWER_SIMILAR = "Similar."

CSV_FIELDS = ["img1", "img2", "score1", "score2", "label"]


# ----------------------------
# I/O helpers
# ----------------------------
def read_labels_file(labels_path: str) -> Dict[str, float]:
    """Read 'labels.txt' where each non-empty line is '<filename> <score>'."""
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
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def write_pairs_csv(path: str, rows: List[Tuple[str, str, float, float, str]]) -> None:
    """Write rows to CSV with exact schema required by your project."""
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
# Core UOL/LMOL logic
# ----------------------------
def label_from_scores(s1: float, s2: float, theta: float) -> str:
    """Return one of 'First.' | 'Second.' | 'Similar.' using the LMOL/UOL rule."""
    diff = s1 - s2
    if abs(diff) <= theta:
        return ANSWER_SIMILAR
    return ANSWER_FIRST if diff > 0 else ANSWER_SECOND


def build_offline_balanced_pairs(
    images: List[str],
    scores: Dict[str, float],
    per_class: int,
    theta: float,
    rng: random.Random,
) -> List[Tuple[str, str, float, float, str]]:
    """
    Construct a class-balanced list of pairwise samples:
      - Sample (i != j) uniformly at random inside 'images'.
      - Assign tri-class labels with theta.
      - Collect exactly 'per_class' samples for each of the three classes.
    """
    n = len(images)
    if n < 2:
        raise ValueError("Need at least two images to build pairs.")

    buckets: Dict[str, List[Tuple[str, str, float, float, str]]] = {
        ANSWER_FIRST: [],
        ANSWER_SECOND: [],
        ANSWER_SIMILAR: [],
    }

    TRIAL_CAP = 10_000_000
    trials = 0

    while (len(buckets[ANSWER_FIRST])  < per_class or
           len(buckets[ANSWER_SECOND]) < per_class or
           len(buckets[ANSWER_SIMILAR])< per_class) and trials < TRIAL_CAP:

        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            trials += 1
            continue

        f1, f2 = images[i], images[j]
        s1, s2 = scores[f1], scores[f2]
        lab = label_from_scores(s1, s2, theta)

        # Enforce strict balance
        if len(buckets[lab]) < per_class:
            buckets[lab].append((f1, f2, s1, s2, lab))

        trials += 1

    if any(len(buckets[k]) < per_class for k in buckets):
        raise RuntimeError(
            "Failed to construct balanced pairs. "
            f"Got counts: First={len(buckets[ANSWER_FIRST])}, "
            f"Second={len(buckets[ANSWER_SECOND])}, "
            f"Similar={len(buckets[ANSWER_SIMILAR])}. "
            "Consider reducing per_class or verifying score distribution."
        )

    # Fixed concatenation order for reproducibility
    rows = buckets[ANSWER_FIRST] + buckets[ANSWER_SECOND] + buckets[ANSWER_SIMILAR]
    return rows


# ----------------------------
# K-fold utilities
# ----------------------------
def kfold_split_images(img_names: List[str], k: int, rng: random.Random) -> List[List[str]]:
    """Shuffle and split image names into k folds as evenly as possible."""
    pool = list(img_names)
    rng.shuffle(pool)
    folds: List[List[str]] = [[] for _ in range(k)]
    for idx, name in enumerate(pool):
        folds[idx % k].append(name)
    return folds


def prepend_dir(rows: List[Tuple[str, str, float, float, str]], image_dir: str) -> List[Tuple[str, str, float, float, str]]:
    """Prepend IMAGE_DIR to image filenames in rows."""
    out: List[Tuple[str, str, float, float, str]] = []
    for f1, f2, s1, s2, lab in rows:
        p1 = os.path.join(image_dir, f1) if image_dir else f1
        p2 = os.path.join(image_dir, f2) if image_dir else f2
        out.append((p1, p2, s1, s2, lab))
    return out


# ----------------------------
# Main (no argparse)
# ----------------------------
def main() -> None:
    # Seeds
    random.seed(SEED)

    # Read labels
    scores = read_labels_file(LABELS_PATH)
    img_names = sorted(scores.keys())

    # Prepare out dir
    ensure_dir(OUT_DIR)

    # Build folds
    base_rng = random.Random(SEED)
    folds = kfold_split_images(img_names, KFOLDS, base_rng)

    # For each fold, build train/eval rows and write to CSV
    for f_idx in range(KFOLDS):
        fold_id = f_idx + 1

        eval_imgs  = folds[f_idx]
        train_imgs = [x for i, fold in enumerate(folds) if i != f_idx for x in fold]

        # Use different RNGs per fold/split for reproducibility
        rng_train = random.Random(SEED + 100 * fold_id)
        rng_eval  = random.Random(SEED + 100 * fold_id + 1)

        train_rows = build_offline_balanced_pairs(
            images=train_imgs,
            scores=scores,
            per_class=TRAIN_PER_CLASS,
            theta=THETA,
            rng=rng_train,
        )
        eval_rows = build_offline_balanced_pairs(
            images=eval_imgs,
            scores=scores,
            per_class=EVAL_PER_CLASS,
            theta=THETA,
            rng=rng_eval,
        )

        # Prepend IMAGE_DIR so CSV stores absolute/relative paths expected by your loader
        train_rows = prepend_dir(train_rows, IMAGE_DIR)
        eval_rows  = prepend_dir(eval_rows,  IMAGE_DIR)

        # Filenames as required by your project
        train_total = 3 * TRAIN_PER_CLASS
        eval_total  = 3 * EVAL_PER_CLASS
        train_csv = os.path.join(OUT_DIR, f"train_fold{fold_id}_{train_total}.csv")
        eval_csv  = os.path.join(OUT_DIR, f"eval_fold{fold_id}_{eval_total}.csv")

        write_pairs_csv(train_csv, train_rows)
        write_pairs_csv(eval_csv,  eval_rows)

        print(f"[Fold {fold_id}] Wrote: {train_csv} and {eval_csv}")

    print("Done. You can now set config.TRAIN_PAIRS_CSV to one of the generated train_fold*.csv")


if __name__ == "__main__":
    main()
