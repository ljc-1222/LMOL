# -*- coding: utf-8 -*-

import csv
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from config import config

@dataclass
class PairRecord:
    img1: str
    img2: str
    score1: float
    score2: float
    label: str  # "First." | "Second." | "Similar."

def label_from_scores(score1: float, score2: float, theta: float = None) -> str:
    """Create a tri-class label using the LMOL/UOL rule with threshold theta=0.2."""
    if theta is None:
        theta = config.THETA
    diff = score1 - score2
    if abs(diff) <= theta:
        return config.ANSWER_SIMILAR
    return config.ANSWER_FIRST if diff > 0 else config.ANSWER_SECOND

def read_pairs_csv(csv_path: str) -> List[PairRecord]:
    """
    Read an offline-constructed CSV of pairs.
    Expected columns: img1,img2,score1,score2,label
      - If 'label' is empty or missing, it will be recomputed using theta=0.2.
    """
    pairs: List[PairRecord] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s1 = float(row["score1"])
            s2 = float(row["score2"])
            if "label" in row and row["label"]:
                label = row["label"].strip()
            else:
                label = label_from_scores(s1, s2, config.THETA)
            pairs.append(PairRecord(
                img1=row["img1"],
                img2=row["img2"],
                score1=s1,
                score2=s2,
                label=label
            ))
    return pairs

def write_pairs_csv(csv_path: str, pairs: List[PairRecord]) -> None:
    """Write pairs to CSV in the expected schema."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img1", "img2", "score1", "score2", "label"])
        writer.writeheader()
        for p in pairs:
            writer.writerow({
                "img1": p.img1, "img2": p.img2,
                "score1": f"{p.score1:.6f}", "score2": f"{p.score2:.6f}",
                "label": p.label
            })

def build_offline_balanced_pairs(
    images: List[str],
    scores: Dict[str, float],
    per_class_M: int,
    seed: int = 42,
    theta: float = None
) -> List[PairRecord]:
    """
    Construct an offline balanced pairs list as required by LMOL:
    - Three classes: First., Second., Similar.
    - Each class gets exactly per_class_M samples.
    - The label rule uses theta=0.2.
    """
    if theta is None:
        theta = config.THETA
    rng = random.Random(seed)
    n = len(images)
    assert n >= 2, "Need at least two images to construct pairs."

    buckets = {
        config.ANSWER_FIRST: [],
        config.ANSWER_SECOND: [],
        config.ANSWER_SIMILAR: []
    }

    trials = 0
    # Try until each bucket reaches M or we exceed a safe trial cap.
    while any(len(buckets[k]) < per_class_M for k in buckets.keys()) and trials < 5_000_000:
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        img1, img2 = images[i], images[j]
        s1, s2 = scores[img1], scores[img2]
        label = label_from_scores(s1, s2, theta)
        # Enforce balance
        if len(buckets[label]) < per_class_M:
            buckets[label].append(PairRecord(img1, img2, s1, s2, label))
        trials += 1

    # Concatenate in fixed order for reproducibility
    balanced = buckets[config.ANSWER_FIRST] + buckets[config.ANSWER_SECOND] + buckets[config.ANSWER_SIMILAR]
    return balanced
