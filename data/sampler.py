# -*- coding: utf-8 -*-
"""interval_sampler.py

Generate two CSV datasets based on labels.txt using 0.1-wide score intervals.
1) interval_sample.csv : up to 10 (or all if <10) images per interval; columns img,score
2) interval_sample_non_overlap.csv : ALL remaining images (no cap) not in set 1; columns img,score

Both CSVs are ordered globally by ascending score (tie -> filename).

Usage (from repo root):
  python -m data.interval_sampler

Config dependencies: uses configs.config for LABELS_PATH and SEED.
"""
import csv
import math
import os
import random
from typing import Dict, List, Tuple, Set

from configs.config import config
from data.generator import read_labels_file

OUTPUT_CSV = "data/interval_sample.csv"
SECOND_OUTPUT_CSV = "data/interval_sample_non_overlap.csv"
INTERVAL_WIDTH = 0.1
MAX_PER_INTERVAL = 10


def interval_key(score: float) -> Tuple[float, float]:
    """Return (lower, upper) bounds for the 0.1 interval containing score.
    Intervals are [lower, upper). The very max score will still map correctly.
    Example: 2.444 -> (2.4, 2.5)
    """
    lower = math.floor(score * 10) / 10.0
    upper = round(lower + INTERVAL_WIDTH, 10)  # avoid FP drift
    return lower, upper


def sample_primary(scores: Dict[str, float], rng: random.Random) -> List[Tuple[str, float]]:
    # Bucket filenames by interval
    buckets: Dict[Tuple[float, float], List[Tuple[str, float]]] = {}
    for fname, sc in scores.items():
        key = interval_key(sc)
        buckets.setdefault(key, []).append((fname, sc))
    # Deterministic ordering inside buckets
    for lst in buckets.values():
        lst.sort(key=lambda x: x[0])
    selected: List[Tuple[str, float]] = []
    for key in sorted(buckets.keys()):  # intervals ascending
        lst = buckets[key]
        if len(lst) <= MAX_PER_INTERVAL:
            chosen = lst
        else:
            chosen = rng.sample(lst, MAX_PER_INTERVAL)
        selected.extend(chosen)
    # Global ordering
    selected.sort(key=lambda x: (x[1], x[0]))
    return selected


def sample_secondary(scores: Dict[str, float], used: Set[str], rng: random.Random) -> List[Tuple[str, float]]:  # rng kept for interface consistency
    """Return ALL remaining (non-overlapping) images, no per-interval cap."""
    buckets: Dict[Tuple[float, float], List[Tuple[str, float]]] = {}
    for fname, sc in scores.items():
        if fname in used:
            continue
        key = interval_key(sc)
        buckets.setdefault(key, []).append((fname, sc))
    for lst in buckets.values():
        lst.sort(key=lambda x: x[0])
    selected: List[Tuple[str, float]] = []
    for key in sorted(buckets.keys()):
        lst = buckets[key]
        if not lst:
            continue
        # No sampling: take all remaining images in this interval
        selected.extend(lst)
    selected.sort(key=lambda x: (x[1], x[0]))
    return selected


def write_primary(selected: List[Tuple[str, float]]):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img", "score"])
        writer.writeheader()
        for fname, sc in selected:
            writer.writerow({"img": fname, "score": f"{sc:.6f}"})


def write_secondary(selected: List[Tuple[str, float]]):
    # Write secondary with score column
    os.makedirs(os.path.dirname(SECOND_OUTPUT_CSV), exist_ok=True)
    with open(SECOND_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img", "score"])
        writer.writeheader()
        for fname, sc in selected:
            writer.writerow({"img": fname, "score": f"{sc:.6f}"})


def main() -> None:
    scores: Dict[str, float] = read_labels_file(config.LABELS_PATH)
    rng_primary = random.Random(config.SEED)
    primary = sample_primary(scores, rng_primary)
    write_primary(primary)

    used_names = {fname for fname, _ in primary}
    rng_secondary = random.Random(config.SEED + 999)
    secondary = sample_secondary(scores, used_names, rng_secondary)
    write_secondary(secondary)

    print(
        f"Wrote {OUTPUT_CSV} ({len(primary)} rows) and {SECOND_OUTPUT_CSV} ({len(secondary)} rows)"
    )


if __name__ == "__main__":
    main()
