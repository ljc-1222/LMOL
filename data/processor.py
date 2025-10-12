# -*- coding: utf-8 -*-
"""
LMOL Data Utilities Module

This module provides comprehensive data processing utilities for the LMOL project.
It handles CSV reading, path resolution, label processing, and data validation.

Key Features:
- Robust CSV reading with automatic label generation
- Flexible path resolution with multiple fallback strategies
- Comprehensive data validation and error handling
- Support for various CSV formats and configurations
- Automatic label generation from score differences

Data Structures:
- PairRecord: Represents a single image pair with scores and labels
- Label generation: Automatic tri-class labeling based on score differences
- Path resolution: Multiple strategies for finding image files

The module is designed to handle various data configurations and provide
robust error handling for production use.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Tuple
import csv
from configs.config import config

@dataclass
class PairRecord:
    """
    Data structure representing a single image pair for LMOL training.
    
    This class encapsulates all information needed for a single training
    sample: two images, their attractiveness scores, and the comparison label.
    
    Attributes:
        img1: Path to the first image
        img2: Path to the second image
        score1: Attractiveness score for the first image
        score2: Attractiveness score for the second image
        label: Comparison label ("First.", "Second.", or "Similar.")
    """
    img1: str
    img2: str
    score1: float
    score2: float
    label: str

def label_from_scores(s1: float, s2: float, theta: float) -> str:
    """
    Generate tri-class label based on score difference.
    
    This function implements the LMOL labeling strategy:
    - If |score1 - score2| <= theta: "Similar."
    - If score1 > score2 + theta: "First."
    - If score2 > score1 + theta: "Second."
    
    Args:
        s1: Attractiveness score for first image
        s2: Attractiveness score for second image
        theta: Threshold for determining similarity
        
    Returns:
        Canonical answer string ("First.", "Second.", or "Similar.")
    """
    d = abs(s1 - s2)
    if d <= theta:
        return config.ANSWER_SIMILAR
    return config.ANSWER_FIRST if s1 > s2 else config.ANSWER_SECOND

def _try_candidates(rel_path: str) -> str | None:
    """
    Try to resolve a relative path using multiple candidate strategies.
    
    This function implements a robust path resolution strategy:
    1. Check if path exists as-is
    2. Try stripping common prefixes
    3. Test multiple root directories
    4. Return absolute path if found
    
    Args:
        rel_path: Relative path to resolve
        
    Returns:
        Absolute path if found, None otherwise
    """
    # 1) Check if path exists as-is (absolute or relative to CWD)
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return str(p.resolve())
    if p.exists():
        return str(p.resolve())

    # 2) Try stripping known prefixes
    stripped_variants = {rel_path}
    for pref in config.STRIP_PREFIXES:
        if rel_path.startswith(pref):
            stripped_variants.add(rel_path[len(pref):])

    # 3) Build candidate roots: IMAGE_DIR (if set), then DATA_ROOTS
    roots: List[Path] = []
    if getattr(config, "IMAGE_DIR", None):
        roots.append(Path(config.IMAGE_DIR))
        roots.append(Path(config.IMAGE_DIR).parent)
    roots.extend(Path(r) for r in config.DATA_ROOTS)

    # 4) Test all (root / variant) combinations
    for root in roots:
        for v in stripped_variants:
            cand = (root / v)
            if cand.exists():
                return str(cand.resolve())

    # 5) As a last resort, try from project root without any root
    for v in stripped_variants:
        cand = Path(v)
        if cand.exists():
            return str(cand.resolve())

    return None

def _resolve_path(p: str) -> str:
    """
    Robust path resolver with comprehensive fallback strategies.
    
    This function implements a multi-step path resolution strategy:
    1. Check if path exists as-is (absolute or relative)
    2. Try multiple candidate roots and prefix-stripped variants
    3. Return best-guess absolute path for consistent error messages
    
    Args:
        p: Path string to resolve
        
    Returns:
        Resolved absolute path string
        
    Note:
        This function is designed to provide consistent behavior
        and clear error messages even when paths cannot be resolved.
    """
    # Fast path: check if absolute path exists
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return str(pp.resolve())

    # Try comprehensive candidate resolution
    cand = _try_candidates(p)
    if cand is not None:
        return cand

    # Best guess fallback (legacy behavior)
    # This ensures deterministic error messages at load time
    base = Path(config.IMAGE_DIR) if getattr(config, "IMAGE_DIR", None) else Path(".")
    return str((base / p).resolve())

def read_pairs_csv(csv_path: str) -> List[PairRecord]:
    """
    Read pairs CSV file and create PairRecord objects.
    
    This function reads CSV files containing image pairs and their scores,
    automatically generating labels if they are missing or invalid.
    
    CSV Format:
        img1,img2,score1,score2,label
        
    Where:
        - img1, img2: Image file paths
        - score1, score2: Attractiveness scores
        - label: Optional comparison label (auto-generated if missing)
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of PairRecord objects with resolved paths and valid labels
        
    Raises:
        FileNotFoundError: If CSV file cannot be found
        ValueError: If CSV format is invalid
    """
    out: List[PairRecord] = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract scores
            s1 = float(row.get("score1", 0))
            s2 = float(row.get("score2", 0))
            
            # Get or generate label
            lbl = (row.get("label") or "").strip()
            if lbl not in config.ANSWER_SET:
                lbl = label_from_scores(s1, s2, config.THETA)
            
            # Create PairRecord with resolved paths
            out.append(PairRecord(
                img1=_resolve_path(row["img1"]),
                img2=_resolve_path(row["img2"]),
                score1=s1, score2=s2, label=lbl
            ))
    
    return out

def write_pairs_csv(csv_path: str, rows: Iterable[Tuple[str, str, float, float, str]]) -> None:
    """
    Write pairs data to CSV file in the expected format.
    
    This function writes image pair data to a CSV file with the standard
    LMOL format: img1, img2, score1, score2, label.
    
    Args:
        csv_path: Path where to write the CSV file
        rows: Iterable of tuples containing (img1, img2, score1, score2, label)
        
    Raises:
        OSError: If file cannot be written
    """
    # Ensure parent directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img1", "img2", "score1", "score2", "label"])
        w.writerows(rows)
