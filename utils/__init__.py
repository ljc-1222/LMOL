# -*- coding: utf-8 -*-
"""
LMOL Utilities Module

This module provides all utility functionality for the LMOL project:
- Constants and configuration
- Seed management for reproducibility
- Scoring algorithms (Bradley-Terry)
- I/O utilities

Key Components:
- constants: Project-wide constants
- seed: Seed management for reproducibility
- scoring: Bradley-Terry scoring algorithm
- io: I/O utilities and helpers
"""

from .constants import *
from .seed import set_seed, get_dataloader_kwargs
from .scoring import bt_score
from .io import resolve_path, ensure_exists

__all__ = [
    # Constants
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_IMAGE_SIZE", 
    "VISION_DIM",
    "TEXT_HIDDEN_DIM",
    "IGNORE_INDEX",
    "PADDING_VALUE",
    "HEADER_SEPARATOR_LENGTH",
    "PROGRESS_UPDATE_INTERVAL",
    "DEFAULT_EVAL_SAMPLES",
    "BRADLEY_TERRY_SEARCH_POINTS",
    "BRADLEY_TERRY_MIN_SCORE",
    "BRADLEY_TERRY_MAX_SCORE",
    "DEFAULT_GRADIENT_CLIP_NORM",
    "DEFAULT_EPSILON",
    "DEFAULT_CSV_ENCODING",
    "DEFAULT_CSV_NEWLINE",
    "LORA_TARGET_MODULES",
    "DEFAULT_BIAS_TYPE",
    "DEFAULT_TASK_TYPE",
    "DEFAULT_IMAGE_CHANNELS",
    "DEFAULT_IMAGE_DTYPE",
    # Seed management
    "set_seed",
    "get_dataloader_kwargs",
    # Scoring
    "bt_score",
    # I/O utilities
    "resolve_path",
    "ensure_exists",
]
