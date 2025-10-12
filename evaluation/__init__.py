# -*- coding: utf-8 -*-
"""
LMOL Evaluation Module

This module provides all evaluation-related functionality for the LMOL project:
- Main evaluation logic and metrics
- Model loading and inference
- Evaluation utilities and helpers

Key Components:
- evaluator: Core evaluation logic and model inference
- metrics: Evaluation metrics and scoring
- main: Evaluation orchestration and reporting
"""

# Lazy imports to avoid dependency issues
def load_single_fold(*args, **kwargs):
    """Lazy import of load_single_fold."""
    from .evaluator import load_single_fold as _load_single_fold
    return _load_single_fold(*args, **kwargs)

def evaluate_fold(*args, **kwargs):
    """Lazy import of evaluate_fold."""
    from .evaluator import evaluate_fold as _evaluate_fold
    return _evaluate_fold(*args, **kwargs)

def free_model(*args, **kwargs):
    """Lazy import of free_model."""
    from .evaluator import free_model as _free_model
    return _free_model(*args, **kwargs)

def _tokens_per_image(*args, **kwargs):
    """Lazy import of _tokens_per_image."""
    from .evaluator import _tokens_per_image as __tokens_per_image
    return __tokens_per_image(*args, **kwargs)

def _build_prompt_prefix(*args, **kwargs):
    """Lazy import of _build_prompt_prefix."""
    from .evaluator import _build_prompt_prefix as __build_prompt_prefix
    return __build_prompt_prefix(*args, **kwargs)

def classify_pair(*args, **kwargs):
    """Lazy import of classify_pair."""
    from .evaluator import classify_pair as _classify_pair
    return _classify_pair(*args, **kwargs)

def plot_and_save_cm(*args, **kwargs):
    """Lazy import of plot_and_save_cm."""
    from .metrics import plot_and_save_cm as _plot_and_save_cm
    return _plot_and_save_cm(*args, **kwargs)

def evaluation_main(*args, **kwargs):
    """Lazy import of evaluation_main."""
    from .main import evaluation_main as _evaluation_main
    return _evaluation_main(*args, **kwargs)

__all__ = [
    "load_single_fold",
    "evaluate_fold", 
    "free_model",
    "_tokens_per_image",
    "_build_prompt_prefix",
    "classify_pair",
    "plot_and_save_cm",
    "evaluation_main",
]
