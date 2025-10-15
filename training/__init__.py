# -*- coding: utf-8 -*-
"""
LMOL Training Module

This module provides all training-related functionality for the LMOL project:
- Classification trainer with swap consistency regularization
- Training callbacks for model saving
- Optimizer setup with dual learning rates
- Main training orchestration

Key Components:
- classification_trainer: LMOLClassificationTrainer implementation
- callbacks: Training callbacks for model saving and monitoring
- optimizer: Optimizer setup with parameter grouping
- main: Main training orchestration and fold management
"""

# Lazy imports to avoid dependency issues

def LMOLClassificationTrainer(*args, **kwargs):
    """Lazy import of LMOLClassificationTrainer."""
    from .classification_trainer import LMOLClassificationTrainer as _LMOLClassificationTrainer
    return _LMOLClassificationTrainer(*args, **kwargs)

def SaveBestTrainingLossCallback(*args, **kwargs):
    """Lazy import of SaveBestTrainingLossCallback."""
    from .callbacks import SaveBestTrainingLossCallback as _SaveBestTrainingLossCallback
    return _SaveBestTrainingLossCallback(*args, **kwargs)

def group_parameters_for_optimizer(*args, **kwargs):
    """Lazy import of group_parameters_for_optimizer."""
    from .optimizer import group_parameters_for_optimizer as _group_parameters_for_optimizer
    return _group_parameters_for_optimizer(*args, **kwargs)

def train_one_fold(*args, **kwargs):
    """Lazy import of train_one_fold."""
    from .main import train_one_fold as _train_one_fold
    return _train_one_fold(*args, **kwargs)

def training_main(*args, **kwargs):
    """Lazy import of training_main."""
    from .main import training_main as _training_main
    return _training_main(*args, **kwargs)

__all__ = [
    "LMOLClassificationTrainer",
    "SaveBestTrainingLossCallback", 
    "group_parameters_for_optimizer",
    "train_one_fold",
    "training_main",
]
