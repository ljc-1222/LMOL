# -*- coding: utf-8 -*-
"""
LMOL Model Module

This module provides all model-related functionality for the LMOL project:
- Model architecture definitions
- Custom projector implementation
- Model factory functions for training and inference
- LoRA integration utilities

Key Components:
- architecture: Core model architecture and utilities
- projector: LMOL custom projector implementation
- factory: Model factory functions for different use cases
"""

from .architecture import (
    _find_projector_handle,
    _set_all_requires_grad,
    _report_model_sizes,
    _breakdown_projector,
)
from .projector import LMOLProjector

# Lazy imports for factory functions to avoid dependency issues
def model_generator():
    """Lazy import of model_generator."""
    from .factory import model_generator as _model_generator
    return _model_generator()

def build_inference_base():
    """Lazy import of build_inference_base."""
    from .factory import build_inference_base as _build_inference_base
    return _build_inference_base()

__all__ = [
    "_find_projector_handle",
    "_set_all_requires_grad", 
    "_report_model_sizes",
    "_breakdown_projector",
    "LMOLProjector",
    "model_generator",
    "build_inference_base",
]
