# -*- coding: utf-8 -*-
"""
LMOL Model Architecture Utilities

This module provides core model architecture utilities for the LMOL project.
It contains helper functions for model manipulation, parameter management,
and architecture analysis.

Key Functions:
- _find_projector_handle: Locate projector modules in LLaVA models
- _set_all_requires_grad: Configure parameter gradients
- _report_model_sizes: Calculate model parameter statistics
- _breakdown_projector: Analyze projector parameter counts
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


def _find_projector_handle(model: nn.Module) -> Tuple[str, Optional[nn.Module]]:
    """
    Find the original projector module in a LLaVA model.
    
    Searches for the projector module using common attribute names used
    by different LLaVA checkpoints. Returns both the attribute path
    and the module instance for replacement.
    
    Args:
        model: LLaVA model to search
        
    Returns:
        Tuple of (attribute_path, module_instance) or ("", None) if not found
    """
    # Common projector attribute names in different LLaVA versions
    candidates = [
        "multi_modal_projector",
        "mm_projector",
        "model.multi_modal_projector",
        "model.mm_projector",
    ]
    
    for dotted in candidates:
        cur = model
        ok = True
        for p in dotted.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and cur is not None:
            return dotted, cur
    return "", None


def _set_all_requires_grad(module: nn.Module, value: bool) -> None:
    """
    Set requires_grad for all parameters in a module.
    
    Args:
        module: PyTorch module
        value: Boolean value to set for requires_grad
    """
    for p in module.parameters():
        p.requires_grad = value


def _report_model_sizes(model: nn.Module) -> Tuple[int, int, float]:
    """
    Calculate and return model size statistics.
    
    Computes total parameters, trainable parameters, and trainable ratio
    without printing (printing is handled in train.py for better organization).
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Tuple of (total_params, trainable_params, trainable_ratio)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable / total) if total else 0.0
    return total, trainable, ratio


def _breakdown_projector(proj: nn.Module) -> int:
    """
    Analyze projector module and return trainable parameter count.
    
    Counts trainable parameters in the projector module for validation.
    This helps ensure proper parameter accounting during model setup.
    
    Args:
        proj: Projector module to analyze
        
    Returns:
        Total number of trainable parameters in the projector
    """
    trainable_sum = 0
    for name, m in proj.named_modules():
        if isinstance(m, nn.Linear):
            n_params = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            if any(p.requires_grad for p in m.parameters()):
                trainable_sum += n_params
    return trainable_sum
