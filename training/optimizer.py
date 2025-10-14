# -*- coding: utf-8 -*-
"""
LMOL Optimizer Setup

This module provides optimizer configuration for the LMOL project:
- Parameter grouping for dual learning rate strategy
- Optimizer creation with proper parameter groups
- Learning rate scheduling setup

Key Features:
- Dual learning rate strategy (LoRA vs Projector)
- Parameter validation and reporting
- Optimizer group management
"""

from typing import List, Dict, Any
import torch
from torch.optim import AdamW


def group_parameters_for_optimizer(model) -> List[Dict[str, Any]]:
    """
    Group model parameters into optimizer groups with different learning rates.
    
    Implements dual learning rate strategy:
    - LoRA parameters: Lower LR (1e-4) for adapting pretrained weights
    - Projector parameters: Higher LR (2e-3) for learning from scratch
    
    Args:
        model: PyTorch model with LoRA and projector parameters
        
    Returns:
        List of parameter groups for optimizer
        
    Raises:
        AssertionError: If no trainable parameters are found
    """
    from configs.config import config
    
    lora_params, proj_params = [], []
    
    # Separate parameters by type
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(p)
        elif "mm_projector" in n or "multi_modal_projector" in n or "projector" in n:
            proj_params.append(p)

    # Create parameter groups with appropriate learning rates
    param_groups: List[Dict[str, Any]] = []
    if proj_params:
        param_groups.append({"params": proj_params, "lr": config.LR_PROJECTION})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": config.LR_LORA})
    
    assert param_groups, "No trainable parameters found for optimizer."

    # Verify parameter counts match
    total_trainable = sum(p.numel() for p in proj_params) + sum(p.numel() for p in lora_params)
    total_trainable_check = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_trainable == total_trainable_check, (
        f"group sum ({total_trainable:,}) != actual trainable ({total_trainable_check:,})"
    )
    return param_groups


def create_optimizer(model, weight_decay: float = 0.01) -> AdamW:
    """
    Create AdamW optimizer with dual learning rate strategy.
    
    Args:
        model: PyTorch model to optimize
        weight_decay: Weight decay regularization coefficient
        
    Returns:
        Configured AdamW optimizer
    """
    param_groups = group_parameters_for_optimizer(model)
    return AdamW(param_groups, betas=(0.9, 0.999), weight_decay=weight_decay)


def create_scheduler(optimizer, total_steps: int, warmup_steps: int, schedule_type: str = "cosine"):
    """
    Create learning rate scheduler with enhanced configuration.
    
    Args:
        optimizer: Optimizer to schedule
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        schedule_type: Type of schedule ("cosine", "linear", or "constant")
        
    Returns:
        Configured scheduler or None if scheduling is disabled
    """
    from configs.config import config
    from .lr_scheduler import create_multi_lr_scheduler, print_lr_schedule_info
    
    if not getattr(config, 'USE_LR_SCHEDULING', False):
        return None
    
    # Calculate warmup ratio
    warmup_ratio = warmup_steps / total_steps if total_steps > 0 else 0.1
    
    # Create enhanced scheduler
    scheduler = create_multi_lr_scheduler(
        optimizer=optimizer,
        num_training_steps=total_steps,
        lora_lr=config.LR_LORA,
        projection_lr=config.LR_PROJECTION,
        warmup_ratio=warmup_ratio,
        schedule_type=schedule_type,
        min_lr_ratio=config.LR_MIN_RATIO
    )
    
    # Print schedule information
    print_lr_schedule_info(scheduler, total_steps, warmup_ratio, schedule_type)
    
    return scheduler
