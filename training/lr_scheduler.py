# -*- coding: utf-8 -*-
"""
Learning Rate Scheduling Utilities for LMOL

This module provides advanced learning rate scheduling capabilities:
- Cosine annealing with optional restarts
- Linear decay scheduling
- Warmup period implementation
- Multi-parameter group scheduling

Key Features:
- Support for different LR schedules (cosine, linear, constant)
- Warmup period for stable training start
- Separate scheduling for LoRA and projection parameters
- Configurable minimum learning rate ratios
"""

import math
from typing import List, Optional
import torch
from torch.optim.lr_scheduler import LambdaLR

from configs.config import config


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.01
) -> LambdaLR:
    """
    Create a cosine learning rate schedule with warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (0.5 for half cycle)
        last_epoch: Last epoch index
        min_lr_ratio: Minimum learning rate ratio
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.01
) -> LambdaLR:
    """
    Create a linear learning rate schedule with warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch index
        min_lr_ratio: Minimum learning rate ratio
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a constant learning rate schedule with warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        last_epoch: Last epoch index
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    schedule_type: str = "cosine",
    min_lr_ratio: float = 0.01,
    num_cycles: float = 0.5
) -> LambdaLR:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps
        warmup_ratio: Fraction of steps for warmup
        schedule_type: Type of schedule ('cosine', 'linear', 'constant')
        min_lr_ratio: Minimum learning rate ratio
        num_cycles: Number of cosine cycles
        
    Returns:
        Configured LambdaLR scheduler
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    if schedule_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            min_lr_ratio=min_lr_ratio
        )
    elif schedule_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio
        )
    elif schedule_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def create_multi_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    lora_lr: float,
    projection_lr: float,
    warmup_ratio: float = 0.1,
    schedule_type: str = "cosine",
    min_lr_ratio: float = 0.01
) -> LambdaLR:
    """
    Create a learning rate scheduler for multi-parameter groups.
    
    This function creates a scheduler that handles different learning rates
    for LoRA and projection parameters separately.
    
    Args:
        optimizer: Optimizer with multiple parameter groups
        num_training_steps: Total number of training steps
        lora_lr: Learning rate for LoRA parameters
        projection_lr: Learning rate for projection parameters
        warmup_ratio: Fraction of steps for warmup
        schedule_type: Type of schedule ('cosine', 'linear', 'constant')
        min_lr_ratio: Minimum learning rate ratio
        
    Returns:
        Configured LambdaLR scheduler
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    def lr_lambda(current_step: int) -> float:
        """
        Calculate learning rate multiplier for all parameter groups.
        
        Args:
            current_step: Current training step
            
        Returns:
            Single learning rate multiplier for all parameter groups
        """
        if current_step < num_warmup_steps:
            # Warmup phase: linear increase from 0 to 1
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Main training phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        if schedule_type == "cosine":
            # Cosine annealing
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif schedule_type == "linear":
            # Linear decay
            return max(min_lr_ratio, 1.0 - progress)
        elif schedule_type == "constant":
            # Constant learning rate
            return 1.0
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return LambdaLR(optimizer, lr_lambda)


def print_lr_schedule_info(
    scheduler: LambdaLR,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
    schedule_type: str = "cosine"
):
    """
    Print learning rate schedule information.
    
    Args:
        scheduler: Learning rate scheduler
        num_training_steps: Total number of training steps
        warmup_ratio: Fraction of steps for warmup
        schedule_type: Type of schedule
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    print(f"\n[LR_SCHEDULE] Learning Rate Schedule Configuration:")
    print(f"  Schedule Type: {schedule_type}")
    print(f"  Total Steps: {num_training_steps}")
    print(f"  Warmup Steps: {num_warmup_steps} ({warmup_ratio:.1%})")
    print(f"  Training Steps: {num_training_steps - num_warmup_steps}")
    
    # Show LR at key points
    key_steps = [0, num_warmup_steps, num_training_steps // 2, num_training_steps - 1]
    print(f"  Learning Rate Schedule:")
    for step in key_steps:
        if step < num_training_steps:
            lr_multiplier = scheduler.lr_lambdas[0](step) if hasattr(scheduler, 'lr_lambdas') else 1.0
            print(f"    Step {step:4d}: LR multiplier = {lr_multiplier:.6f}")
    
    print()
