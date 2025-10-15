# -*- coding: utf-8 -*-
"""
LMOL Seed Management Module

This module provides comprehensive seed management for reproducible training
and evaluation across Python, NumPy, PyTorch, and CUDA operations.

Key Features:
- Cross-platform seed setting for all random number generators
- CUDA-specific seed management for GPU operations
- DataLoader worker seed management for reproducible data loading
- Optional deterministic CUDA operations (with performance trade-off)
- Fork safety configuration for multiprocessing

Usage:
    # Basic seed setting
    set_seed(42)
    
    # Get DataLoader kwargs for reproducible data loading
    dataloader_kwargs = get_dataloader_kwargs(42)
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    # For absolute reproducibility (slower)
    set_seed(42, deterministic_cudnn=True)
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional

def set_seed(seed: int, deterministic_cudnn: bool = False, enable_numerical_stability: bool = True) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    This function ensures reproducible behavior across:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generators
    - CUDA random number generators
    - DataLoader worker processes
    
    Args:
        seed: The random seed to set
        deterministic_cudnn: If True, sets cudnn.deterministic=True which ensures
                           reproducible convolution operations at the cost of performance.
                           Only set to True when absolute reproducibility is required.
        enable_numerical_stability: If True, enables additional numerical stability settings
    
    Note:
        Even with these settings, some operations may still be non-deterministic:
        - DataLoader with num_workers > 0 (each worker has its own seed)
        - Some CUDA operations may be non-deterministic despite cudnn.deterministic=True
        - Certain PyTorch operations on tensors with varying sizes
    """
    # Set basic seeds for Python and NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Configure CUDA backend settings
        if deterministic_cudnn:
            # Warning: this can slow down training significantly
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            # Faster, but not fully reproducible
            cudnn.deterministic = False
            cudnn.benchmark = True
        
        # Clear CUDA cache to avoid inherited states
        torch.cuda.empty_cache()
        
    # Set fork safety for reproducibility in DataLoader workers
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Enable numerical stability features
    if enable_numerical_stability:
        # Set high precision for matrix multiplications (A100 optimized)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        
        # Enable anomaly detection for debugging (can be disabled in production)
        # torch.autograd.set_detect_anomaly(True)  # Uncomment for debugging
        
        # Set numerical stability warnings
        import warnings
        warnings.filterwarnings("error", category=RuntimeWarning, message=".*overflow.*")
        warnings.filterwarnings("error", category=RuntimeWarning, message=".*invalid value.*")
        
        print(f"[SEED] Numerical stability features enabled for seed {seed}")
    
    # Define worker initialization function for DataLoader
    def seed_worker(worker_id: int) -> None:
        """
        Set seed for DataLoader workers to ensure reproducibility.
        
        This function is used as worker_init_fn in DataLoader to ensure
        that each worker process has a unique but deterministic seed.
        
        Args:
            worker_id: Unique identifier for the worker process
        """
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
    # Return worker init function for use with DataLoader
    return seed_worker

def get_dataloader_kwargs(seed: int) -> dict:
    """
    Get DataLoader configuration for reproducible data loading.
    
    This function returns the necessary keyword arguments for DataLoader
    to ensure reproducible data loading across different runs.
    
    Args:
        seed: Base seed value for worker initialization
        
    Returns:
        Dictionary containing:
        - worker_init_fn: Function to initialize each worker with unique seed
        - generator: PyTorch generator for reproducible sampling
    
    Usage:
        dataloader_kwargs = get_dataloader_kwargs(42)
        dataloader = DataLoader(dataset, **dataloader_kwargs)
    """
    # Create a unique generator for DataLoader sampling
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Get worker initialization function
    worker_init_fn = set_seed(seed)
    
    return {
        "worker_init_fn": worker_init_fn,
        "generator": g
    }


def check_numerical_stability(model: torch.nn.Module, sample_input: torch.Tensor, 
                            device: torch.device = None) -> dict:
    """
    Check numerical stability of model operations.
    
    Args:
        model: PyTorch model to check
        sample_input: Sample input tensor
        device: Device to run checks on (default: same as sample_input)
        
    Returns:
        Dictionary containing stability check results
    """
    if device is None:
        device = sample_input.device
    
    model.eval()
    with torch.no_grad():
        try:
            # Forward pass
            output = model(sample_input)
            
            # Check for NaN/Inf in output
            has_nan = torch.isnan(output).any().item() if hasattr(output, 'isnan') else False
            has_inf = torch.isinf(output).any().item() if hasattr(output, 'isinf') else False
            
            # Check parameter values
            param_stats = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_stats[name] = {
                        'has_nan': torch.isnan(param).any().item(),
                        'has_inf': torch.isinf(param).any().item(),
                        'min_val': param.min().item(),
                        'max_val': param.max().item(),
                        'mean_val': param.mean().item(),
                        'std_val': param.std().item(),
                    }
            
            # Count problematic parameters
            nan_params = sum(1 for stats in param_stats.values() if stats['has_nan'])
            inf_params = sum(1 for stats in param_stats.values() if stats['has_inf'])
            
            return {
                'output_has_nan': has_nan,
                'output_has_inf': has_inf,
                'param_stats': param_stats,
                'total_params': len(param_stats),
                'nan_params': nan_params,
                'inf_params': inf_params,
                'is_stable': not (has_nan or has_inf or nan_params > 0 or inf_params > 0)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'is_stable': False
            }


def enable_deterministic_training(seed: int = 42) -> None:
    """
    Enable fully deterministic training (slower but reproducible).
    
    Args:
        seed: Random seed to use
    """
    print(f"[DETERMINISTIC] Enabling fully deterministic training with seed {seed}")
    
    # Set seed with deterministic CUDA
    set_seed(seed, deterministic_cudnn=True, enable_numerical_stability=True)
    
    # Additional deterministic settings
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables for deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    print(f"[DETERMINISTIC] Deterministic training enabled (may be slower)")
