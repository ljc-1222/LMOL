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

def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
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
