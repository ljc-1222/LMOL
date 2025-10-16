# -*- coding: utf-8 -*-
"""
Gradient Auditing Utilities for PyTorch Training

This module provides comprehensive gradient monitoring and debugging utilities
to detect and fix gradient vanishing, exploding, and other training issues.
"""

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler


class GradientAudit:
    """
    Comprehensive gradient auditing system for PyTorch training.
    """
    
    def __init__(self, 
                 enabled: bool = True,
                 patterns: List[str] = None,
                 max_records: int = 1024):
        """
        Initialize gradient auditing system.
        
        Args:
            enabled: Whether gradient auditing is enabled
            patterns: List of module patterns to monitor
            max_records: Maximum number of records to keep in memory
        """
        self.enabled = enabled
        self.patterns = patterns or ["Linear", "Conv", "Embedding", "LayerNorm", "BatchNorm", "Dropout"]
        self.max_records = max_records
        
        # Data storage
        self.activation_stats = deque(maxlen=max_records)
        self.gradient_stats = deque(maxlen=max_records)
        self.hooks = []
        
        # AMP diagnostics
        self.amp_dtype = None
        self.amp_scaler = None
        self.overflow_count = 0
        self.underflow_count = 0
        
        # Memory tracking
        self.memory_stats = deque(maxlen=max_records)
        
        # Step counter
        self.step_count = 0
    
    def register_activation_and_grad_hooks(self, module: nn.Module, patterns: List[str]) -> None:
        """
        Register forward and backward hooks for gradient monitoring.
        
        Args:
            module: PyTorch model to monitor
            patterns: List of module patterns to monitor
        """
        if not self.enabled:
            return
        
        for name, submodule in module.named_modules():
            if any(pattern in submodule.__class__.__name__ for pattern in patterns):
                # Forward hook for activation statistics
                forward_hook = self._create_forward_hook(name)
                handle = submodule.register_forward_hook(forward_hook)
                self.hooks.append(handle)
                
                # Backward hook for gradient statistics
                backward_hook = self._create_backward_hook(name)
                handle = submodule.register_backward_hook(backward_hook)
                self.hooks.append(handle)
    
    def _create_forward_hook(self, name: str):
        """Create forward hook for activation monitoring."""
        def hook(module, input, output):
            if not self.enabled:
                return
            
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    stats = {
                        'name': name,
                        'step': self.step_count,
                        'timestamp': time.time(),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                    }
                    
                    # Saturation analysis for specific activations
                    if isinstance(module, (nn.ReLU, nn.ReLU6)):
                        stats['saturation_ratio'] = (output == 0).float().mean().item()
                    elif isinstance(module, nn.Sigmoid):
                        stats['saturation_ratio'] = ((output < 1e-3) | (output > 1 - 1e-3)).float().mean().item()
                    elif isinstance(module, nn.Tanh):
                        stats['saturation_ratio'] = (output.abs() > 0.99).float().mean().item()
                    
                    self.activation_stats.append(stats)
        
        return hook
    
    def _create_backward_hook(self, name: str):
        """Create backward hook for gradient monitoring."""
        def hook(module, grad_input, grad_output):
            if not self.enabled:
                return
            
            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                grad = grad_output[0]
                with torch.no_grad():
                    stats = {
                        'name': name,
                        'step': self.step_count,
                        'timestamp': time.time(),
                        'grad_norm_l2': grad.norm().item(),
                        'grad_norm_inf': grad.norm(float('inf')).item(),
                    }
                    self.gradient_stats.append(stats)
        
        return hook
    
    def setup_amp_diagnostics(self, amp_dtype: str, amp_scaler: Optional[GradScaler] = None) -> None:
        """
        Setup AMP diagnostics.
        
        Args:
            amp_dtype: AMP dtype ('fp16', 'bf16', or 'none')
            amp_scaler: Optional gradient scaler for fp16
        """
        self.amp_dtype = amp_dtype
        self.amp_scaler = amp_scaler
    
    def log_amp_stats(self, step: int) -> None:
        """Log AMP statistics."""
        if not self.enabled or self.amp_dtype is None:
            return
        
        if self.amp_dtype == 'fp16' and self.amp_scaler is not None:
            scale = self.amp_scaler.get_scale()
            if scale < 1.0:
                self.overflow_count += 1
            elif scale > 1.0:
                self.underflow_count += 1
    
    def log_memory_stats(self, step: int) -> None:
        """Log memory statistics."""
        if not self.enabled or not torch.cuda.is_available():
            return
        
        memory_stats = {
            'step': step,
            'timestamp': time.time(),
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
        }
        self.memory_stats.append(memory_stats)
    
    def log_grad_table(self, model: nn.Module, save_csv: bool = True) -> None:
        """
        Log gradient table for all parameters.
        
        Args:
            model: PyTorch model
            save_csv: Whether to save CSV file
        """
        if not self.enabled:
            return
        
        grad_data = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm_l2 = param.grad.norm().item()
                    grad_norm_inf = param.grad.norm(float('inf')).item()
                else:
                    grad_norm_l2 = 0.0
                    grad_norm_inf = 0.0
                
                grad_data.append({
                    'name': name,
                    'shape': str(list(param.shape)),
                    'grad_norm_l2': grad_norm_l2,
                    'grad_norm_inf': grad_norm_inf,
                })
        
        if save_csv:
            import csv
            csv_path = f"grad_table_step_{self.step_count}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['name', 'shape', 'grad_norm_l2', 'grad_norm_inf'])
                writer.writeheader()
                writer.writerows(grad_data)
    
    def plot_gradient_flow(self, model: nn.Module, save_path: str = "gradient_flow.png") -> None:
        """
        Plot gradient flow visualization.
        
        Args:
            model: PyTorch model
            save_path: Path to save the plot
        """
        if not MATPLOTLIB_AVAILABLE or not self.enabled:
            return
        
        # Collect gradient norms
        grad_norms = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                else:
                    grad_norm = 0.0
                grad_norms.append(grad_norm)
                layer_names.append(name.split('.')[-1])  # Use last part of name
        
        # Create plot
        plt.figure(figsize=(12, 6))
        x = range(len(grad_norms))
        y = [np.log10(max(norm, 1e-12)) for norm in grad_norms]  # Log scale with minimum
        
        plt.bar(x, y)
        plt.xlabel('Layer Index')
        plt.ylabel('log10(Gradient L2 Norm)')
        plt.title('Gradient Flow Visualization')
        plt.xticks(x[::max(1, len(x)//10)], layer_names[::max(1, len(x)//10)], rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def check_param_updates(self, model: nn.Module, pre_params: Dict[str, torch.Tensor], eps: float = 1e-12) -> Dict[str, float]:
        """
        Check parameter updates after optimizer step.
        
        Args:
            model: PyTorch model
            pre_params: Parameters before optimizer step
            eps: Minimum update threshold
            
        Returns:
            Dictionary of parameter update norms
        """
        if not self.enabled:
            return {}
        
        updates = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in pre_params:
                delta = (param - pre_params[name]).norm().item()
                updates[name] = delta
        
        if updates:
            max_update = max(updates.values())
            if max_update < eps:
                raise RuntimeError(f"No meaningful parameter updates detected (max update: {max_update:.2e} < {eps:.2e})")
        
        return updates
    
    def increment_step(self) -> None:
        """Increment step counter."""
        self.step_count += 1
    
    def cleanup(self) -> None:
        """Clean up hooks and resources."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# Convenience functions
def register_activation_and_grad_hooks(module: nn.Module, patterns: List[str]) -> GradientAudit:
    """Register hooks and return audit instance."""
    audit = GradientAudit()
    audit.register_activation_and_grad_hooks(module, patterns)
    return audit


def assert_grad_flow(module: nn.Module, tiny: float = 1e-12, warn_only: bool = False) -> None:
    """Assert gradient flow health."""
    audit = GradientAudit()
    audit.assert_grad_flow(module, tiny, warn_only)


def check_param_updates(model: nn.Module, pre_params: Dict[str, torch.Tensor], eps: float = 1e-12) -> Dict[str, float]:
    """Check parameter updates."""
    audit = GradientAudit()
    return audit.check_param_updates(model, pre_params, eps)