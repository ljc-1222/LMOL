# -*- coding: utf-8 -*-
"""
Gradient Health Diagnostics for LMOL Training

This module provides comprehensive gradient monitoring and diagnostics for PyTorch training:
- Per-parameter gradient statistics collection
- Global gradient norm computation and clipping
- Gradient anomaly detection (NaN/Inf/zero gradients)
- Autograd anomaly detection
- AMP underflow detection
- Training parameter validation

Key Features:
- Backward hooks for real-time gradient monitoring
- Comprehensive gradient health reporting
- Memory-efficient gradient statistics collection
- Integration with training loops
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

# Set up logging
logger = logging.getLogger(__name__)


class GradientDiagnostics:
    """
    Comprehensive gradient health monitoring and diagnostics.
    
    This class provides real-time gradient monitoring during training to detect
    and diagnose common issues like vanishing gradients, exploding gradients,
    NaN/Inf gradients, and broken backpropagation.
    """
    
    def __init__(self, 
                 log_interval: int = 10,
                 clip_grad_norm: float = 0.0,
                 detect_anomaly: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initialize gradient diagnostics.
        
        Args:
            log_interval: How often to log gradient statistics (every N steps)
            clip_grad_norm: Global gradient clipping threshold (0 = disabled)
            detect_anomaly: Enable autograd anomaly detection
            device: Device for tensor operations
        """
        self.log_interval = log_interval
        self.clip_grad_norm = clip_grad_norm
        self.detect_anomaly = detect_anomaly
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Gradient statistics storage
        self.grad_stats: Dict[str, Dict[str, Any]] = {}
        self.global_grad_norm = 0.0
        self.step_count = 0
        
        # Hooks for gradient monitoring
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # AMP scaler monitoring
        self.scaler_scale = 1.0
        self.scaler_underflow_warnings = 0
        
        # Anomaly detection state
        self.anomaly_enabled = False
        
    def register_model_hooks(self, model: nn.Module) -> None:
        """
        Register backward hooks on all trainable parameters.
        
        Args:
            model: PyTorch model to monitor
        """
        # Clear existing hooks
        self.clear_hooks()
        
        # Register hooks on all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(grad, name)
                )
                self.hooks.append(hook)
                
        logger.info(f"Registered gradient hooks on {len(self.hooks)} parameters")
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _gradient_hook(self, grad: Optional[torch.Tensor], param_name: str) -> None:
        """
        Hook function called during backward pass for each parameter.
        
        Args:
            grad: Gradient tensor (may be None)
            param_name: Name of the parameter
        """
        if grad is None:
            self.grad_stats[param_name] = {
                'grad_is_none_count': 1,
                'grad_all_zero_count': 0,
                'grad_nan_inf_count': 0,
                'grad_norm': 0.0,
                'grad_abs_mean': 0.0,
                'grad_abs_max': 0.0,
            }
            return
        
        # Compute gradient statistics
        grad_norm = grad.norm().item() if grad.numel() > 0 else 0.0
        grad_abs_mean = grad.abs().mean().item() if grad.numel() > 0 else 0.0
        grad_abs_max = grad.abs().max().item() if grad.numel() > 0 else 0.0
        
        # Check for anomalies
        grad_is_none = 0
        grad_all_zero = 1 if grad_norm < 1e-8 else 0
        grad_nan_inf = 1 if not torch.isfinite(grad).all() else 0
        
        self.grad_stats[param_name] = {
            'grad_is_none_count': grad_is_none,
            'grad_all_zero_count': grad_all_zero,
            'grad_nan_inf_count': grad_nan_inf,
            'grad_norm': grad_norm,
            'grad_abs_mean': grad_abs_mean,
            'grad_abs_max': grad_abs_max,
        }
    
    def compute_global_grad_norm(self, model: nn.Module) -> float:
        """
        Compute global gradient norm across all parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Global gradient norm
        """
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            self.global_grad_norm = total_norm ** (1. / 2)
        else:
            self.global_grad_norm = 0.0
            
        return self.global_grad_norm
    
    def apply_gradient_clipping(self, model: nn.Module) -> float:
        """
        Apply global gradient clipping if enabled.
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient norm after clipping
        """
        if self.clip_grad_norm <= 0:
            return self.compute_global_grad_norm(model)
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
        
        # Compute norm after clipping
        return self.compute_global_grad_norm(model)
    
    def log_gradient_stats(self, step: int, model: nn.Module) -> None:
        """
        Log comprehensive gradient statistics.
        
        Args:
            step: Current training step
            model: PyTorch model
        """
        if step % self.log_interval != 0:
            return
        
        # Compute global gradient norm
        global_norm = self.compute_global_grad_norm(model)
        
        # Aggregate statistics
        total_params = 0
        zero_grad_count = 0
        nan_inf_count = 0
        max_grad_abs = 0.0
        grad_norms = []
        
        for param_name, stats in self.grad_stats.items():
            total_params += 1
            if stats['grad_all_zero_count'] > 0:
                zero_grad_count += 1
            if stats['grad_nan_inf_count'] > 0:
                nan_inf_count += 1
            max_grad_abs = max(max_grad_abs, stats['grad_abs_max'])
            if stats['grad_norm'] > 0:
                grad_norms.append(stats['grad_norm'])
        
        # Calculate additional statistics
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        min_grad_norm = min(grad_norms) if grad_norms else 0.0
        max_grad_norm = max(grad_norms) if grad_norms else 0.0
        
        # Log comprehensive summary
        print(f"[GRAD] Step {step:4d} | Global: {global_norm:.2e} | "
              f"Zero: {zero_grad_count}/{total_params} | "
              f"NaN/Inf: {nan_inf_count}/{total_params} | "
              f"Max: {max_grad_abs:.2e} | "
              f"Avg: {avg_grad_norm:.2e} | "
              f"Range: [{min_grad_norm:.2e}, {max_grad_norm:.2e}]")
        
        # Log detailed statistics for problematic parameters
        if zero_grad_count > 0 or nan_inf_count > 0:
            print(f"[GRAD] Problematic parameters:")
            for param_name, stats in self.grad_stats.items():
                if stats['grad_all_zero_count'] > 0 or stats['grad_nan_inf_count'] > 0:
                    print(f"  {param_name}: norm={stats['grad_norm']:.2e}, "
                          f"zero={stats['grad_all_zero_count']}, "
                          f"nan_inf={stats['grad_nan_inf_count']}")
        
        # Check for gradient health issues
        if zero_grad_count > total_params * 0.5:
            print(f"[WARNING] {zero_grad_count}/{total_params} parameters have zero gradients!")
        
        if nan_inf_count > 0:
            print(f"[ERROR] {nan_inf_count}/{total_params} parameters have NaN/Inf gradients!")
        
        if global_norm > 100.0:
            print(f"[WARNING] Very large global gradient norm: {global_norm:.2e}")
        elif global_norm < 1e-8:
            print(f"[WARNING] Very small global gradient norm: {global_norm:.2e}")
    
    def enable_anomaly_detection(self) -> None:
        """Enable PyTorch autograd anomaly detection."""
        if self.detect_anomaly and not self.anomaly_enabled:
            torch.autograd.set_detect_anomaly(True)
            self.anomaly_enabled = True
            logger.info("Autograd anomaly detection enabled")
    
    def disable_anomaly_detection(self) -> None:
        """Disable PyTorch autograd anomaly detection."""
        if self.anomaly_enabled:
            torch.autograd.set_detect_anomaly(False)
            self.anomaly_enabled = False
            logger.info("Autograd anomaly detection disabled")
    
    def check_amp_underflow(self, scaler: Optional[torch.cuda.amp.GradScaler]) -> None:
        """
        Check for AMP underflow and warn if necessary.
        
        Args:
            scaler: GradScaler instance (None for bf16)
        """
        if scaler is not None:
            current_scale = scaler.get_scale()
            if current_scale < 2.0:
                self.scaler_underflow_warnings += 1
                if self.scaler_underflow_warnings % 10 == 1:  # Warn every 10 occurrences
                    logger.warning(f"AMP underflow detected: scale={current_scale:.2e} "
                                 f"(warnings: {self.scaler_underflow_warnings})")
    
    def validate_training_setup(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """
        Validate training setup for common issues.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
        """
        # Check model device placement
        model_device = next(model.parameters()).device
        print(f"[GRAD] Model device: {model_device}")
        
        # Check optimizer device
        for i, group in enumerate(optimizer.param_groups):
            if group['params']:
                param_device = next(iter(group['params'])).device
                print(f"[GRAD] Optimizer group {i} device: {param_device}")
        
        # Check for frozen parameters
        frozen_params = []
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        print(f"[GRAD] Trainable parameters: {len(trainable_params)}")
        print(f"[GRAD] Frozen parameters: {len(frozen_params)}")
        
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found!")
        
        # Check optimizer parameter groups
        print(f"[GRAD] Optimizer groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            lr = group.get('lr', 0.0)
            weight_decay = group.get('weight_decay', 0.0)
            param_count = len(group['params'])
            print(f"  Group {i}: {param_count} params, LR={lr:.2e}, WD={weight_decay:.2e}")
            
            # Warn about suspicious learning rates
            if lr > 1.0:
                logger.warning(f"Group {i} has very high learning rate: {lr:.2e}")
            elif lr < 1e-8:
                logger.warning(f"Group {i} has very low learning rate: {lr:.2e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_hooks()
        self.disable_anomaly_detection()


def assert_all_requires_grad(model: nn.Module, expected_trainable: bool = True) -> None:
    """
    Assert that all parameters have the expected requires_grad setting.
    
    Args:
        model: PyTorch model
        expected_trainable: Whether parameters should be trainable
    """
    for name, param in model.named_parameters():
        if param.requires_grad != expected_trainable:
            raise AssertionError(
                f"Parameter {name} has requires_grad={param.requires_grad}, "
                f"expected {expected_trainable}"
            )


def check_inplace_operations(model: nn.Module) -> List[str]:
    """
    Check for in-place operations that might break gradients.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of parameter names with potential in-place operation issues
    """
    problematic_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU)):
            if hasattr(module, 'inplace') and module.inplace:
                problematic_params.append(f"{name} (inplace=True)")
        elif isinstance(module, nn.Dropout):
            if hasattr(module, 'inplace') and module.inplace:
                problematic_params.append(f"{name} (inplace=True)")
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            if hasattr(module, 'inplace') and module.inplace:
                problematic_params.append(f"{name} (inplace=True)")
    
    return problematic_params


def check_gradient_flow(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    Check gradient flow through the model.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor
        
    Returns:
        Dictionary containing gradient flow analysis
    """
    model.train()
    
    # Forward pass
    output = model(sample_input)
    
    # Check if output requires grad
    requires_grad = output.requires_grad if hasattr(output, 'requires_grad') else False
    
    # Check for NaN/Inf in output
    has_nan = torch.isnan(output).any().item() if hasattr(output, 'isnan') else False
    has_inf = torch.isinf(output).any().item() if hasattr(output, 'isinf') else False
    
    # Check parameter gradients
    param_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_grads.append({
                'name': name,
                'has_grad': param.grad is not None,
                'grad_norm': param.grad.norm().item() if param.grad is not None else 0.0,
                'grad_has_nan': torch.isnan(param.grad).any().item() if param.grad is not None else False,
                'grad_has_inf': torch.isinf(param.grad).any().item() if param.grad is not None else False,
            })
    
    return {
        'output_requires_grad': requires_grad,
        'output_has_nan': has_nan,
        'output_has_inf': has_inf,
        'param_grads': param_grads,
        'total_params': len(param_grads),
        'params_with_grads': sum(1 for p in param_grads if p['has_grad']),
        'params_with_nan': sum(1 for p in param_grads if p['grad_has_nan']),
        'params_with_inf': sum(1 for p in param_grads if p['grad_has_inf']),
    }


def setup_gradient_diagnostics(
    model: nn.Module,
    log_interval: int = 10,
    clip_grad_norm: float = 0.0,
    detect_anomaly: bool = False
) -> GradientDiagnostics:
    """
    Set up gradient diagnostics for a model.
    
    Args:
        model: PyTorch model to monitor
        log_interval: How often to log gradient statistics
        clip_grad_norm: Global gradient clipping threshold (0 = disabled)
        detect_anomaly: Enable autograd anomaly detection
        
    Returns:
        Configured GradientDiagnostics instance
    """
    device = next(model.parameters()).device
    diagnostics = GradientDiagnostics(
        log_interval=log_interval,
        clip_grad_norm=clip_grad_norm,
        detect_anomaly=detect_anomaly,
        device=device
    )
    
    # Register hooks
    diagnostics.register_model_hooks(model)
    
    # Enable anomaly detection if requested
    if detect_anomaly:
        diagnostics.enable_anomaly_detection()
    
    return diagnostics
