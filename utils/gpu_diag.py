# -*- coding: utf-8 -*-
"""
GPU Diagnostics and Monitoring Utilities for LMOL

This module provides comprehensive GPU diagnostics and monitoring capabilities
for the LMOL training pipeline. It includes device verification, memory
monitoring, and performance tracking utilities.

Key Features:
- Device placement verification and assertions
- Memory usage monitoring and logging
- GPU utilization tracking (with optional NVML)
- Performance diagnostics and health checks
- A100-specific optimizations and warnings
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
import torch
import logging

# Optional NVML for advanced GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUDeviceManager:
    """
    Centralized GPU device management and diagnostics.
    
    This class provides a single source of truth for device management
    and comprehensive GPU diagnostics for the LMOL training pipeline.
    """
    
    def __init__(self, target_device: int = 0):
        """
        Initialize GPU device manager.
        
        Args:
            target_device: Target GPU device index (default: 0 for single A100)
        """
        self.target_device = target_device
        self.device = None
        self.device_properties = None
        self._nvml_initialized = False
        
        # Initialize device
        self._initialize_device()
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            self._initialize_nvml()
    
    def _initialize_device(self) -> None:
        """Initialize PyTorch device and verify CUDA availability."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
            return
        
        # Verify we're targeting the correct device
        available_devices = torch.cuda.device_count()
        if self.target_device >= available_devices:
            raise RuntimeError(f"Target device {self.target_device} not available. "
                             f"Only {available_devices} devices found.")
        
        # Set the target device
        torch.cuda.set_device(self.target_device)
        self.device = torch.device(f"cuda:{self.target_device}")
        
        # Get device properties
        self.device_properties = torch.cuda.get_device_properties(self.target_device)
        
        # Verify we're using the expected A100
        device_name = self.device_properties.name
        if "A100" not in device_name:
            logger.warning(f"Expected A100 GPU, but found: {device_name}")
        
        logger.info(f"Initialized device: {device_name} (CUDA {self.device_properties.major}.{self.device_properties.minor})")
    
    def _initialize_nvml(self) -> None:
        """Initialize NVML for advanced GPU monitoring."""
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.info("NVML initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            self._nvml_initialized = False
    
    def get_device(self) -> torch.device:
        """Get the canonical device for this training session."""
        return self.device
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get comprehensive device information.
        
        Returns:
            Dictionary containing device information
        """
        if self.device.type == "cpu":
            return {
                "device_type": "cpu",
                "cuda_available": False,
                "device_count": 0,
                "current_device": None,
                "device_name": "CPU",
                "compute_capability": None,
                "total_memory": None,
                "free_memory": None,
                "used_memory": None,
            }
        
        # Get memory info
        total_memory, free_memory = torch.cuda.mem_get_info(self.device)
        used_memory = total_memory - free_memory
        
        info = {
            "device_type": "cuda",
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": self.device_properties.name,
            "compute_capability": f"{self.device_properties.major}.{self.device_properties.minor}",
            "total_memory": total_memory,
            "free_memory": free_memory,
            "used_memory": used_memory,
            "total_memory_gb": total_memory / (1024**3),
            "free_memory_gb": free_memory / (1024**3),
            "used_memory_gb": used_memory / (1024**3),
        }
        
        # Add NVML info if available
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.target_device)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                info.update({
                    "gpu_utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                    "nvml_total_memory": memory_info.total,
                    "nvml_free_memory": memory_info.free,
                    "nvml_used_memory": memory_info.used,
                })
            except Exception as e:
                logger.warning(f"Failed to get NVML info: {e}")
        
        return info
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary containing memory statistics
        """
        if self.device.type == "cpu":
            return {"error": "Memory stats not available for CPU"}
        
        stats = {
            "allocated": torch.cuda.memory_allocated(self.device),
            "reserved": torch.cuda.memory_reserved(self.device),
            "max_allocated": torch.cuda.max_memory_allocated(self.device),
            "max_reserved": torch.cuda.max_memory_reserved(self.device),
        }
        
        # Convert to GB
        gb_stats = {}
        for key, value in stats.items():
            if isinstance(value, int):
                gb_stats[f"{key}_gb"] = value / (1024**3)
        stats.update(gb_stats)
        
        return stats
    
    def reset_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def assert_all_cuda(self, module: torch.nn.Module) -> None:
        """
        Assert that all parameters and buffers in a module are on CUDA.
        
        Args:
            module: PyTorch module to check
            
        Raises:
            RuntimeError: If any parameter or buffer is not on CUDA
        """
        if self.device.type == "cpu":
            return  # Skip check for CPU
        
        cuda_device = self.device
        
        # Check parameters
        for name, param in module.named_parameters():
            if param.device != cuda_device:
                raise RuntimeError(f"Parameter '{name}' is on {param.device}, expected {cuda_device}")
        
        # Check buffers
        for name, buffer in module.named_buffers():
            if buffer.device != cuda_device:
                raise RuntimeError(f"Buffer '{name}' is on {buffer.device}, expected {cuda_device}")
    
    def verify_training_setup(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> None:
        """
        Verify that training setup is correct for GPU training.
        
        Args:
            model: Training model
            batch: Training batch
            
        Raises:
            RuntimeError: If training setup is incorrect
        """
        if self.device.type == "cpu":
            return  # Skip check for CPU
        
        # Check model device placement
        self.assert_all_cuda(model)
        
        # Check batch device placement
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                raise RuntimeError(f"Batch tensor '{key}' is on {tensor.device}, expected {self.device}")
        
        # Verify at least one parameter is on CUDA
        has_cuda_params = any(p.device.type == "cuda" for p in model.parameters())
        if not has_cuda_params:
            raise RuntimeError("No model parameters found on CUDA device")
        
        # Verify at least one batch tensor is on CUDA
        has_cuda_batch = any(t.device.type == "cuda" for t in batch.values() if isinstance(t, torch.Tensor))
        if not has_cuda_batch:
            raise RuntimeError("No batch tensors found on CUDA device")
    
    def log_memory_usage(self, step: int, prefix: str = "") -> None:
        """
        Log current memory usage.
        
        Args:
            step: Current training step
            prefix: Optional prefix for log message
        """
        if self.device.type == "cpu":
            return
        
        stats = self.get_memory_stats()
        info = self.get_device_info()
        
        # Get utilization if available
        gpu_util = info.get("gpu_utilization", "N/A")
        mem_util = info.get("memory_utilization", "N/A")
        
        logger.info(f"{prefix}Step {step}: "
                   f"Allocated: {stats['allocated_gb']:.2f}GB, "
                   f"Reserved: {stats['reserved_gb']:.2f}GB, "
                   f"Peak: {stats['max_allocated_gb']:.2f}GB, "
                   f"GPU Util: {gpu_util}%, "
                   f"Mem Util: {mem_util}%")


def create_device_manager(target_device: int = 0) -> GPUDeviceManager:
    """
    Create a GPU device manager instance.
    
    Args:
        target_device: Target GPU device index
        
    Returns:
        GPUDeviceManager instance
    """
    return GPUDeviceManager(target_device)


def assert_all_cuda(module: torch.nn.Module, device: torch.device) -> None:
    """
    Assert that all parameters and buffers in a module are on the specified device.
    
    Args:
        module: PyTorch module to check
        device: Expected device
        
    Raises:
        RuntimeError: If any parameter or buffer is not on the expected device
    """
    if device.type == "cpu":
        return  # Skip check for CPU
    
    # Check parameters
    for name, param in module.named_parameters():
        if param.device != device:
            raise RuntimeError(f"Parameter '{name}' is on {param.device}, expected {device}")
    
    # Check buffers
    for name, buffer in module.named_buffers():
        if buffer.device != device:
            raise RuntimeError(f"Buffer '{name}' is on {buffer.device}, expected {device}")


def get_canonical_device() -> torch.device:
    """
    Get the canonical device for this training session.
    
    Returns:
        Canonical device (cuda:0 if available, otherwise cpu)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_gpu_environment() -> Dict[str, Any]:
    """
    Set up GPU environment and return configuration.
    
    Returns:
        Dictionary containing GPU configuration
    """
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        try:
            visible_devices = [int(x) for x in cuda_visible.split(",")]
            if visible_devices[0] != 0:
                logger.warning(f"CUDA_VISIBLE_DEVICES={cuda_visible} - expected to start with 0")
        except ValueError:
            logger.warning(f"Invalid CUDA_VISIBLE_DEVICES: {cuda_visible}")
    else:
        logger.info("CUDA_VISIBLE_DEVICES not set - using default device 0")
    
    # Set PyTorch optimizations
    if torch.cuda.is_available():
        # Enable optimized matmul kernels
        torch.set_float32_matmul_precision("high")
        
        # Set CuDNN benchmark (will be overridden by config)
        torch.backends.cudnn.benchmark = True
        
        logger.info("PyTorch GPU optimizations enabled")
    
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_visible_devices": cuda_visible,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
