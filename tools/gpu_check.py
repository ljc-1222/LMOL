#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Environment Check Script for LMOL

This script performs a comprehensive check of the GPU environment
and PyTorch configuration for the LMOL training pipeline.

Usage:
    python tools/gpu_check.py

Features:
- PyTorch and CUDA version information
- GPU device detection and properties
- Memory information and utilization
- A100-specific optimizations check
- Environment variable validation
"""

import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from utils.gpu_diag import GPUDeviceManager, setup_gpu_environment


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_info(key: str, value: Any) -> None:
    """Print key-value information."""
    print(f"{key:25s}: {value}")


def check_pytorch_installation() -> None:
    """Check PyTorch installation and versions."""
    print_section("PyTorch Installation")
    
    print_info("PyTorch Version", torch.__version__)
    print_info("CUDA Available", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print_info("CUDA Version", torch.version.cuda)
        print_info("cuDNN Version", torch.backends.cudnn.version())
        print_info("cuDNN Enabled", torch.backends.cudnn.enabled)
        print_info("cuDNN Benchmark", torch.backends.cudnn.benchmark)
    else:
        print_info("CUDA Version", "Not available")
        print_info("cuDNN Version", "Not available")


def check_gpu_devices() -> None:
    """Check available GPU devices."""
    print_section("GPU Devices")
    
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return
    
    device_count = torch.cuda.device_count()
    print_info("Device Count", device_count)
    
    for i in range(device_count):
        print(f"\nGPU {i}:")
        props = torch.cuda.get_device_properties(i)
        print_info("  Name", props.name)
        print_info("  Compute Capability", f"{props.major}.{props.minor}")
        print_info("  Total Memory", f"{props.total_memory / (1024**3):.2f} GB")
        print_info("  Multiprocessors", props.multi_processor_count)
        print_info("  Max Threads per Block", getattr(props, 'max_threads_per_block', 'N/A'))
        print_info("  Max Threads per Multiprocessor", getattr(props, 'max_threads_per_multiprocessor', 'N/A'))
        
        # Check if this is an A100
        if "A100" in props.name:
            print_info("  A100 Optimizations", "Available")
        else:
            print_info("  A100 Optimizations", "Not available (not A100)")


def check_memory_usage() -> None:
    """Check GPU memory usage."""
    print_section("Memory Usage")
    
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return
    
    device_manager = GPUDeviceManager(target_device=0)
    info = device_manager.get_device_info()
    stats = device_manager.get_memory_stats()
    
    print_info("Current Device", f"cuda:{info['current_device']}")
    print_info("Device Name", info['device_name'])
    print_info("Total Memory", f"{info['total_memory_gb']:.2f} GB")
    print_info("Free Memory", f"{info['free_memory_gb']:.2f} GB")
    print_info("Used Memory", f"{info['used_memory_gb']:.2f} GB")
    print_info("Memory Utilization", f"{(info['used_memory'] / info['total_memory']) * 100:.1f}%")
    
    if 'gpu_utilization' in info:
        print_info("GPU Utilization", f"{info['gpu_utilization']}%")
        print_info("Memory Utilization", f"{info['memory_utilization']}%")
    else:
        print_info("GPU Utilization", "N/A (NVML not available)")
        print_info("Memory Utilization", "N/A (NVML not available)")
    
    print("\nPyTorch Memory Stats:")
    print_info("  Allocated", f"{stats['allocated_gb']:.2f} GB")
    print_info("  Reserved", f"{stats['reserved_gb']:.2f} GB")
    print_info("  Max Allocated", f"{stats['max_allocated_gb']:.2f} GB")
    print_info("  Max Reserved", f"{stats['max_reserved_gb']:.2f} GB")


def check_environment() -> None:
    """Check environment variables and configuration."""
    print_section("Environment Configuration")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print_info("CUDA_VISIBLE_DEVICES", cuda_visible or "Not set")
    
    if cuda_visible:
        try:
            visible_devices = [int(x) for x in cuda_visible.split(",")]
            if visible_devices[0] != 0:
                print("  WARNING: CUDA_VISIBLE_DEVICES should start with 0 for single A100 setup")
            else:
                print("  OK: CUDA_VISIBLE_DEVICES starts with 0")
        except ValueError:
            print("  ERROR: Invalid CUDA_VISIBLE_DEVICES format")
    
    # Check other relevant environment variables
    print_info("TOKENIZERS_PARALLELISM", os.environ.get("TOKENIZERS_PARALLELISM", "Not set"))
    print_info("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "Not set"))
    print_info("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "Not set"))
    
    # Check PyTorch optimizations
    print_info("Float32 Matmul Precision", torch.get_float32_matmul_precision())
    print_info("cuDNN Benchmark", torch.backends.cudnn.benchmark)
    print_info("cuDNN Deterministic", torch.backends.cudnn.deterministic)


def check_optional_dependencies() -> None:
    """Check optional dependencies for advanced monitoring."""
    print_section("Optional Dependencies")
    
    # Check NVML
    try:
        import pynvml
        print_info("pynvml", "Available")
        print_info("NVML Version", pynvml.nvmlSystemGetDriverVersion().decode())
    except ImportError:
        print_info("pynvml", "Not available")
        print("  Install with: pip install pynvml")
    
    # Check other optional dependencies
    optional_deps = [
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "flash_attn",
    ]
    
    for dep in optional_deps:
        try:
            __import__(dep)
            print_info(dep, "Available")
        except ImportError:
            print_info(dep, "Not available")


def check_training_readiness() -> None:
    """Check if the environment is ready for training."""
    print_section("Training Readiness Check")
    
    issues = []
    warnings = []
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        issues.append("CUDA not available - training will use CPU")
    else:
        # Check device count
        device_count = torch.cuda.device_count()
        if device_count == 0:
            issues.append("No CUDA devices detected")
        elif device_count > 1:
            warnings.append(f"Multiple GPUs detected ({device_count}) - using only GPU 0")
        
        # Check A100
        if device_count > 0:
            props = torch.cuda.get_device_properties(0)
            if "A100" not in props.name:
                warnings.append(f"Expected A100 GPU, found: {props.name}")
            
            # Check memory
            total_memory = props.total_memory / (1024**3)
            if total_memory < 30:  # A100 should have 40GB+
                warnings.append(f"Low GPU memory: {total_memory:.1f}GB (A100 has 40GB+)")
    
    # Check environment variables
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible and not cuda_visible.startswith("0"):
        warnings.append("CUDA_VISIBLE_DEVICES should start with 0")
    
    # Print results
    if issues:
        print("ISSUES (must be fixed):")
        for issue in issues:
            print(f"  ❌ {issue}")
    
    if warnings:
        print("\nWARNINGS (should be addressed):")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if not issues and not warnings:
        print("✅ Environment is ready for training!")
    elif not issues:
        print("✅ Environment is ready for training (with warnings)")
    else:
        print("❌ Environment has issues that must be fixed")


def main():
    """Main function to run all checks."""
    print("LMOL GPU Environment Check")
    print("=" * 60)
    
    # Set up GPU environment
    env_config = setup_gpu_environment()
    
    # Run all checks
    check_pytorch_installation()
    check_gpu_devices()
    check_memory_usage()
    check_environment()
    check_optional_dependencies()
    check_training_readiness()
    
    print_section("Summary")
    print("GPU check complete. Review any issues or warnings above.")
    print("\nFor training, run:")
    print("  python training/main.py --max_steps 2 --amp_dtype bf16 --cudnn_benchmark true --gpu_log_interval 1")


if __name__ == "__main__":
    main()
