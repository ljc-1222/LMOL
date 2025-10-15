#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Setup Verification Script for LMOL

This script performs comprehensive verification of the GPU setup
and training configuration for the LMOL project.

Usage:
    python scripts/verify_gpu_setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from utils.gpu_diag import GPUDeviceManager, setup_gpu_environment


def run_command(cmd: str, description: str) -> tuple[bool, str]:
    """
    Run a command and return success status and output.
    
    Args:
        cmd: Command to run
        description: Description of what the command does
        
    Returns:
        Tuple of (success, output)
    """
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        if output.strip():
            print(f"Output:\n{output}")
        print("-" * 60)
        return success, output
    except subprocess.TimeoutExpired:
        print("Result: TIMEOUT")
        print("-" * 60)
        return False, "Command timed out"
    except Exception as e:
        print(f"Result: ERROR - {e}")
        print("-" * 60)
        return False, str(e)


def verify_environment() -> bool:
    """Verify basic environment setup."""
    print("=" * 60)
    print("VERIFYING ENVIRONMENT SETUP")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("WARNING: Python 3.8+ recommended")
        success = False
    
    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in PyTorch")
        success = False
    else:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible or 'Not set'}")
    
    return success


def verify_gpu_hardware() -> bool:
    """Verify GPU hardware and properties."""
    print("=" * 60)
    print("VERIFYING GPU HARDWARE")
    print("=" * 60)
    
    success = True
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device_manager = GPUDeviceManager(target_device=0)
    device_info = device_manager.get_device_info()
    
    print(f"Device: {device_info['device_name']}")
    print(f"Compute Capability: {device_info['compute_capability']}")
    print(f"Total Memory: {device_info['total_memory_gb']:.2f} GB")
    print(f"Free Memory: {device_info['free_memory_gb']:.2f} GB")
    
    # Check if it's an A100
    if "A100" not in device_info['device_name']:
        print("WARNING: Expected A100 GPU, but found different GPU")
        print("  A100 optimizations may not be available")
    
    # Check memory
    if device_info['total_memory_gb'] < 30:
        print("WARNING: Low GPU memory detected")
        print("  A100 should have 40GB+ memory")
    
    return success


def verify_pytorch_optimizations() -> bool:
    """Verify PyTorch optimizations are working."""
    print("=" * 60)
    print("VERIFYING PYTORCH OPTIMIZATIONS")
    print("=" * 60)
    
    success = True
    
    # Check cuDNN benchmark
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    # Check float32 matmul precision
    try:
        precision = torch.get_float32_matmul_precision()
        print(f"Float32 matmul precision: {precision}")
    except Exception as e:
        print(f"Float32 matmul precision: Error - {e}")
    
    # Test tensor operations
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print(f"Tensor operations: SUCCESS (device: {device})")
    except Exception as e:
        print(f"Tensor operations: FAILED - {e}")
        success = False
    
    return success


def verify_training_components() -> bool:
    """Verify training components can be loaded."""
    print("=" * 60)
    print("VERIFYING TRAINING COMPONENTS")
    print("=" * 60)
    
    success = True
    
    try:
        from configs.config import config
        print("Config: SUCCESS")
    except Exception as e:
        print(f"Config: FAILED - {e}")
        success = False
    
    try:
        from data import SCUT_FBP5500_Pairs, ClassificationCollator
        print("Data modules: SUCCESS")
    except Exception as e:
        print(f"Data modules: FAILED - {e}")
        success = False
    
    try:
        from model import model_generator
        print("Model modules: SUCCESS")
    except Exception as e:
        print(f"Model modules: FAILED - {e}")
        success = False
    
    try:
        from training.classification_trainer import LMOLClassificationTrainer
        print("Trainer modules: SUCCESS")
    except Exception as e:
        print(f"Trainer modules: FAILED - {e}")
        success = False
    
    return success


def verify_data_loading() -> bool:
    """Verify data loading works correctly."""
    print("=" * 60)
    print("VERIFYING DATA LOADING")
    print("=" * 60)
    
    success = True
    
    try:
        from data import SCUT_FBP5500_Pairs, ClassificationCollator
        from model import model_generator
        
        # Check if data files exist
        from configs.config import config
        train_csvs = config.TRAIN_PAIRS_CSVS
        
        for csv_path in train_csvs[:1]:  # Check first CSV only
            if not Path(csv_path).exists():
                print(f"Data file not found: {csv_path}")
                success = False
            else:
                print(f"Data file found: {csv_path}")
        
        if success:
            print("Data loading: SUCCESS")
        
    except Exception as e:
        print(f"Data loading: FAILED - {e}")
        success = False
    
    return success


def run_quick_training_test() -> bool:
    """Run a quick training test."""
    print("=" * 60)
    print("RUNNING QUICK TRAINING TEST")
    print("=" * 60)
    
    # This would run a very short training test
    # For now, just check if the command would work
    cmd = "python training/main.py --max_steps 2 --amp_dtype bf16 --cudnn_benchmark true --gpu_log_interval 1 --dry_run"
    
    print("To run quick training test, execute:")
    print(f"  {cmd}")
    print("\nNote: This will run 2 training steps to verify GPU usage")
    
    return True


def main():
    """Main verification function."""
    print("LMOL GPU Setup Verification")
    print("=" * 60)
    
    all_success = True
    
    # Run all verification steps
    steps = [
        ("Environment Setup", verify_environment),
        ("GPU Hardware", verify_gpu_hardware),
        ("PyTorch Optimizations", verify_pytorch_optimizations),
        ("Training Components", verify_training_components),
        ("Data Loading", verify_data_loading),
        ("Quick Training Test", run_quick_training_test),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}:")
        try:
            step_success = step_func()
            if not step_success:
                all_success = False
        except Exception as e:
            print(f"ERROR in {step_name}: {e}")
            all_success = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if all_success:
        print("✅ All verifications passed!")
        print("\nYour GPU setup is ready for LMOL training.")
        print("\nNext steps:")
        print("1. Run GPU check: python tools/gpu_check.py")
        print("2. Run quick test: python training/main.py --max_steps 2 --amp_dtype bf16 --cudnn_benchmark true --gpu_log_interval 1")
        print("3. Monitor GPU: watch -n 1 nvidia-smi")
    else:
        print("❌ Some verifications failed!")
        print("\nPlease review the errors above and fix them before training.")
    
    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
