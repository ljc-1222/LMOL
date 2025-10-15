# -*- coding: utf-8 -*-
"""
Command Line Interface for LMOL Training

This module provides CLI argument parsing for the LMOL training pipeline
with GPU performance optimization flags.

Usage:
    python training/main.py --amp_dtype bf16 --cudnn_benchmark true --gpu_log_interval 100
"""

import argparse
from typing import Any, Dict


def parse_training_args() -> Dict[str, Any]:
    """
    Parse command line arguments for LMOL training.
    
    Returns:
        Dictionary containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="LMOL Training with GPU Performance Optimizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # GPU Performance Arguments
    gpu_group = parser.add_argument_group("GPU Performance")
    gpu_group.add_argument(
        "--amp_dtype",
        type=str,
        choices=["bf16", "fp16", "none"],
        default="bf16",
        help="Mixed precision dtype: bf16 (A100 optimized), fp16, or none"
    )
    gpu_group.add_argument(
        "--cudnn_benchmark",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Enable cuDNN benchmark for consistent input sizes"
    )
    gpu_group.add_argument(
        "--compile",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Enable torch.compile for faster execution"
    )
    gpu_group.add_argument(
        "--gpu_log_interval",
        type=int,
        default=100,
        help="Log GPU memory and utilization every N steps (0 to disable)"
    )
    gpu_group.add_argument(
        "--grad_log_interval",
        type=int,
        default=10,
        help="Gradient statistics logging interval (0 = disabled)"
    )
    gpu_group.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.0,
        help="Global gradient clipping threshold (0 = disabled)"
    )
    gpu_group.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable PyTorch autograd anomaly detection"
    )
    
    # Training Arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (for testing)"
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Per-device batch size (overrides config)"
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    # Data Arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory path (overrides config)"
    )
    data_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (overrides config)"
    )
    
    # Debug Arguments
    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    debug_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform a dry run without actual training"
    )
    debug_group.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable fully deterministic training (slower but reproducible)"
    )
    debug_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Convert to dictionary
    args_dict = vars(args)
    
    # Convert string booleans to actual booleans
    args_dict["cudnn_benchmark"] = args_dict["cudnn_benchmark"] == "true"
    args_dict["compile"] = args_dict["compile"] == "true"
    
    return args_dict


def apply_cli_overrides(config, args: Dict[str, Any]) -> None:
    """
    Apply CLI argument overrides to config.
    
    Args:
        config: Configuration object to modify
        args: Parsed CLI arguments
    """
    # GPU Performance overrides
    if args.get("amp_dtype"):
        config.AMP_DTYPE = args["amp_dtype"]
    
    if args.get("cudnn_benchmark") is not None:
        config.CUDNN_BENCHMARK = args["cudnn_benchmark"]
    
    if args.get("compile") is not None:
        config.USE_TORCH_COMPILE = args["compile"]
    
    if args.get("gpu_log_interval") is not None:
        config.GPU_LOG_INTERVAL = args["gpu_log_interval"]
    
    if args.get("grad_log_interval") is not None:
        config.GRAD_LOG_INTERVAL = args["grad_log_interval"]
    
    if args.get("clip_grad_norm") is not None:
        config.GRADIENT_CLIP_NORM = args["clip_grad_norm"]
    
    if args.get("detect_anomaly") is not None:
        config.DETECT_ANOMALY = args["detect_anomaly"]
    
    # Training overrides
    if args.get("batch_size"):
        config.PER_DEVICE_TRAIN_BATCH_SIZE = args["batch_size"]
    
    if args.get("learning_rate"):
        config.LR_LORA = args["learning_rate"]
        config.LR_PROJECTION = args["learning_rate"] * 10  # Keep 10x ratio
    
    if args.get("epochs"):
        config.NUM_EPOCHS = args["epochs"]
    
    # Data overrides
    if args.get("data_dir"):
        config.IMAGE_DIR = args["data_dir"]
    
    if args.get("output_dir"):
        config.OUTPUT_DIR = args["output_dir"]


def print_training_banner(config, args: Dict[str, Any]) -> None:
    """
    Print training configuration banner.
    
    Args:
        config: Configuration object
        args: Parsed CLI arguments
    """
    print("\n" + "="*80)
    print("  LMOL Training Configuration")
    print("="*80)
    
    # GPU Configuration
    print(f"GPU Configuration:")
    print(f"  AMP Dtype:        {config.AMP_DTYPE}")
    print(f"  cuDNN Benchmark:  {config.CUDNN_BENCHMARK}")
    print(f"  torch.compile:    {config.USE_TORCH_COMPILE}")
    print(f"  GPU Log Interval: {config.GPU_LOG_INTERVAL}")
    
    # Training Configuration
    print(f"\nTraining Configuration:")
    print(f"  Batch Size:       {config.PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  Learning Rate:    {config.LR_LORA} (LoRA), {config.LR_PROJECTION} (Projector)")
    print(f"  Epochs:           {config.NUM_EPOCHS}")
    print(f"  Max Steps:        {args.get('max_steps', 'All')}")
    
    # Data Configuration
    print(f"\nData Configuration:")
    print(f"  Image Directory:  {config.IMAGE_DIR}")
    print(f"  Output Directory: {config.OUTPUT_DIR}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test CLI parsing
    args = parse_training_args()
    print("Parsed arguments:")
    for key, value in args.items():
        print(f"  {key}: {value}")
