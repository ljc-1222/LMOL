#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Sanity Check Runner for LMOL

This script runs a minimal training loop to verify that:
- Model can be loaded and moved to device
- Forward pass works correctly
- Loss computation is valid
- Backward pass works without errors
- Gradients are computed and non-zero
- Optimizer step works correctly

Usage:
    python -m tools.train_sanity --max_steps 3 --amp_dtype bf16 --grad_log_interval 1
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config import config
from utils.grad_diag import setup_gradient_diagnostics, assert_all_requires_grad
from utils.gpu_diag import GPUDeviceManager, setup_gpu_environment
from data import SCUT_FBP5500_Pairs, ClassificationCollator
from model import model_generator
from training.optimizer import create_optimizer
from training.classification_trainer import LMOLClassificationTrainer
from transformers import TrainingArguments


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LMOL Training Sanity Check")
    
    # Training parameters
    parser.add_argument("--max_steps", type=int, default=3, 
                       help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for sanity check")
    
    # AMP configuration
    parser.add_argument("--amp_dtype", type=str, choices=["bf16", "fp16", "none"], 
                       default="bf16", help="Mixed precision dtype")
    
    # Gradient diagnostics
    parser.add_argument("--grad_log_interval", type=int, default=1,
                       help="Gradient logging interval")
    parser.add_argument("--detect_anomaly", action="store_true",
                       help="Enable autograd anomaly detection")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                       help="Global gradient clipping threshold (0 = disabled)")
    
    # Data configuration
    parser.add_argument("--train_csv", type=str, 
                       default="data/pairs/train_fold1_45000.csv",
                       help="Training CSV file")
    
    return parser.parse_args()


def setup_device_and_environment():
    """Set up GPU environment and device."""
    print("=" * 80)
    print("LMOL TRAINING SANITY CHECK")
    print("=" * 80)
    
    # Set up GPU environment
    print("[INFO] Setting up GPU environment...")
    env_config = setup_gpu_environment()
    device_manager = GPUDeviceManager(target_device=0)
    device = device_manager.get_device()
    
    # Print GPU information
    device_info = device_manager.get_device_info()
    print(f"[INFO] Using device: {device_info['device_name']} (CUDA {device_info['compute_capability']})")
    print(f"[INFO] Total memory: {device_info['total_memory_gb']:.2f} GB")
    print(f"[INFO] Free memory: {device_info['free_memory_gb']:.2f} GB")
    
    # Set up PyTorch optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] cuDNN benchmark: True")
        
        # Set mixed precision dtype
        if args.amp_dtype == "bf16":
            print(f"[INFO] Using bfloat16 mixed precision (A100 optimized)")
        elif args.amp_dtype == "fp16":
            print(f"[INFO] Using float16 mixed precision")
        else:
            print(f"[INFO] Using full precision (no AMP)")
    else:
        print("[WARN] CUDA not available, falling back to CPU training")
    
    return device, device_manager


def load_model_and_data(device, args):
    """Load model and data for sanity check."""
    print("\n[INFO] Loading model and data...")
    
    # Load model
    model, processor, tokenizer, fast_processor = model_generator()
    model.train()
    
    # Disable built-in loss computation
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    # Move model to device BEFORE optimizer creation
    model = model.to(device)
    print(f"[INFO] Model moved to device: {device}")
    
    # Verify model device placement
    model_device = next(model.parameters()).device
    if model_device != device:
        raise RuntimeError(f"Model not on correct device: {model_device} != {device}")
    
    # Load dataset
    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    
    ds = SCUT_FBP5500_Pairs(str(train_csv))
    print(f"[INFO] Loaded dataset with {len(ds)} samples")
    
    # Create data collator
    collator = ClassificationCollator(processor, tokenizer, max_length=config.MAX_SEQ_LEN, is_training=True)
    
    return model, processor, tokenizer, ds, collator


def setup_optimizer_and_diagnostics(model, device, args):
    """Set up optimizer and gradient diagnostics."""
    print("\n[INFO] Setting up optimizer and diagnostics...")
    
    # Create optimizer
    optimizer = create_optimizer(model, config.WEIGHT_DECAY)
    
    # Print optimizer information
    print(f"[INFO] Optimizer: {type(optimizer).__name__}")
    for i, group in enumerate(optimizer.param_groups):
        lr = group.get('lr', 0.0)
        weight_decay = group.get('weight_decay', 0.0)
        param_count = len(group['params'])
        print(f"  Group {i}: {param_count} params, LR={lr:.2e}, WD={weight_decay:.2e}")
    
    # Set up gradient diagnostics
    diagnostics = setup_gradient_diagnostics(
        model=model,
        log_interval=args.grad_log_interval,
        clip_grad_norm=args.clip_grad_norm,
        detect_anomaly=args.detect_anomaly
    )
    
    # Validate training setup
    diagnostics.validate_training_setup(model, optimizer)
    
    return optimizer, diagnostics


def run_sanity_check(model, ds, collator, optimizer, diagnostics, device, args):
    """Run the actual sanity check training loop."""
    print(f"\n[INFO] Running sanity check for {args.max_steps} steps...")
    print("-" * 80)
    
    # Set up mixed precision
    if args.amp_dtype == "bf16":
        autocast_dtype = torch.bfloat16
        scaler = None
        print(f"[INFO] Using bf16 mixed precision (no scaler needed)")
    elif args.amp_dtype == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        print(f"[INFO] Using fp16 mixed precision with GradScaler")
    else:
        autocast_dtype = None
        scaler = None
        print(f"[INFO] Using full precision (no AMP)")
    
    # Check for in-place operations
    from utils.grad_diag import check_inplace_operations
    inplace_issues = check_inplace_operations(model)
    if inplace_issues:
        print(f"[WARNING] Found {len(inplace_issues)} potential in-place operation issues:")
        for issue in inplace_issues:
            print(f"  - {issue}")
    else:
        print(f"[INFO] No in-place operation issues detected")
    
    # Check numerical stability
    from utils.seed import check_numerical_stability
    print(f"[INFO] Checking numerical stability...")
    dummy_input = torch.randint(0, 1000, (1, 10), device=device)
    stability_check = check_numerical_stability(model, dummy_input, device)
    
    if stability_check.get('is_stable', False):
        print(f"[PASS] Model is numerically stable")
    else:
        print(f"[WARNING] Model numerical stability issues detected:")
        if stability_check.get('output_has_nan', False):
            print(f"  - Output contains NaN values")
        if stability_check.get('output_has_inf', False):
            print(f"  - Output contains Inf values")
        if stability_check.get('nan_params', 0) > 0:
            print(f"  - {stability_check['nan_params']} parameters contain NaN")
        if stability_check.get('inf_params', 0) > 0:
            print(f"  - {stability_check['inf_params']} parameters contain Inf")
    
    # Create data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )
    
    # Training loop
    model.train()
    step = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if step >= args.max_steps:
            break
        
        print(f"\n[STEP {step+1}] Processing batch {batch_idx+1}...")
        
        # Move batch to device
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Print batch information
        print(f"  Batch size: {batch['input_ids'].shape[0]}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Labels shape: {batch['label_ids'].shape}")
        print(f"  Labels: {batch['label_ids'].tolist()}")
        
        # Check batch device placement
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.device != device:
                    print(f"  [ERROR] Batch tensor '{key}' on {tensor.device}, expected {device}")
                    return False
        
        # Check for empty batches
        if batch['input_ids'].shape[0] == 0:
            print(f"  [ERROR] Empty batch detected!")
            return False
        
        # Zero gradients with proper method
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision - disable built-in loss computation
        if autocast_dtype is not None:
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                # Remove labels to prevent built-in loss computation
                inputs_without_labels = {k: v for k, v in batch.items() if k not in ['label_ids', 'pair_ids', 'labels']}
                outputs = model(**inputs_without_labels)
                
                # Compute loss (simplified version)
                logits = outputs.logits
                last_token_logits = logits[:, -1, :]
                
                # Use simple classification approach for sanity check
                first_token_id = 0
                second_token_id = 1  
                similar_token_id = 2
                
                # Extract classification logits
                classification_logits = torch.stack([
                    last_token_logits[:, first_token_id],
                    last_token_logits[:, second_token_id],
                    last_token_logits[:, similar_token_id]
                ], dim=1)
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(
                    classification_logits,
                    batch['label_ids'],
                    reduction='mean'
                )
        else:
            # Remove labels to prevent built-in loss computation
            inputs_without_labels = {k: v for k, v in batch.items() if k not in ['label_ids', 'pair_ids', 'labels']}
            outputs = model(**inputs_without_labels)
            logits = outputs.logits
            last_token_logits = logits[:, -1, :]
            
            # Simple classification approach
            first_token_id = 0
            second_token_id = 1
            similar_token_id = 2
            
            classification_logits = torch.stack([
                last_token_logits[:, first_token_id],
                last_token_logits[:, second_token_id],
                last_token_logits[:, similar_token_id]
            ], dim=1)
            
            loss = torch.nn.functional.cross_entropy(
                classification_logits,
                batch['label_ids'],
                reduction='mean'
            )
        
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Loss requires_grad: {loss.requires_grad}")
        print(f"  Loss device: {loss.device}")
        
        # Check for loss anomalies
        if not torch.isfinite(loss):
            print(f"  [ERROR] Loss is not finite: {loss.item()}")
            return False
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        print(f"  Backward pass completed")
        
        # Check gradients comprehensively
        grad_norm = diagnostics.compute_global_grad_norm(model)
        print(f"  Global gradient norm: {grad_norm:.6f}")
        
        if grad_norm == 0.0:
            print(f"  [ERROR] Global gradient norm is zero!")
            return False
        
        # Check individual parameter gradients
        zero_grad_count = 0
        nan_inf_count = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is None:
                    print(f"  [WARNING] Parameter '{name}' has no gradient (may be unused in this batch)")
                    # Don't fail for unused parameters, just warn
                elif param.grad.norm() == 0.0:
                    zero_grad_count += 1
                elif not torch.isfinite(param.grad).all():
                    nan_inf_count += 1
                    print(f"  [ERROR] Parameter '{name}' has NaN/Inf gradients!")
                    return False
        
        print(f"  Gradient health: {total_params - zero_grad_count}/{total_params} non-zero, "
              f"{nan_inf_count} NaN/Inf")
        
        if zero_grad_count > total_params * 0.5:
            print(f"  [WARNING] {zero_grad_count}/{total_params} parameters have zero gradients!")
        
        # Apply gradient clipping
        if args.clip_grad_norm > 0:
            clipped_norm = diagnostics.apply_gradient_clipping(model)
            print(f"  Gradient norm after clipping: {clipped_norm:.6f}")
        
        # Check AMP underflow
        if scaler is not None:
            diagnostics.check_amp_underflow(scaler)
            print(f"  AMP scaler scale: {scaler.get_scale():.2e}")
        
        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        print(f"  Optimizer step completed")
        
        # Log gradient statistics
        diagnostics.log_gradient_stats(step, model)
        
        step += 1
    
    print(f"\n[SUCCESS] Sanity check completed for {step} steps!")
    return True


def main():
    """Main sanity check function."""
    global args
    args = parse_args()
    
    try:
        # Set up device and environment
        device, device_manager = setup_device_and_environment()
        
        # Load model and data
        model, processor, tokenizer, ds, collator = load_model_and_data(device, args)
        
        # Set up optimizer and diagnostics
        optimizer, diagnostics = setup_optimizer_and_diagnostics(model, device, args)
        
        # Run sanity check
        success = run_sanity_check(model, ds, collator, optimizer, diagnostics, device, args)
        
        # Cleanup
        diagnostics.cleanup()
        
        if success:
            print("\n" + "=" * 80)
            print("SANITY CHECK PASSED - Training setup is working correctly!")
            print("=" * 80)
            return 0
        else:
            print("\n" + "=" * 80)
            print("SANITY CHECK FAILED - Issues detected in training setup!")
            print("=" * 80)
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] Sanity check failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
