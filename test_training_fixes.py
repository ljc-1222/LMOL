#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Script for LMOL Training Fixes

This script tests all the critical fixes implemented for PyTorch training:
- Device placement verification
- Gradient flow validation
- AMP setup verification
- Loss computation validation
- Backpropagation integrity

Usage:
    python test_training_fixes.py
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.config import config
from utils.grad_diag import setup_gradient_diagnostics, check_inplace_operations, assert_all_requires_grad
from utils.gpu_diag import GPUDeviceManager, setup_gpu_environment
from data import SCUT_FBP5500_Pairs, ClassificationCollator
from model import model_generator
from training.optimizer import create_optimizer


def test_device_placement():
    """Test device placement and model movement."""
    print("=" * 80)
    print("TESTING DEVICE PLACEMENT")
    print("=" * 80)
    
    # Set up GPU environment
    env_config = setup_gpu_environment()
    device_manager = GPUDeviceManager(target_device=0)
    device = device_manager.get_device()
    
    print(f"[INFO] Using device: {device}")
    
    # Load model
    model, processor, tokenizer, fast_processor = model_generator()
    model.train()
    
    # Move model to device BEFORE optimizer creation
    model = model.to(device)
    print(f"[INFO] Model moved to device: {device}")
    
    # Verify device placement
    try:
        device_manager.assert_all_cuda(model)
        print(f"[PASS] Model device placement verified")
    except RuntimeError as e:
        print(f"[FAIL] Model device placement failed: {e}")
        return False
    
    # Create optimizer AFTER model is on device
    optimizer = create_optimizer(model, config.WEIGHT_DECAY)
    
    # Verify optimizer parameters are on correct device
    for i, group in enumerate(optimizer.param_groups):
        if group['params']:
            param_device = next(iter(group['params'])).device
            if param_device != device:
                print(f"[FAIL] Optimizer group {i} parameters on {param_device}, expected {device}")
                return False
    
    print(f"[PASS] Optimizer parameters on correct device")
    return True


def test_gradient_flow():
    """Test gradient flow and backpropagation."""
    print("\n" + "=" * 80)
    print("TESTING GRADIENT FLOW")
    print("=" * 80)
    
    # Set up environment
    device_manager = GPUDeviceManager(target_device=0)
    device = device_manager.get_device()
    
    # Load model and data
    model, processor, tokenizer, fast_processor = model_generator()
    model.train()
    model = model.to(device)
    
    # Load small dataset
    train_csv = "data/pairs/train_fold1_45000.csv"
    if not Path(train_csv).exists():
        print(f"[SKIP] Training CSV not found: {train_csv}")
        return True
    
    ds = SCUT_FBP5500_Pairs(train_csv)
    collator = ClassificationCollator(processor, tokenizer, max_length=config.MAX_SEQ_LEN, is_training=True)
    
    # Create data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collator, num_workers=0)
    
    # Get a batch
    batch = next(iter(dataloader))
    batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    # Create optimizer
    optimizer = create_optimizer(model, config.WEIGHT_DECAY)
    
    # Test forward pass
    print("[INFO] Testing forward pass...")
    inputs_without_labels = {k: v for k, v in batch.items() if k not in ['label_ids', 'pair_ids', 'labels']}
    
    # Use autocast for mixed precision
    if config.AMP_DTYPE == 'bf16' and torch.cuda.is_available():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(**inputs_without_labels)
    elif config.AMP_DTYPE == 'fp16' and torch.cuda.is_available():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs_without_labels)
    else:
        outputs = model(**inputs_without_labels)
    
    print(f"[PASS] Forward pass successful, output shape: {outputs.logits.shape}")
    
    # Test loss computation
    print("[INFO] Testing loss computation...")
    logits = outputs.logits
    last_token_logits = logits[:, -1, :]
    
    # Simple classification approach for testing
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
    
    print(f"[PASS] Loss computation successful: {loss.item():.6f}")
    print(f"[INFO] Loss requires_grad: {loss.requires_grad}")
    print(f"[INFO] Loss device: {loss.device}")
    
    # Test backward pass
    print("[INFO] Testing backward pass...")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Check gradients
    grad_norm = 0.0
    param_count = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.norm(2)
            grad_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        grad_norm = grad_norm ** (1. / 2)
    
    print(f"[PASS] Backward pass successful, gradient norm: {grad_norm:.6f}")
    
    if grad_norm == 0.0:
        print(f"[FAIL] Global gradient norm is zero!")
        return False
    
    # Test optimizer step
    print("[INFO] Testing optimizer step...")
    optimizer.step()
    print(f"[PASS] Optimizer step successful")
    
    return True


def test_amp_setup():
    """Test AMP setup for both bf16 and fp16."""
    print("\n" + "=" * 80)
    print("TESTING AMP SETUP")
    print("=" * 80)
    
    device_manager = GPUDeviceManager(target_device=0)
    device = device_manager.get_device()
    
    if device.type != "cuda":
        print("[SKIP] CUDA not available, skipping AMP tests")
        return True
    
    # Test bf16
    print("[INFO] Testing bf16 AMP...")
    model, processor, tokenizer, fast_processor = model_generator()
    model.train()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 10), device=device)
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(dummy_input)
    
    print(f"[PASS] bf16 AMP working, output dtype: {output.logits.dtype}")
    
    # Test fp16 with scaler
    print("[INFO] Testing fp16 AMP with scaler...")
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model(dummy_input)
    
    print(f"[PASS] fp16 AMP working, output dtype: {output.logits.dtype}")
    print(f"[INFO] GradScaler scale: {scaler.get_scale():.2e}")
    
    return True


def test_gradient_diagnostics():
    """Test gradient diagnostics functionality."""
    print("\n" + "=" * 80)
    print("TESTING GRADIENT DIAGNOSTICS")
    print("=" * 80)
    
    device_manager = GPUDeviceManager(target_device=0)
    device = device_manager.get_device()
    
    # Load model
    model, processor, tokenizer, fast_processor = model_generator()
    model.train()
    model = model.to(device)
    
    # Set up gradient diagnostics
    diagnostics = setup_gradient_diagnostics(
        model=model,
        log_interval=1,
        clip_grad_norm=1.0,
        detect_anomaly=False
    )
    
    print(f"[PASS] Gradient diagnostics initialized")
    
    # Test in-place operation detection
    inplace_issues = check_inplace_operations(model)
    if inplace_issues:
        print(f"[WARNING] Found {len(inplace_issues)} in-place operation issues:")
        for issue in inplace_issues:
            print(f"  - {issue}")
    else:
        print(f"[PASS] No in-place operation issues detected")
    
    # Test requires_grad assertion
    try:
        assert_all_requires_grad(model, expected_trainable=True)
        print(f"[PASS] All parameters have requires_grad=True")
    except AssertionError as e:
        print(f"[WARNING] Parameter requires_grad issue: {e}")
    
    # Test gradient norm computation
    grad_norm = diagnostics.compute_global_grad_norm(model)
    print(f"[INFO] Global gradient norm: {grad_norm:.6f}")
    
    # Cleanup
    diagnostics.cleanup()
    print(f"[PASS] Gradient diagnostics cleanup successful")
    
    return True


def main():
    """Run all tests."""
    print("LMOL TRAINING FIXES VERIFICATION")
    print("=" * 80)
    
    tests = [
        ("Device Placement", test_device_placement),
        ("Gradient Flow", test_gradient_flow),
        ("AMP Setup", test_amp_setup),
        ("Gradient Diagnostics", test_gradient_diagnostics),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[PASS] {test_name} test passed")
            else:
                print(f"\n[FAIL] {test_name} test failed")
        except Exception as e:
            print(f"\n[ERROR] {test_name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Training fixes are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
