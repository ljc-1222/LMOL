# LMOL GPU Performance Optimization Guide

This guide covers the GPU performance optimizations implemented for the LMOL training pipeline, specifically designed for NVIDIA A100 GPUs.

## Overview

The LMOL project has been enhanced with comprehensive GPU performance optimizations including:

- **Device Management**: Centralized GPU device handling with verification
- **Mixed Precision Training**: Optimized for A100 with bfloat16 support
- **Memory Management**: Advanced GPU memory monitoring and cleanup
- **DataLoader Optimization**: Efficient data transfer with pin_memory and non_blocking
- **Performance Monitoring**: Real-time GPU utilization and memory tracking
- **CuDNN Optimization**: Benchmark mode for consistent input sizes
- **torch.compile Support**: Optional model compilation for faster execution

## Quick Start

### 1. Environment Check

```bash
# Check GPU environment and PyTorch setup
python tools/gpu_check.py

# Comprehensive verification
python scripts/verify_gpu_setup.py
```

### 2. Quick Training Test

```bash
# Run a quick test with 2 steps
python training/main.py --max_steps 2 --amp_dtype bf16 --cudnn_benchmark true --gpu_log_interval 1

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi
```

### 3. Full Training

```bash
# Full training with A100 optimizations
python training/main.py --amp_dtype bf16 --cudnn_benchmark true --gpu_log_interval 100
```

## Configuration Options

### GPU Performance Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--amp_dtype` | `bf16` | Mixed precision: `bf16` (A100 optimized), `fp16`, or `none` |
| `--cudnn_benchmark` | `true` | Enable cuDNN benchmark for consistent input sizes |
| `--compile` | `false` | Enable torch.compile for faster execution |
| `--gpu_log_interval` | `100` | Log GPU stats every N steps (0 to disable) |

### Training Flags

| Flag | Description |
|------|-------------|
| `--max_steps` | Limit training steps (for testing) |
| `--batch_size` | Override per-device batch size |
| `--learning_rate` | Override learning rate |
| `--epochs` | Override number of epochs |

## Architecture Changes

### 1. Device Management (`utils/gpu_diag.py`)

- **GPUDeviceManager**: Centralized device management
- **Device Verification**: Automatic verification of model and batch placement
- **Memory Monitoring**: Real-time GPU memory and utilization tracking
- **A100 Optimizations**: Specific optimizations for A100 GPUs

### 2. Training Pipeline Updates

- **Canonical Device**: Single device management throughout training
- **Device Assertions**: Verification that all tensors are on correct device
- **Memory Cleanup**: Intelligent memory management during training
- **Performance Logging**: Regular GPU performance metrics

### 3. DataLoader Optimizations

- **pin_memory=True**: Faster GPU transfer
- **non_blocking=True**: Asynchronous data transfer
- **num_workers>=4**: Parallel data loading
- **persistent_workers=True**: Reduced worker startup overhead

### 4. Mixed Precision Training

- **bfloat16 Default**: Optimized for A100 architecture
- **GradScaler Management**: Automatic scaling for fp16, disabled for bf16
- **Autocast Context**: Proper mixed precision context management

## Performance Monitoring

### GPU Metrics Logged

Every `gpu_log_interval` steps, the following metrics are logged:

- **Memory Usage**: Allocated, reserved, peak allocated
- **GPU Utilization**: GPU and memory utilization (if NVML available)
- **Device Information**: Current device and compute capability

### Memory Management

- **Peak Memory Tracking**: Reset at epoch start
- **Memory Cleanup**: Automatic cleanup every N batches
- **Memory Warnings**: Alerts for high memory usage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 16`
   - Enable gradient checkpointing (already enabled)
   - Check for memory leaks in custom code

2. **Device Placement Errors**
   - Ensure all tensors are moved to device with `.to(device)`
   - Use `non_blocking=True` for DataLoader transfers
   - Verify model is on correct device before training

3. **Performance Issues**
   - Enable cuDNN benchmark: `--cudnn_benchmark true`
   - Use bfloat16: `--amp_dtype bf16`
   - Consider torch.compile: `--compile true`

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Run GPU diagnostics
python tools/gpu_check.py
```

## Best Practices

### 1. Device Management

- Always use the canonical device: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Move model to device once after creation: `model = model.to(device)`
- Use `non_blocking=True` for DataLoader transfers
- Verify device placement with assertions

### 2. Memory Management

- Reset peak memory stats at epoch start
- Use gradient checkpointing for large models
- Monitor memory usage regularly
- Clean up unused tensors

### 3. Performance Optimization

- Use bfloat16 on A100 for best performance
- Enable cuDNN benchmark for consistent input sizes
- Use appropriate batch sizes for your GPU memory
- Monitor GPU utilization to identify bottlenecks

### 4. Training Stability

- Start with conservative learning rates
- Use gradient clipping for stability
- Monitor gradient norms
- Use warmup for learning rate scheduling

## File Structure

```
LMOL/
├── utils/
│   └── gpu_diag.py              # GPU diagnostics and device management
├── tools/
│   └── gpu_check.py             # GPU environment check script
├── training/
│   ├── cli.py                   # CLI argument parsing
│   ├── main.py                  # Updated training pipeline
│   └── classification_trainer.py # Updated trainer with GPU monitoring
├── scripts/
│   └── verify_gpu_setup.py      # Comprehensive verification script
└── configs/
    └── config.py                # Updated configuration with GPU flags
```

## Verification Checklist

- [ ] GPU environment check passes
- [ ] Model loads on correct device
- [ ] DataLoader uses pin_memory and non_blocking
- [ ] Mixed precision training works
- [ ] GPU memory monitoring works
- [ ] Device placement assertions pass
- [ ] Training runs without errors
- [ ] GPU utilization is reasonable (>80%)

## Support

For issues related to GPU performance:

1. Run `python tools/gpu_check.py` to diagnose issues
2. Check the verification script output
3. Review GPU memory usage with `nvidia-smi`
4. Check PyTorch CUDA installation
5. Verify A100-specific optimizations are enabled

## Performance Expectations

On a single A100 GPU:

- **Memory Usage**: ~20-30GB for full model
- **Training Speed**: ~2-5 samples/second (depending on batch size)
- **GPU Utilization**: 80-95% during training
- **Memory Utilization**: 60-80% of 40GB A100

These optimizations should provide significant performance improvements over the baseline implementation.
