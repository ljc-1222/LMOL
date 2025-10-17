# -*- coding: utf-8 -*-
"""
LMOL Training Main Orchestration

This module provides the main training orchestration for the LMOL project:
- Single fold training pipeline
- Main training coordination
- Training utilities and validation

Key Features:
- Complete training pipeline for one fold
- Comprehensive configuration and validation
- Performance optimizations and monitoring
- Clean training output and logging
"""

import json
import math
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import torch
from transformers import TrainingArguments

# Suppress verbose logging from various libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Fix HuggingFace tokenizers parallelism warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress verbose output from various libraries
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

from configs.config import config
from utils.constants import HEADER_SEPARATOR_LENGTH
from utils import set_seed
from utils.csv_logger import create_training_logger, CSVTrainLogger
from utils.gpu_diag import GPUDeviceManager, setup_gpu_environment, get_canonical_device
from data import SCUT_FBP5500_Pairs, ClassificationCollator
from model import model_generator
from .cli import parse_training_args, apply_cli_overrides, print_training_banner
# Lazy imports to avoid dependency issues
# from .callbacks import SaveBestTrainingLossCallback
# from .optimizer import group_parameters_for_optimizer, create_optimizer, create_scheduler


def _fmt_num(n: float, precision: int = 6) -> str:
    """
    Format number with thousands separator and fixed precision.
    
    Args:
        n: Number to format (int or float)
        precision: Decimal precision for float numbers
        
    Returns:
        Formatted string with thousands separators
    """
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:,.{precision}f}"


def _print_header(title: str):
    """
    Print a formatted header with separator lines.
    
    Args:
        title: Header title to display
    """
    sep = "=" * HEADER_SEPARATOR_LENGTH
    print(f"\n{sep}\n  {title}\n{sep}\n")


def resolve_path(p: Union[str, Path]) -> Path:
    """
    Convert string path to Path object if needed.
    
    Args:
        p: Path string or Path object
        
    Returns:
        Path object
    """
    return p if isinstance(p, Path) else Path(p)


def ensure_exists(p: Path) -> Path:
    """
    Ensure path exists, raise error if not.
    
    Args:
        p: Path to check
        
    Returns:
        Path object if exists
        
    Raises:
        FileNotFoundError: If path does not exist
    """
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p


def first_batch_sanity_check(trainer):
    """
    Perform sanity check on first training batch.
    
    Validates that the data collator produces the expected batch structure:
    - Required keys present (input_ids, attention_mask, labels, pixel_values, label_ids, pair_ids)
    - Correct batch dimensions
    - Proper pair ID alignment for swap consistency
    - Valid answer spans in labels
    
    Args:
        trainer: Trainer instance to check
        
    Raises:
        AssertionError: If batch structure is invalid
    """
    batch = next(iter(trainer.get_train_dataloader()))
    req = ["input_ids", "attention_mask", "labels", "pixel_values", "label_ids", "pair_ids"]
    
    # Check required keys
    for k in req:
        assert k in batch, f"missing '{k}'"
    
    # Check batch dimensions
    N = batch["input_ids"].shape[0]
    assert N > 0, "Empty batch"
    assert batch["pixel_values"].shape[0] == 2 * N, "pixel_values must be 2 images per text sample"
    
    # Check pair ID alignment
    p = batch["pair_ids"].tolist()
    for i in range(0, len(p), 2):
        assert i//2 == p[i] == p[i+1], "pair_ids must be [0,0,1,1,...]"
    
    # Check answer spans
    labels = batch["labels"]
    nz = torch.nonzero(labels != -100, as_tuple=False)
    assert nz.numel() > 0, "no answer span"


def train_one_fold(fold_idx: int, train_csv: Path, out_dir: Path, max_steps: Optional[int] = None):
    """
    Train LMOL model for a single fold using PyTorch's default single-process handling.
    
    This function implements the complete training pipeline for one fold:
    1. Set up reproducible random seeds
    2. Load dataset and create data collator
    3. Initialize model with LoRA and custom projector
    4. Set up dual learning rate optimizer and scheduler
    5. Configure training arguments with performance optimizations
    6. Train with weighted cross-entropy + consistency loss
    7. Save final model and training metadata
    
    The training uses PyTorch's implicit single-process handling which can
    automatically utilize available GPUs if configured by the framework.
    
    Args:
        fold_idx: Fold index (1-based)
        train_csv: Path to training CSV file
        out_dir: Output directory for this fold
    """
    # Set up GPU environment and device management
    print(f"[INFO] Setting up GPU environment...")
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
        # Set cuDNN benchmark based on config
        torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
        print(f"[INFO] cuDNN benchmark: {config.CUDNN_BENCHMARK}")
        
        # Set mixed precision dtype
        if config.AMP_DTYPE == "bf16":
            print(f"[INFO] Using bfloat16 mixed precision (A100 optimized)")
            # Set high precision for matrix multiplications on A100
            torch.set_float32_matmul_precision("high")
        elif config.AMP_DTYPE == "fp16":
            print(f"[INFO] Using float16 mixed precision")
        else:
            print(f"[INFO] Using full precision (no AMP)")
    else:
        print("[WARN] CUDA not available, falling back to CPU training")
    
    # Suppress warnings during training
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*use_cache=True.*is incompatible with gradient checkpointing.*")
    warnings.filterwarnings("ignore", message=".*loss_type=None.*was set in the config but it is unrecognised.*")
    
    # Lazy imports to avoid dependency issues
    from training.classification_trainer import LMOLClassificationTrainer
    from training.callbacks import SaveBestTrainingLossCallback
    from training.optimizer import group_parameters_for_optimizer, create_optimizer, create_scheduler
    
    _print_header(f"Fold {fold_idx} Training")
    
    # Set reproducible seed for this fold (same seed across all processes)
    seed = 42 + fold_idx
    set_seed(seed, enable_numerical_stability=True)
    
    # Initialize dataset, model, and processor
    ds = SCUT_FBP5500_Pairs(str(train_csv))
    model, processor, tokenizer, fast_processor = model_generator()
    model.train()
    
    # CRITICAL: Move model to canonical device BEFORE optimizer creation
    model = model.to(device)
    print(f"[INFO] Model moved to device: {device}")
    
    # Verify model device placement
    try:
        device_manager.assert_all_cuda(model)
        print(f"[INFO] Model device placement verified: all parameters on {device}")
    except RuntimeError as e:
        print(f"[ERROR] Model device placement failed: {e}")
        raise
    
    # Apply torch.compile if enabled
    if config.USE_TORCH_COMPILE and device.type == "cuda":
        try:
            print(f"[INFO] Applying torch.compile with mode: {config.TORCH_COMPILE_MODE}")
            model = torch.compile(model, mode=config.TORCH_COMPILE_MODE)
            print(f"[INFO] torch.compile applied successfully")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
            print(f"[INFO] Continuing without torch.compile")

    # Calculate training parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Account for swap doubling: 45,000 pairs → 90,000 samples
    effective_samples = len(ds) * 2 if config.SWAP_DOUBLE else len(ds)
    
    # Calculate effective batch size (no DDP scaling)
    effective_batch_size = config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS * 2
    
    steps_per_epoch = math.ceil(effective_samples / effective_batch_size)
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    
    # Override total_steps if max_steps is provided (for testing)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
        print(f"[INFO] Limited to {max_steps} steps for testing")
    
    warmup_steps = int(math.ceil(total_steps * config.LR_WARMUP_RATIO))
    
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters detected")
    
    # Verify optimizer setup
    param_groups = group_parameters_for_optimizer(model)
    print(f"  Optimizer: {len(param_groups)} parameter groups")
    for i, group in enumerate(param_groups):
        group_name = "Projector" if i == 0 else "LoRA"
        group_params = sum(p.numel() for p in group['params'])
        print(f"    {group_name}: {group_params:,} parameters, Learning Rate (LR) = {group['lr']:.4e}")
    
    
    # Analyze class distribution in training data
    print(f"Class Distribution Analysis:")
    from collections import Counter
    class_counts = Counter()
    for sample in ds:
        class_counts[sample.label] += 1
    
    total = sum(class_counts.values())
    for label in sorted(class_counts.keys()):
        count = class_counts[label]
        print(f"  {label:8s}: {count:6,} samples ({count/total:6.2%})")
    
    # Check if weights are appropriate
    print(f"\nClass Weights:")
    print(f"  First:   1.00 (default)")
    print(f"  Second:  1.00 (default)")
    print(f"  Similar: {config.WSIM:.2f} (config.WSIM)")
    
    # Calculate recommended weights based on inverse frequency
    if len(class_counts) == 3:
        weight_first = total / (3 * max(class_counts.get(config.ANSWER_FIRST, 1), 1))
        weight_second = total / (3 * max(class_counts.get(config.ANSWER_SECOND, 1), 1))
        weight_similar = total / (3 * max(class_counts.get(config.ANSWER_SIMILAR, 1), 1))
        print(f"\nRecommended Inverse Frequency Weights:")
        print(f"  First:   {weight_first:.2f}")
        print(f"  Second:  {weight_second:.2f}")
        print(f"  Similar: {weight_similar:.2f}")
    print()

    # Calculate batches per epoch
    batches_per_epoch = steps_per_epoch
    
    # Print comprehensive training configuration
    print(f"Configuration:")
    print(f"  Data:      {len(ds):,} pairs, using CSV file `{train_csv.name}`")
    print(f"  Model:     {trainable_params:,}/{total_params:,} parameters, with {trainable_params:,} trainable parameters, which is {trainable_params/total_params:.4%} trainable")
    print(f"  Batch:     `{config.PER_DEVICE_TRAIN_BATCH_SIZE} × {config.GRADIENT_ACCUMULATION_STEPS} × 2 swap = {effective_batch_size} samples per batch`")
    print(f"  Batches:   {batches_per_epoch:,} batches per epoch (calculated as {effective_samples:,} ÷ {effective_batch_size})")
    print(f"  LR:        Projector = {config.LR_PROJECTION:.4e}, LoRA = {config.LR_LORA:.4e}, Weight Decay (WD) = {config.WEIGHT_DECAY:.4e}")
    print(f"  Loss:      WSIM = {config.WSIM:.4f}, CONS = {config.CONS_WEIGHT:.4f}, SWAP_CE = {getattr(config, 'SWAP_CE_WEIGHT', 1.0):.4f}")
    
    # Format schedule information with LR scheduler type
    schedule_type = config.LR_SCHEDULE_TYPE if getattr(config, 'USE_LR_SCHEDULING', False) else "constant"
    print(f"  Schedule:  {config.NUM_EPOCHS} epoch × {batches_per_epoch:,} batches = {batches_per_epoch * config.NUM_EPOCHS:,} total batches. Warmup is {warmup_steps:,} batches ({config.LR_WARMUP_RATIO:.4%}), and the LR schedule is `{schedule_type}`")
    
    # DataLoader workers configuration
    print(f"  WORKERS:   {config.DATALOADER_NUM_WORKERS}")
    
    print(f"  Speedup:   Workers = {config.DATALOADER_NUM_WORKERS}, PinMem = {config.DATALOADER_PIN_MEMORY}, " +
          f"FastProc = {fast_processor}, Compile = {getattr(config, 'USE_TORCH_COMPILE', False)}, " +
          f"FlashAttn = {getattr(config, 'USE_FLASH_ATTENTION', False)}")
    print(f"  Seed:      {seed}")
    print()

    # Create data collator (classification approach only)
    collator = ClassificationCollator(processor, tokenizer, max_length=config.MAX_SEQ_LEN, is_training=True)
    print(f"[TRAINING] Using classification-based training approach")

    # Configure training arguments
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LR_LORA,  # Base LR; projector has its own in optimizer groups
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=config.LOGGING_STEPS,
        logging_strategy="steps",
        save_strategy="no",  # Saving handled manually (best/last)
        fp16=(config.AMP_DTYPE == "fp16"), bf16=(config.AMP_DTYPE == "bf16"),  # Use new AMP_DTYPE config
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
        dataloader_persistent_workers=config.DATALOADER_PERSISTENT_WORKERS,
        dataloader_drop_last=config.DATALOADER_DROP_LAST,
        dataloader_prefetch_factor=getattr(config, 'DATALOADER_PREFETCH_FACTOR', 2),
        report_to=[],  # Disable wandb/tensorboard
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        remove_unused_columns=config.REMOVE_UNUSED_COLUMNS,
        logging_first_step=False,  # Disable first step logging to use custom format
        disable_tqdm=False,  # Show progress bar
        max_grad_norm=config.GRADIENT_CLIP_NORM,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        seed=seed,
        data_seed=seed,
        # Flash Attention optimizations handled in model initialization
    )

    # Set up dual learning rate optimizer AFTER model is on device
    optimizer = create_optimizer(model, config.WEIGHT_DECAY)
    
    # Verify optimizer parameters are on correct device
    for i, group in enumerate(optimizer.param_groups):
        if group['params']:
            param_device = next(iter(group['params'])).device
            if param_device != device:
                print(f"[WARNING] Optimizer group {i} parameters on {param_device}, expected {device}")
    
    # Configure learning rate scheduler
    scheduler = create_scheduler(optimizer, total_steps, warmup_steps, config.LR_SCHEDULE_TYPE)

    # Initialize classification trainer
    trainer = LMOLClassificationTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=ds,
        tokenizer=tokenizer,
        w_sim=config.WSIM,
        cons_weight=config.CONS_WEIGHT,
        swap_ce_weight=getattr(config, 'SWAP_CE_WEIGHT', 1.0),
        optimizers=(optimizer, scheduler),
        use_dynamic_consistency=getattr(config, 'USE_DYNAMIC_CONSISTENCY', False),
        cons_weight_start=getattr(config, 'CONS_WEIGHT_START', 5.0),
        cons_weight_end=getattr(config, 'CONS_WEIGHT_END', 20.0),
        cons_weight_ramp_ratio=getattr(config, 'CONS_WEIGHT_RAMP_RATIO', 0.5),
        total_steps=total_steps,
    )
    

    # DataLoader will use default RandomSampler
    print(f"[DATA] Using default RandomSampler for dataset with {len(ds)} samples")

    # Create CSV logger for training metrics
    # out_dir already contains the fold directory, so we pass it directly
    csv_logger = CSVTrainLogger(out_dir, "training_log.csv")
    print(f"[CSV_LOG] Training metrics will be logged to: {csv_logger.get_log_path()}")
    
    # Add callback for automatic model saving and CSV logging
    best_dir = out_dir / "best"
    trainer.add_callback(SaveBestTrainingLossCallback(best_dir, processor, tokenizer, csv_logger))

    # Perform sanity check on first batch
    first_batch_sanity_check(trainer)

    # Start training
    print("Training started...")
    print("-" * 72)
    
    t0 = time.time()
    try:
        trainer.train()
    finally:
        # Clean up gradient diagnostics
        if hasattr(trainer, 'cleanup'):
            trainer.cleanup()
    t1 = time.time()
    
    print("-" * 72)

    # Final model saving is now handled by the callback as last_{epoch}
    
    # Report best model saved by callback
    best_meta_path = best_dir / "best_meta.json"
    if best_meta_path.exists():
        try:
            best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
            print(f"[TRAINING COMPLETE] Best model: loss = {best_meta.get('best_loss', 'N/A'):.6f} | Step: {best_meta.get('global_step', 'N/A'):,} | Epoch: {best_meta.get('epoch', 'N/A'):.2f} | Saved to: {best_dir}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[INFO] Best model saved to {best_dir} (metadata parsing failed: {e})")
    else:
        print("[WARNING] No best model was saved during training")

    # Save training metadata
    (out_dir / "run_meta.json").write_text(json.dumps({
        "samples": len(ds),
        "effective_samples": effective_samples,
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "epochs": config.NUM_EPOCHS,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "time_min": round((t1 - t0)/60, 2),
        "swap_double": bool(config.SWAP_DOUBLE),
        "swap_ce_weight": float(getattr(config, 'SWAP_CE_WEIGHT', 1.0)),
        "cons_weight": float(config.CONS_WEIGHT),
        "wsim": float(config.WSIM),
        "seed": seed,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }, indent=2), encoding="utf-8")

    print(f"\n[INFO] Fold {fold_idx} complete: {(t1-t0)/60:.4f} min | Saved to: {out_dir.name}\n")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def training_main():
    """
    Main training function for LMOL using PyTorch's default single-process handling.
    
    Orchestrates the complete training pipeline:
    1. Parse CLI arguments and apply overrides
    2. Set up GPU environment and device management
    3. Create timestamped output directory
    4. Load and validate training CSV files
    5. Train each fold sequentially
    6. Clean up GPU memory between folds
    7. Report final training summary
    
    The training implements 5-fold cross-validation where each fold
    trains a separate model adapter on 4/5 of the data and evaluates
    on the remaining 1/5.
    
    Uses PyTorch's implicit single-process handling which can automatically
    utilize available GPUs if configured by the framework.
    """
    # Parse CLI arguments
    args = parse_training_args()
    
    # Apply CLI overrides to config
    apply_cli_overrides(config, args)
    
    # Print training banner
    print_training_banner(config, args)
    
    # Set up GPU environment and device management
    print(f"[INFO] Setting up GPU environment...")
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
        # Set cuDNN benchmark based on config
        torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK
        print(f"[INFO] cuDNN benchmark: {config.CUDNN_BENCHMARK}")
        
        # Set mixed precision dtype
        if config.AMP_DTYPE == "bf16":
            print(f"[INFO] Using bfloat16 mixed precision (A100 optimized)")
        elif config.AMP_DTYPE == "fp16":
            print(f"[INFO] Using float16 mixed precision")
        else:
            print(f"[INFO] Using full precision (no AMP)")
    else:
        print("[WARN] CUDA not available, falling back to CPU training")
    
    # Create timestamped output directory
    run_dir = resolve_path(getattr(config, "OUTPUT_DIR", "outputs")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _print_header(f"LMOL Training - Run: {run_dir.name}")

    # Load and validate training CSV files
    csvs = [ensure_exists(resolve_path(p)) for p in config.TRAIN_PAIRS_CSVS]
    print(f"Training {len(csvs)} fold(s)\n")
    
    # Train each fold
    for i, csv in enumerate(csvs, 1):
        fold_dir = run_dir / f"fold{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        train_one_fold(i, csv, fold_dir, max_steps=args.get('max_steps'))
        
        # Clean up GPU memory between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final summary
    _print_header("All Folds Complete")
    print(f"[INFO] Trained {len(csvs)} fold(s) successfully")
    print(f"[INFO] Output directory: {run_dir}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU count: {torch.cuda.device_count()}")
    print()


if __name__ == "__main__":
    training_main()
