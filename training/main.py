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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import TrainingArguments

# Fix HuggingFace tokenizers parallelism warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.config import config
from utils.constants import HEADER_SEPARATOR_LENGTH
from utils import set_seed
from data import SCUT_FBP5500_Pairs, LlavaPairsCollator
from model import model_generator
# Lazy imports to avoid dependency issues
# from .trainer import WeightedSwapConsistencyTrainer
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


def resolve_path(p: str | Path) -> Path:
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


def train_one_fold(fold_idx: int, train_csv: Path, out_dir: Path):
    """
    Train LMOL model for a single fold.
    
    This function implements the complete training pipeline for one fold:
    1. Set up reproducible random seeds
    2. Load dataset and create data collator
    3. Initialize model with LoRA and custom projector
    4. Set up dual learning rate optimizer and scheduler
    5. Configure training arguments with performance optimizations
    6. Train with weighted cross-entropy + consistency loss
    7. Save final model and training metadata
    
    Args:
        fold_idx: Fold index (1-based)
        train_csv: Path to training CSV file
        out_dir: Output directory for this fold
    """
    # Suppress warnings during training
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*use_cache=True.*is incompatible with gradient checkpointing.*")
    warnings.filterwarnings("ignore", message=".*loss_type=None.*was set in the config but it is unrecognised.*")
    
    # Lazy imports to avoid dependency issues
    from .trainer import WeightedSwapConsistencyTrainer
    from .callbacks import SaveBestTrainingLossCallback
    from .optimizer import group_parameters_for_optimizer, create_optimizer, create_scheduler
    
    _print_header(f"Fold {fold_idx} Training")
    
    # Set reproducible seed for this fold
    seed = 42 + fold_idx
    set_seed(seed)
    
    # Initialize dataset, model, and processor
    ds = SCUT_FBP5500_Pairs(str(train_csv))
    model, processor, tokenizer, fast_processor = model_generator()
    model.train()

    # Calculate training parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Account for swap doubling: 45,000 pairs → 90,000 samples
    effective_samples = len(ds) * 2 if config.SWAP_DOUBLE else len(ds)
    effective_batch_size = config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS * 2
    steps_per_epoch = math.ceil(effective_samples / effective_batch_size)
    total_steps = steps_per_epoch * config.NUM_EPOCHS
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
    
    # Count trainable layers
    trainable_layers = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Trainable layers: {trainable_layers}\n")
    
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
    print(f"  Speedup:   Workers = {config.DATALOADER_NUM_WORKERS}, PinMem = {config.DATALOADER_PIN_MEMORY}, " +
          f"FastProc = {fast_processor}, Compile = {getattr(config, 'USE_TORCH_COMPILE', False)}, " +
          f"FlashAttn = {getattr(config, 'USE_FLASH_ATTENTION', False)}")
    print(f"  Seed:      {seed}")
    print()

    # Create data collator with swap doubling
    collator = LlavaPairsCollator(processor=processor, tokenizer=tokenizer,
                                  max_length=config.MAX_SEQ_LEN, is_training=True)

    # Configure training arguments with performance optimizations
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
        fp16=config.FP16, bf16=config.BF16,  # Use config-based precision
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
        dataloader_persistent_workers=config.DATALOADER_PERSISTENT_WORKERS,
        dataloader_drop_last=config.DATALOADER_DROP_LAST,
        dataloader_prefetch_factor=getattr(config, 'DATALOADER_PREFETCH_FACTOR', 2),
        report_to=[],  # Disable wandb/tensorboard
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        remove_unused_columns=config.REMOVE_UNUSED_COLUMNS,
        logging_first_step=False,  # Disable first step logging to use custom format
        disable_tqdm=False,  # Keep progress bars
        max_grad_norm=config.GRADIENT_CLIP_NORM,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        seed=seed,
        data_seed=seed,
        # Flash Attention optimizations handled in model initialization
    )

    # Set up dual learning rate optimizer
    optimizer = create_optimizer(model, config.WEIGHT_DECAY)
    
    # Configure learning rate scheduler
    scheduler = create_scheduler(optimizer, total_steps, warmup_steps, config.LR_SCHEDULE_TYPE)

    # Initialize custom trainer with swap consistency
    trainer = WeightedSwapConsistencyTrainer(
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

    # Add callback for automatic model saving
    best_dir = out_dir / "best"
    last_dir = out_dir / "last"
    trainer.add_callback(SaveBestTrainingLossCallback(best_dir, processor, tokenizer))

    # Perform sanity check on first batch
    first_batch_sanity_check(trainer)

    # Start training
    print("Training started...")
    print("-" * 72)
    t0 = time.time()
    trainer.train()
    t1 = time.time()
    print("-" * 72)

    # Save final model and metadata
    last_dir.mkdir(parents=True, exist_ok=True)
    try:
        trainer.save_model(str(last_dir))
    except Exception:
        torch.save(model.state_dict(), last_dir / "pytorch_model.bin")
    
    try:
        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)
        print(f"[INFO] Final model saved to {last_dir}")
    except Exception as e:
        print(f"[WARN] Failed to save final model: {e}")
    
    # Report best model saved by callback
    best_meta_path = best_dir / "best_meta.json"
    if best_meta_path.exists():
        try:
            best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
            print(f"[TRAINING COMPLETE] Best model: loss = {best_meta.get('best_loss', 'N/A'):.6f} | Step: {best_meta.get('global_step', 'N/A'):,} | Epoch: {best_meta.get('epoch', 'N/A'):.2f} | Saved to: {best_dir}")
        except Exception as e:
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
    }, indent=2), encoding="utf-8")

    print(f"\n[INFO] Fold {fold_idx} complete: {(t1-t0)/60:.4f} min | Saved to: {out_dir.name}\n")


def training_main():
    """
    Main training function for LMOL.
    
    Orchestrates the complete training pipeline:
    1. Create timestamped output directory
    2. Load and validate training CSV files
    3. Train each fold sequentially
    4. Clean up GPU memory between folds
    5. Report final training summary
    
    The training implements 5-fold cross-validation where each fold
    trains a separate model adapter on 4/5 of the data and evaluates
    on the remaining 1/5.
    """
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
        train_one_fold(i, csv, fold_dir)
        
        # Clean up GPU memory between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Additional memory cleanup using memory manager
        from utils.memory_manager import cleanup_model_memory
        # Note: trainer variable is not accessible here, so we rely on the trainer's own cleanup
    
    # Final summary
    _print_header("All Folds Complete")
    print(f"[INFO] Trained {len(csvs)} fold(s) successfully")
    print(f"[INFO] Output directory: {run_dir}\n")


if __name__ == "__main__":
    training_main()
