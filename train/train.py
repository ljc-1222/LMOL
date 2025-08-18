# -*- coding: utf-8 -*-

import os
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Fix tokenizer parallelism warning before importing torch/transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from transformers import (
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    TrainerCallback,
)

from configs.config import config
from utils.set_seed import set_seed
from data.dataset import SCUT_FBP5500_Pairs
from utils.data_collator import LlavaPairsCollator
from model.model import model_generator

# Project root when running "python3 -m train.train" from LMOL/
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # -> LMOL/


def resolve_path(p: str) -> Path:
    """Resolve a path relative to the project root (LMOL/), unless it's absolute."""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ensure_exists(path: Path) -> Path:
    """Ensure that a given file path exists; raise with a helpful message otherwise."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return path


def group_parameters_for_optimizer(model) -> List[Dict[str, Any]]:
    """
    Create two parameter groups:
      - Projection (mm_projector or multi_modal_projector): lr = config.LR_PROJECTION
      - LoRA-adapted weights (modules whose names contain 'lora_'): lr = config.LR_LORA
    All others remain frozen (requires_grad=False).
    """
    proj_params, lora_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(p)
        elif "mm_projector" in n or "multi_modal_projector" in n:
            proj_params.append(p)
        else:
            pass

    param_groups: List[Dict[str, Any]] = []
    if proj_params:
        param_groups.append({"params": proj_params, "lr": config.LR_PROJECTION})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": config.LR_LORA})
    assert param_groups, "No trainable parameters found for optimizer."
    return param_groups


# ------------------------------
# Save-best-by-training-loss callback
# ------------------------------
class SaveBestTrainingLossCallback(TrainerCallback):
    """
    Save to `save_dir` whenever the training loss improves.
    Also write a concise 'best_meta.json' alongside the weights.
    """

    def __init__(self, save_dir: str | Path, processor, tokenizer):
        self.best_loss: float = float("inf")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.latest_eval: Dict[str, float] = {}
        self._processor = processor
        self._tokenizer = tokenizer

    def _write_meta(self, loss: float, state) -> None:
        meta = {
            "best_loss": loss,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        (self.save_dir / "best_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def on_log(self, args, state, control, logs: Dict[str, float] | None = None, model=None, **kw):
        if logs is None:
            return

        # Keep the newest eval metrics if present (eval is disabled by default here).
        for k in ("eval_loss", "eval_accuracy"):
            if k in logs:
                self.latest_eval[k] = float(logs[k])

        if "loss" not in logs:
            return

        loss = float(logs["loss"])
        if loss < self.best_loss:
            self.best_loss = loss

            # Save model & processors to save_dir
            if model is not None:
                # Note: we do not rely on trainer's built-in save_strategy; we save here.
                model.save_pretrained(self.save_dir, safe_serialization=False)
                try:
                    self._processor.save_pretrained(self.save_dir)
                except Exception:
                    pass
                try:
                    self._tokenizer.save_pretrained(self.save_dir)
                except Exception:
                    pass

                # rank 0 print
                if getattr(args, "local_rank", -1) in (-1, 0):
                    print(f"[SaveBest] new best training loss {loss:.6f} → saved to {self.save_dir}")

            self._write_meta(loss, state)


def train_one_fold(fold_idx: int, train_csv_path: Path, fold_dir: Path) -> None:
    """Train the model for a single fold on the given CSV dataset path."""
    # Reproducibility
    os.environ["TORCH_CHECKPOINT_USE_REENTRANT"] = "1"
    set_seed(config.SEED)

    # Dataset
    dataset = SCUT_FBP5500_Pairs(str(train_csv_path))
    if len(dataset) == 0:
        raise RuntimeError(f"Empty training dataset loaded from: {train_csv_path}")

    # Model / Processor / Tokenizer
    model, processor, tokenizer = model_generator()

    # Collator
    collator = LlavaPairsCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=config.MAX_SEQ_LEN
    )

    # Steps estimate for logs/scheduler
    eff_batch = config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = max(1, math.ceil(len(dataset) / eff_batch))
    total_steps = steps_per_epoch * int(config.NUM_EPOCHS)

    print("=" * 80)
    print(f"[Fold {fold_idx}]")
    print(f" - CSV: {train_csv_path}")
    print(f" - Samples: {len(dataset)} | Effective batch: {eff_batch}")
    print(f" - Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f" - LR (proj): {config.LR_PROJECTION} | LR (lora): {config.LR_LORA}")
    print(f" - Saving best to: {fold_dir}")
    print("=" * 80)

    # Training arguments (note: use eval_strategy per your request)
    args = TrainingArguments(
        output_dir=str(fold_dir),           
        overwrite_output_dir=True,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LR_LORA,       
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=config.LOGGING_STEPS,
        logging_strategy="steps",
        eval_strategy="no",                   
        save_strategy="no",                  
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        report_to=[],                       
        disable_tqdm=False,
    )

    # Optimizer with parameter groups
    param_groups = group_parameters_for_optimizer(model)
    optimizer = AdamW(param_groups, betas=config.BETA, weight_decay=config.WEIGHT_DECAY)

    # Scheduler: cosine to 0
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # Trainer with our callback
    save_cb = SaveBestTrainingLossCallback(save_dir=fold_dir, processor=processor, tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
        callbacks=[save_cb],
    )

    print(f"[Fold {fold_idx}] Start training with dataset: {train_csv_path}")
    trainer.train()
    print(f"[Fold {fold_idx}] Finished. Best checkpoint (by training loss) stored in: {fold_dir}")


def main():
    # -----------------------------
    # Build the base RUN_DIR once
    # -----------------------------
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = getattr(config, "MODEL_SAVE_DIR", None)
    if base_dir is None:
        base_dir = config.OUTPUT_DIR  # fall back to OUTPUT_DIR if MODEL_SAVE_DIR is missing
    RUN_DIR = resolve_path(base_dir) / run_stamp
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[RunDir] checkpoints will be saved under: {RUN_DIR.resolve()}")

    # Resolve five training CSV paths and ensure they exist
    train_csvs = [ensure_exists(resolve_path(p)) for p in config.TRAIN_PAIRS_CSVS]

    # 5-fold training loop — each fold saves to RUN_DIR/fold{n}
    for fold_idx, train_csv in enumerate(train_csvs, start=1):
        fold_dir = RUN_DIR / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_one_fold(fold_idx, train_csv, fold_dir)

        # Optional: free GPU cache between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"All folds finished. Best checkpoints are under: {RUN_DIR}")


if __name__ == "__main__":
    main()
