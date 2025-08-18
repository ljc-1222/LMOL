# -*- coding: utf-8 -*-
# All comments must be in English.

import math
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, get_cosine_schedule_with_warmup

from configs.config import config
from utils.set_seed import set_seed
from data.dataset import SCUT_FBP5500_Pairs
from utils.data_collator import LlavaPairsCollator
from model.model import model_generator

def group_parameters_for_optimizer(model) -> List[Dict[str, Any]]:
    """
    Create two parameter groups:
      - Projection (mm_projector or multi_modal_projector): lr = 2e-5
      - LoRA-adapted weights (injected modules with 'lora_'): lr = 2e-4
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
            # keep frozen groups out of optimizer
            pass

    param_groups: List[Dict[str, Any]] = []
    if len(proj_params) > 0:
        param_groups.append({"params": proj_params, "lr": config.LR_PROJECTION})
    if len(lora_params) > 0:
        param_groups.append({"params": lora_params, "lr": config.LR_LORA})

    assert len(param_groups) > 0, "No trainable parameters found for optimizer."
    return param_groups

def main():
    set_seed(config.SEED)

    # Prepare data
    train_dataset = SCUT_FBP5500_Pairs(config.TRAIN_PAIRS_CSV)

    # Build model / processor / tokenizer
    model, processor, tokenizer = model_generator()

    # Collator
    collator = LlavaPairsCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=config.MAX_SEQ_LEN
    )

    # Training arguments (no evaluation; BT scoring intentionally omitted as requested)
    args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LR_LORA,  # not used directly because we pass custom optimizer
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=config.LOGGING_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        save_steps=config.SAVE_STEPS,
        evaluation_strategy="no",
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        report_to=[],
    )

    # Build optimizer with param groups
    param_groups = group_parameters_for_optimizer(model)
    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler: cosine to 0
    # Estimate total steps
    steps_per_epoch = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    total_steps = max(steps_per_epoch * args.num_train_epochs, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

if __name__ == "__main__":
    main()
