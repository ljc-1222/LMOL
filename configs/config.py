# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    # ----------------------------
    # Model & Processor
    # ----------------------------
    MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
    IMAGE_SIZE: int = 336

    # ----------------------------
    # Data locations
    # ----------------------------
    PAIRS_OUT_DIR: str = "data/pairs"

    # Use immutable tuples here to avoid dataclass mutable-default error
    TRAIN_PAIRS_CSVS: Tuple[str, ...] = (
        "data/pairs/train_fold1_90000.csv",
        "data/pairs/train_fold2_90000.csv",
        "data/pairs/train_fold3_90000.csv",
        "data/pairs/train_fold4_90000.csv",
        "data/pairs/train_fold5_90000.csv",
    )
    EVAL_PAIRS_CSVS: Tuple[str, ...] = (
        "data/pairs/eval_fold1_3000.csv",
        "data/pairs/eval_fold2_3000.csv",
        "data/pairs/eval_fold3_3000.csv",
        "data/pairs/eval_fold4_3000.csv",
        "data/pairs/eval_fold5_3000.csv",
    )

    # ----------------------------
    # Instruction & Answers (LMOL)
    # ----------------------------
    QUESTION_TEXT: str = "which face looks more attractive?"
    ANSWER_FIRST: str = "First."
    ANSWER_SECOND: str = "Second."
    ANSWER_SIMILAR: str = "Similar."
    ANSWER_SET: List[str] = None  # will be filled below

    # ----------------------------
    # Order-learning rule (UOL/LMOL)
    # ----------------------------
    THETA: float = 0.2

    # ----------------------------
    # Training schedule
    # ----------------------------
    NUM_EPOCHS: int = 1
    PER_DEVICE_TRAIN_BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATION_STEPS: int = 8
    DATALOADER_NUM_WORKERS: int = 4
    MAX_SEQ_LEN: int = 2048

    # LoRA / QLoRA
    LORA_R: int = 8
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05

    # 4-bit quantization
    USE_4BIT: bool = True
    BNB_4BIT_COMPUTE_DTYPE: str = "bfloat16"
    BNB_4BIT_QUANT_TYPE: str = "nf4"
    BNB_DOUBLE_QUANT: bool = True

    # Optim & LR
    LR_PROJECTION: float = 2e-5
    LR_LORA: float = 2e-4
    WEIGHT_DECAY: float = 0.0
    WARMUP_STEPS: int = 0
    BETA: tuple = (0.9, 0.999)

    # Logging / Saving
    OUTPUT_DIR: str = "model/checkpoints/llava-1.5-7b-hf"
    LOGGING_STEPS: int = 10

    # Misc
    SEED: int = 42

    def __post_init__(self):
        # Fill answer set once at init
        if self.ANSWER_SET is None:
            self.ANSWER_SET = [self.ANSWER_FIRST, self.ANSWER_SECOND, self.ANSWER_SIMILAR]


config = Config()
