# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # ----------------------------
    # Model & Processor
    # ----------------------------
    MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"  # LLaVA-1.5-7B as in LMOL
    IMAGE_SIZE: int = 336  # CLIP ViT-L/14-336px

    # ----------------------------
    # Instruction & Answers (strictly match paper)
    # ----------------------------
    QUESTION_TEXT: str = "which face looks more attractive?"
    ANSWER_FIRST: str = "First."
    ANSWER_SECOND: str = "Second."
    ANSWER_SIMILAR: str = "Similar."
    ANSWER_SET: List[str] = None  # will be filled in __post_init__

    # ----------------------------
    # Pair construction rule
    # ----------------------------
    THETA: float = 0.2  # threshold for "Similar" as in LMOL/UOL

    # ----------------------------
    # Data paths (edit to your layout)
    # ----------------------------
    TRAIN_PAIRS_CSV: str = "./pairs/train_fold1_90000.csv"  # offline, balanced, per-fold 90k
    # You can prepare more folds as needed; here we only train (epoch=1) on one fold per paper.

    # ----------------------------
    # Training schedule (strict to LMOL)
    # ----------------------------
    NUM_EPOCHS: int = 1
    PER_DEVICE_TRAIN_BATCH_SIZE: int = 1  # large-token context; adjust with GPU memory
    GRADIENT_ACCUMULATION_STEPS: int = 8  # effective batch size
    DATALOADER_NUM_WORKERS: int = 4
    MAX_SEQ_LEN: int = 2048

    # LoRA / QLoRA (paper: r=8, lambda(scale)=4)
    LORA_R: int = 8
    LORA_ALPHA: int = 4
    LORA_DROPOUT: float = 0.05

    # 4-bit quantization
    USE_4BIT: bool = True
    BNB_4BIT_COMPUTE_DTYPE: str = "bfloat16"  # "bfloat16" or "float16"
    BNB_4BIT_QUANT_TYPE: str = "nf4"
    BNB_DOUBLE_QUANT: bool = True

    # Optim & LR (paper: proj 2e-5; LoRA 2e-4; cosine to 0)
    LR_PROJECTION: float = 2e-5
    LR_LORA: float = 2e-4
    WEIGHT_DECAY: float = 0.0
    WARMUP_STEPS: int = 0

    # Logging / Saving
    OUTPUT_DIR: str = "./outputs_lmol_llava7b"
    LOGGING_STEPS: int = 50
    SAVE_TOTAL_LIMIT: int = 1
    SAVE_STEPS: int = 0  # no periodic saving; we only run 1 epoch

    # Reproducibility
    SEED: int = 42

    def __post_init__(self):
        if self.ANSWER_SET is None:
            self.ANSWER_SET = [self.ANSWER_FIRST, self.ANSWER_SECOND, self.ANSWER_SIMILAR]


config = Config()
