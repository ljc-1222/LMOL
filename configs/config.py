# -*- coding: utf-8 -*-
"""
LMOL Configuration Module

This module defines the complete configuration for the LMOL (Large Multimodal model for 
facial attractiveness comparison) training and evaluation pipeline. The configuration 
includes data paths, model parameters, training hyperparameters, and optimization settings.

Key Features:
- Comprehensive parameter coverage for all training aspects
- Dynamic consistency weighting for improved training stability
- Dual learning rate strategy for LoRA and projector parameters
- Advanced optimization settings for memory and speed
- Flexible path resolution for different deployment environments
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """
    LMOL Configuration Class
    
    Central configuration dataclass containing all parameters for the LMOL training
    and evaluation pipeline. Organized into logical sections for easy maintenance.
    """
    
    # ============================================================================
    # DATA CONFIGURATION
    # ============================================================================
    
    IMAGE_DIR: str = "data/raw/SCUT-FBP5500/Images"
    """Primary image directory root (can be empty string if not used)"""
    
    LABELS_PATH: str = "data/labels.txt"
    """Path to the SCUT-FBP5500 labels file containing image scores"""
    
    TRAIN_PAIRS_CSVS: List[str] = field(default_factory=lambda: [
        "data/pairs/train_fold1_45000.csv",
        "data/pairs/train_fold2_45000.csv", 
        "data/pairs/train_fold3_45000.csv",
        "data/pairs/train_fold4_45000.csv",
        "data/pairs/train_fold5_45000.csv",
    ])
    """List of training CSV files for 5-fold cross-validation"""
    
    EVAL_PAIRS_CSVS: List[str] = field(default_factory=lambda: [
        "data/pairs/eval_fold1_604450.csv",
        "data/pairs/eval_fold2_604450.csv",
        "data/pairs/eval_fold3_604450.csv", 
        "data/pairs/eval_fold4_604450.csv",
        "data/pairs/eval_fold5_604450.csv",
    ])
    """List of evaluation CSV files for 5-fold cross-validation"""
    
    TRAIN_PER_CLASS: int = 15000
    """Number of training samples per class (First/Second/Similar)"""
    
    PAIRS_OUT_DIR: str = "data/pairs"
    """Output directory for generated pair CSV files"""
    
    KFOLDS: int = 5
    """Number of folds for cross-validation"""
    
    THETA: float = 0.2
    """Threshold for determining 'Similar' class (score difference <= THETA)"""
    
    NEAR_BAND: float = 0.15
    """Tighter threshold for near-boundary sampling in Similar class"""
    
    NEAR_RATIO: float = 0.5
    """Probability of using near-boundary sampling for Similar class"""
    
    SEED: int = 42
    """Random seed for reproducibility across all operations"""
    
    # Path resolution configuration
    DATA_ROOTS: List[str] = field(default_factory=lambda: [
        ".",                 # Project root directory
        "data",              # Project data directory  
        "data/images",       # Legacy images directory
        "data/raw/SCUT-FBP5500/Images",  # Actual images directory
    ])
    """List of root directories to search when resolving relative image paths"""
    
    STRIP_PREFIXES: List[str] = field(default_factory=lambda: [
        "data/images/",
        "data/raw/SCUT-FBP5500/Images/",
        "images/", 
        "./",
    ])
    """Prefixes to strip from CSV paths before joining with DATA_ROOTS"""
    
    # ============================================================================
    # MODEL ARCHITECTURE CONFIGURATION
    # ============================================================================
    
    MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
    """HuggingFace model identifier for LLaVA-1.5-7B base model"""
    
    IMAGE_SIZE: int = 336
    """Input image resolution (336x336 pixels for CLIP ViT-L/14)"""
    
    MAX_SEQ_LEN: int = 1250
    """Maximum sequence length for text tokens"""
    
    # LoRA configuration
    USE_4BIT: bool = True
    """Enable 4-bit quantization to reduce memory usage"""
    
    LORA_R: int = 8
    """LoRA rank parameter (controls adaptation capacity)"""
    
    LORA_ALPHA: int = 32
    """LoRA alpha parameter (scaling factor for adaptation)"""
    
    LORA_DROPOUT: float = 0.05
    """LoRA dropout rate for regularization"""
    
    # ============================================================================
    # LOSS FUNCTION CONFIGURATION
    # ============================================================================
    
    WSIM: float = 1.05
    """Class weight for 'Similar' class in cross-entropy loss"""
    
    CONS_WEIGHT: float = 20
    """Base consistency weight for swap consistency regularization"""
    
    SWAP_CE_WEIGHT: float = 1.0
    """Weight multiplier for cross-entropy loss on swapped samples"""
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    
    # Batch processing (2 pairs → 4 samples → 16 steps → 64 samples)
    PER_DEVICE_TRAIN_BATCH_SIZE: int = 32
    """Process 2 pairs at a time (generates 4 samples with swap doubling)"""
    
    GRADIENT_ACCUMULATION_STEPS: int = 1
    """Accumulate gradients over 16 steps before parameter update"""
    
    DATALOADER_NUM_WORKERS: int = 16
    """Number of DataLoader worker processes for parallel data loading"""
    
    DATALOADER_PIN_MEMORY: bool = True
    """Pin memory for faster GPU transfer"""
    
    DATALOADER_PERSISTENT_WORKERS: bool = True
    """Keep DataLoader workers alive between epochs"""
    
    DATALOADER_DROP_LAST: bool = True
    """Drop incomplete batches for consistent batch sizes"""
    
    DATALOADER_PREFETCH_FACTOR: int = 2
    """Number of batches to prefetch per worker"""
    
    # Learning rate configuration
    LR_LORA: float = 5e-5
    """Learning rate for LoRA parameters (reduced for stability)"""
    
    LR_PROJECTION: float = 5e-6
    """Learning rate for projector parameters (reduced for stability)"""
    
    WEIGHT_DECAY: float = 0.1
    """Weight decay regularization coefficient"""
    
    LOGGING_STEPS: int = 32
    """Log training metrics every N steps"""
    
    NUM_EPOCHS: int = 3
    """Number of training epochs"""
    
    # ============================================================================
    # TRAINING METHOD CONFIGURATION
    # ============================================================================
    
    # Note: Only classification-based training is supported (generation-based had data leakage)
    
    # ============================================================================
    # TRAINING HEALTH MONITORING
    # ============================================================================
    
    # Early stopping configuration
    USE_EARLY_STOPPING: bool = True
    """Enable early stopping based on validation loss"""
    
    EARLY_STOPPING_PATIENCE: int = 5
    """Number of epochs to wait before stopping if no improvement"""
    
    EARLY_STOPPING_MIN_DELTA: float = 1e-4
    """Minimum change to qualify as an improvement"""
    
    # Gradient monitoring
    MONITOR_GRADIENT_NORMS: bool = True
    """Enable gradient norm monitoring and warnings"""
    
    GRADIENT_NORM_THRESHOLD: float = 30
    """Threshold for gradient norm warnings (reduced for earlier detection)"""
    
    GRADIENT_NORM_MIN_THRESHOLD: float = 1e-12
    """Minimum threshold for gradient norm warnings (very small to allow tiny gradients)"""
    
    # Per-class accuracy monitoring
    LOG_PER_CLASS_ACCURACY: bool = True
    """Enable per-class accuracy logging during training"""
    
    # Loss monitoring
    MONITOR_LOSS_ANOMALIES: bool = True
    """Enable loss anomaly detection (NaN, Inf, too small)"""
    
    MIN_REASONABLE_LOSS: float = 1e-3
    """Minimum reasonable loss value (below this triggers warning)"""
    
    # ============================================================================
    # ADVANCED TRAINING FEATURES
    # ============================================================================
    
    # Logit masking for 3-class classification
    USE_LOGIT_MASKING: bool = True
    """Enable logit masking to only consider 3 answer tokens during training"""
    
    # Learning rate scheduling
    USE_LR_SCHEDULING: bool = True
    """Enable learning rate scheduling for better convergence"""
    
    LR_SCHEDULE_TYPE: str = "cosine"
    """Type of learning rate schedule: 'cosine', 'linear', or 'constant'"""
    
    LR_WARMUP_RATIO: float = 0.05
    """Fraction of total steps for learning rate warmup (reduced for gentler ramp-up)"""
    
    LR_MIN_RATIO: float = 0.01
    """Minimum learning rate ratio (final LR = initial LR * min_ratio)"""
    
    # Cosine annealing specific
    LR_COSINE_RESTART: bool = False
    """Enable cosine annealing with restarts"""
    
    LR_COSINE_RESTART_PERIOD: int = 1000
    """Period for cosine annealing restarts (in steps)"""
    
    # Dynamic consistency weighting
    USE_DYNAMIC_CONSISTENCY: bool = True
    """Enable dynamic consistency weight for improved training stability"""
    
    CONS_WEIGHT_START: float = 5
    """Starting consistency weight (reduced for more conservative training)"""
    
    CONS_WEIGHT_END: float = 2
    """Final consistency weight (reduced for more stable training)"""
    
    CONS_WEIGHT_RAMP_RATIO: float = 0.5
    """Fraction of training steps to reach final consistency weight"""
    
    # Training strategy flags
    SWAP_DOUBLE: bool = True
    """Enable swap doubling: each pair generates (A,B) and (B,A) samples"""
    
    EFFECTIVE_BATCH_DOUBLE: bool = True
    """Account for doubled batch size in scheduler when SWAP_DOUBLE=True"""
    
    FORWARD_STRATEGY: str = "double"
    """Forward pass strategy: 'double' for swap doubling, 'single' otherwise"""
    
    # ============================================================================
    # MEMORY MANAGEMENT CONFIGURATION
    # ============================================================================
    
    # GPU memory management
    ENABLE_MEMORY_CLEANUP: bool = True
    """Enable intelligent memory cleanup during training"""
    
    MEMORY_CLEANUP_FREQUENCY: int = 10
    """How often to perform memory cleanup (every N batches)"""
    
    # ============================================================================
    # PERFORMANCE OPTIMIZATION CONFIGURATION
    # ============================================================================
    
    USE_TORCH_COMPILE: bool = True
    """Enable torch.compile for faster model execution (default: False for stability)"""
    
    TORCH_COMPILE_MODE: str = "max-autotune"
    """torch.compile mode: 'default', 'reduce-overhead', 'max-autotune'"""
    
    USE_FLASH_ATTENTION: bool = True
    """Enable Flash Attention 2 for faster attention computation"""
    
    FLASH_ATTENTION_BACKEND: str = "sdpa"
    """Flash Attention backend: 'flash_attn', 'sdpa', or 'eager'"""
    
    CUDNN_BENCHMARK: bool = True
    """Enable cuDNN benchmark for consistent input sizes (default: True for A100)"""
    
    GPU_LOG_INTERVAL: int = 100
    """Log GPU memory and utilization every N steps (0 to disable)"""
    
    GRADIENT_CLIP_NORM: float = 0.5
    """Gradient clipping threshold for training stability (1.0 is a good default for most cases)"""
    
    REMOVE_UNUSED_COLUMNS: bool = False
    """Keep all columns to avoid processing overhead"""
    
    # Mixed precision training
    AMP_DTYPE: str = "bf16"
    """Mixed precision dtype: 'bf16', 'fp16', or 'none' (default: bf16 for A100)"""
    
    FP16: bool = False
    """Enable 16-bit floating point precision for faster training (deprecated, use AMP_DTYPE)"""
    
    BF16: bool = True
    """Enable bfloat16 precision (better for A100) (deprecated, use AMP_DTYPE)"""
    
    # Memory optimization
    GRADIENT_CHECKPOINTING: bool = True
    """Enable gradient checkpointing to reduce memory usage"""
    
    # Memory management configuration
    ENABLE_MEMORY_CLEANUP: bool = True
    """Enable intelligent memory cleanup during training"""
    
    MEMORY_CLEANUP_FREQUENCY: int = 100
    """How often to perform memory cleanup (every N batches) - higher = less overhead"""
    
    MEMORY_MONITORING: bool = False
    """Enable memory usage monitoring and logging"""
    
    # ============================================================================
    # OUTPUT CONFIGURATION
    # ============================================================================
    
    OUTPUT_DIR: str = "model/checkpoints/llava-1.5-7b-hf_3class"
    """Base directory for saving model checkpoints and logs"""
    
    # ============================================================================
    # PROMPT CONFIGURATION
    # ============================================================================
    
    QUESTION_TEXT: str = (
        "Compare the facial attractiveness of these two images. "
        "Which face looks more attractive? "
        "Answer with exactly one word: "
        "First (left image), Second (right image), or Similar (equally attractive)"
    )
    """Question prompt for facial attractiveness comparison task"""
    
    ANSWER_FIRST: str = "First"
    """Answer string when left image is more attractive"""
    
    ANSWER_SECOND: str = "Second"
    """Answer string when right image is more attractive"""
    
    ANSWER_SIMILAR: str = "Similar"
    """Answer string when both images are similarly attractive"""
    
    # ============================================================================
    # GRADIENT DIAGNOSTICS CONFIGURATION
    # ============================================================================
    
    GRAD_LOG_INTERVAL: int = 0
    """How often to log gradient statistics (every N steps, 0 to disable)"""
    
    DETECT_ANOMALY: bool = False
    """Enable PyTorch autograd anomaly detection for debugging"""
    
    GRADIENT_CLIP_NORM: float = 1.0
    """Global gradient clipping threshold (0 = disabled)"""
    
    GRAD_CLIP: float = 0.0
    """Gradient clipping threshold (0 = disabled)"""

    def __post_init__(self):
        """
        Post-initialization processing.
        
        Performs derived computations and path normalization after dataclass
        initialization.
        """
        # Create answer set for validation
        self.ANSWER_SET = {self.ANSWER_FIRST, self.ANSWER_SECOND, self.ANSWER_SIMILAR}
        
        # Normalize data roots to absolute paths for consistent path resolution
        self.DATA_ROOTS = [str(Path(r).resolve()) for r in self.DATA_ROOTS]


# Global configuration instance
config = Config()
