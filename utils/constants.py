# -*- coding: utf-8 -*-
"""
LMOL Constants Module

This module defines all constants used throughout the LMOL project.
Constants are organized by category for easy maintenance and reference.

Categories:
- Vision model constants (patch sizes, dimensions)
- Token constants (special tokens, padding values)
- Display constants (formatting, progress updates)
- Evaluation constants (sample sizes, scoring ranges)
- Training constants (gradient clipping, numerical stability)
- File constants (encoding, formatting)
- Model architecture constants (LoRA targets, bias types)
- Image processing constants (channels, data types)
"""

# ============================================================================
# VISION MODEL CONSTANTS
# ============================================================================

DEFAULT_PATCH_SIZE = 14
"""Default patch size for vision transformer (CLIP ViT-L/14)"""

DEFAULT_IMAGE_SIZE = 336
"""Default input image size (336x336 pixels for CLIP ViT-L/14)"""

VISION_DIM = 1024
"""Vision feature dimension from CLIP ViT-L/14 encoder"""

TEXT_HIDDEN_DIM = 4096
"""Text hidden dimension for LLaVA-1.5-7B language model"""

# ============================================================================
# TOKEN CONSTANTS
# ============================================================================

IGNORE_INDEX = -100
"""PyTorch ignore index for loss computation (ignored tokens)"""

PADDING_VALUE = 0
"""Padding value for attention masks"""

# ============================================================================
# DISPLAY CONSTANTS
# ============================================================================

HEADER_SEPARATOR_LENGTH = 72
"""Length of separator lines for formatted headers"""

PROGRESS_UPDATE_INTERVAL = 100
"""Update progress every N samples during processing"""

# ============================================================================
# EVALUATION CONSTANTS
# ============================================================================

DEFAULT_EVAL_SAMPLES = 1000
"""Default number of samples for evaluation"""

BRADLEY_TERRY_SEARCH_POINTS = 200
"""Number of search points for Bradley-Terry model fitting"""

BRADLEY_TERRY_MIN_SCORE = 1.0
"""Minimum score for Bradley-Terry model"""

BRADLEY_TERRY_MAX_SCORE = 5.0
"""Maximum score for Bradley-Terry model"""

# ============================================================================
# TRAINING CONSTANTS
# ============================================================================

DEFAULT_GRADIENT_CLIP_NORM = 1.0
"""Default gradient clipping threshold for training stability"""

DEFAULT_EPSILON = 1e-8
"""Default epsilon value for numerical stability"""

# ============================================================================
# FILE CONSTANTS
# ============================================================================

DEFAULT_CSV_ENCODING = "utf-8"
"""Default encoding for CSV files"""

DEFAULT_CSV_NEWLINE = ""
"""Default newline character for CSV files"""

# ============================================================================
# MODEL ARCHITECTURE CONSTANTS
# ============================================================================

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
"""Target modules for LoRA adaptation (attention projections)"""

DEFAULT_BIAS_TYPE = "none"
"""Default bias type for LoRA configuration"""

DEFAULT_TASK_TYPE = "CAUSAL_LM"
"""Default task type for LoRA configuration"""

# ============================================================================
# IMAGE PROCESSING CONSTANTS
# ============================================================================

DEFAULT_IMAGE_CHANNELS = 3
"""Default number of image channels (RGB)"""

DEFAULT_IMAGE_DTYPE = "float32"
"""Default data type for image tensors"""
