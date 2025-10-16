# -*- coding: utf-8 -*-
"""
LMOL Model Factory Functions

This module provides factory functions for creating LMOL models for different
use cases: training and inference. It handles model loading, configuration,
and optimization setup.

Key Functions:
- model_generator: Build complete training model with LoRA and custom projector
- build_inference_base: Build base model for inference without LoRA adaptations
"""

from __future__ import annotations

from typing import Tuple
import os
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model

# Suppress verbose logging from transformers and other libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

# Fix HuggingFace tokenizers parallelism warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Allow BitsAndBytes to use CPU offloading without strict validation
# This is needed when GPU memory is limited and some modules need to be on CPU/disk
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

# Suppress HuggingFace verbose output during model loading
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from configs.config import config
from .architecture import _find_projector_handle, _set_all_requires_grad, _report_model_sizes, _breakdown_projector
from .projector import LMOLProjector


def model_generator():
    """
    Build LMOL model for training with LoRA adaptations and custom projector.
    
    This function creates the complete LMOL training model:
    1. Load base LLaVA-1.5-7B model with optional 4-bit quantization
    2. Replace original projector with custom LMOLProjector
    3. Apply LoRA adaptations to attention projections
    4. Configure training parameters (only LoRA + projector trainable)
    5. Enable gradient checkpointing and input gradients
    6. Apply performance optimizations (torch.compile)
    
    Architecture:
    - Base: LLaVA-1.5-7B (7B parameters, mostly frozen)
    - Projector: LMOLProjector (1024 → 4096 → 4096, trainable)
    - LoRA: Applied to q_proj, k_proj, v_proj, o_proj (trainable)
    - Trainable: ~50M parameters (~0.7% of total)
    
    Returns:
        Tuple of (model, processor, tokenizer, fast_processor) ready for training
            - fast_processor: bool indicating if fast processor was successfully loaded
    """
    # ============================================================================
    # TOKENIZER AND PROCESSOR SETUP
    # ============================================================================
    
    # Load tokenizer with fast processing
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load processor with fast image processing
    fast_processor = False
    try:
        processor = AutoProcessor.from_pretrained(config.MODEL_ID, use_fast=True)
        fast_processor = True
    except Exception:
        # Fallback to slow processor if fast not available
        processor = AutoProcessor.from_pretrained(config.MODEL_ID)
        print("[WARN] Using slow processor (functionality unaffected)")

    # ============================================================================
    # QUANTIZATION CONFIGURATION
    # ============================================================================
    
    # Configure 4-bit quantization for memory efficiency
    quant_cfg = None
    if getattr(config, "USE_4BIT", True):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # ============================================================================
    # MODEL LOADING WITH OPTIMIZATIONS
    # ============================================================================
    
    # Prepare model loading arguments with explicit GPU device mapping
    device_map = "auto"
    if torch.cuda.is_available():
        # Use "auto" for proper device mapping with quantization
        device_map = "auto"
        print(f"[INFO] Using device_map: {device_map} (CUDA available)")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "quantization_config": quant_cfg,
    }
    
    # Add flash attention configuration if enabled
    if getattr(config, "USE_FLASH_ATTENTION", False):
        model_kwargs["attn_implementation"] = getattr(config, "FLASH_ATTENTION_BACKEND", "flash_attn")
    
    # Load base LLaVA model
    print(f"[INFO] Loading model with device_map: {device_map}")
    model = LlavaForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        **model_kwargs
    )
    
    # Verify model device placement
    model_device = next(model.parameters()).device
    print(f"[INFO] Model loaded on device: {model_device}")
    if torch.cuda.is_available() and not model_device.type == 'cuda':
        print(f"[WARN] Model not on GPU! This may cause performance issues.")
    elif torch.cuda.is_available():
        print(f"[INFO] Model successfully loaded on GPU: {model_device}")

    # ============================================================================
    # MODEL CONFIGURATION FOR TRAINING
    # ============================================================================
    
    # Disable caching for gradient checkpointing compatibility
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # Enable input gradients for proper gradient flow
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ============================================================================
    # PROJECTOR REPLACEMENT
    # ============================================================================
    
    # Find and replace the original projector with LMOL projector
    proj_path, old_proj = _find_projector_handle(model)
    assert old_proj is not None, "Projector not found on the loaded LLaVA model."

    # Create LMOL projector with proper dimensions
    # CLIP ViT-L/14-336px → 1024-dim vision patches, 4096-dim LLM hidden
    device = next(model.parameters()).device
    lmol_proj = LMOLProjector(vision_dim=1024, text_hidden_dim=4096).to(
        device=device, dtype=torch.bfloat16
    )

    # Replace projector using the same attribute path
    holder = model
    parts = proj_path.split(".")
    for p in parts[:-1]:
        holder = getattr(holder, p)
    setattr(holder, parts[-1], lmol_proj)

    # ============================================================================
    # LORA INTEGRATION
    # ============================================================================
    
    # Configure LoRA for efficient parameter adaptation
    lora_cfg = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,  # Projector is part of base model; no 2nd copy
    )
    
    # Suppress PEFT model summary output
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        model = get_peft_model(model, lora_cfg)
    finally:
        sys.stdout = old_stdout

    # ============================================================================
    # PARAMETER CONFIGURATION
    # ============================================================================
    
    # Freeze all parameters first
    _set_all_requires_grad(model, False)
    
    # Enable gradients only for LoRA parameters
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    
    # Enable gradients for projector parameters
    for p in lmol_proj.parameters():
        p.requires_grad = True

    # ============================================================================
    # FINAL SETUP AND REPORTING
    # ============================================================================
    
    # Report model statistics
    _report_model_sizes(model)
    _ = _breakdown_projector(lmol_proj)

    # Set model to training mode
    model.train()
    
    return model, processor, tokenizer, fast_processor


def build_inference_base():
    """
    Build base LLaVA model for inference without LoRA adaptations.
    
    This function creates a base model configured identically to the training
    model but without LoRA adaptations. It's used for loading saved LoRA
    adapters during inference.
    
    Key Features:
    - Same quantization and optimization settings as training
    - Custom LMOLProjector for consistent architecture
    - Gradient checkpointing enabled for memory efficiency
    - Model set to evaluation mode
    
    This base model is designed to work with PeftModel.from_pretrained()
    to load saved LoRA adapters for inference.
    
    Returns:
        Base LLaVA model ready for LoRA adapter loading
    """
    # ============================================================================
    # QUANTIZATION CONFIGURATION
    # ============================================================================
    
    # Configure 4-bit quantization (same as training)
    quant_cfg = None
    if getattr(config, "USE_4BIT", True):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload for better compatibility
        )
    
    # Prepare model loading arguments with explicit GPU device mapping
    device_map = "auto"
    if torch.cuda.is_available():
        # Use "auto" for proper device mapping with quantization
        device_map = "auto"
        print(f"[INFO] Using device_map: {device_map} (CUDA available)")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "quantization_config": quant_cfg,
        "low_cpu_mem_usage": True,  # Reduce CPU memory usage during loading
    }
    
    # Add flash attention configuration if enabled
    if getattr(config, "USE_FLASH_ATTENTION", False):
        model_kwargs["attn_implementation"] = getattr(config, "FLASH_ATTENTION_BACKEND", "flash_attn")
    
    # Load base model with same configuration as training
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            config.MODEL_ID,
            **model_kwargs
        )
    except ValueError as e:
        # If quantization validation fails, try fallback strategies
        if "validate_environment" in str(e) or "dispatched" in str(e):
            print(f"[WARN] Quantization validation failed: {str(e)[:200]}")
            
            # Strategy 1: Try with explicit memory constraints
            if torch.cuda.is_available() and "max_memory" not in model_kwargs:
                print(f"[INFO] Retrying with explicit GPU memory allocation...")
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                max_memory = {0: f"{int(gpu_memory * 0.80 / 1024**3)}GiB", "cpu": "32GiB"}
                model_kwargs["max_memory"] = max_memory
                print(f"[INFO] max_memory: {max_memory}")
                try:
                    model = LlavaForConditionalGeneration.from_pretrained(
                        config.MODEL_ID,
                        **model_kwargs
                    )
                    print(f"[INFO] Successfully loaded with explicit memory allocation")
                except Exception as e2:
                    print(f"[WARN] Explicit memory allocation failed: {str(e2)[:200]}")
                    # Strategy 2: Disable quantization and load in bfloat16
                    print(f"[INFO] Falling back to non-quantized bfloat16 loading...")
                    model_kwargs.pop("quantization_config", None)
                    model_kwargs.pop("max_memory", None)
                    model = LlavaForConditionalGeneration.from_pretrained(
                        config.MODEL_ID,
                        **model_kwargs
                    )
                    print(f"[INFO] Successfully loaded without quantization (bfloat16)")
            else:
                # Strategy 2: Disable quantization directly
                print(f"[INFO] Disabling quantization and retrying...")
                model_kwargs.pop("quantization_config", None)
                model = LlavaForConditionalGeneration.from_pretrained(
                    config.MODEL_ID,
                    **model_kwargs
                )
                print(f"[INFO] Successfully loaded without quantization")
        else:
            raise
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    # Disable caching for gradient checkpointing compatibility
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    
    # Enable gradient checkpointing (same as training)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    
    # Enable input gradients (same as training)
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    # ============================================================================
    # PROJECTOR REPLACEMENT
    # ============================================================================
    
    # Replace original projector with LMOLProjector (same as training)
    proj_path, old_proj = _find_projector_handle(model)
    assert old_proj is not None, "Projector not found on the loaded LLaVA model."
    
    # Create LMOL projector with same dimensions as training
    device = next(model.parameters()).device
    lmol_proj = LMOLProjector(vision_dim=1024, text_hidden_dim=4096).to(
        device=device, dtype=torch.bfloat16
    )
    
    # Replace projector using the same attribute path
    holder = model
    parts = proj_path.split(".")
    for p in parts[:-1]:
        holder = getattr(holder, p)
    setattr(holder, parts[-1], lmol_proj)
    
    # Set model to evaluation mode
    model.eval()
    
    return model
