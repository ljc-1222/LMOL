# -*- coding: utf-8 -*-

from typing import Tuple, List
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from configs.config import config

def model_generator() -> Tuple[torch.nn.Module, any, any]:
    """
    Build LLaVA-1.5-7B with QLoRA for LMOL order learning.
    - Freeze the vision tower.
    - Train the multimodal projector (mm_projector) with lr=2e-5.
    - Attach LoRA to the LLM attention/projection with r=8, alpha=4 and lr=2e-4.
    Returns: (model, processor, tokenizer)
    """
    # 4-bit quantization config for QLoRA
    bnb_config = None
    if config.USE_4BIT:
        compute_dtype = torch.bfloat16 if config.BNB_4BIT_COMPUTE_DTYPE == "bfloat16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.BNB_DOUBLE_QUANT,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE
        )

    # Load model and processor
    model = LlavaForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)

    # Enable gradient checkpointing to reduce memory
    model.gradient_checkpointing_enable()

    # Freeze vision tower (CLIP ViT) exactly as LMOL does not fine-tune the encoder
    if hasattr(model, "vision_tower") and model.vision_tower is not None:
        for p in model.vision_tower.parameters():
            p.requires_grad = False

    # Ensure mm_projector is trainable
    projector_names: List[str] = []
    if hasattr(model, "multi_modal_projector") and model.multi_modal_projector is not None:
        for n, p in model.multi_modal_projector.named_parameters():
            p.requires_grad = True
            projector_names.append(f"multi_modal_projector.{n}")
    elif hasattr(model, "mm_projector") and model.mm_projector is not None:
        for n, p in model.mm_projector.named_parameters():
            p.requires_grad = True
            projector_names.append(f"mm_projector.{n}")
    else:
        projector_names = []  # some checkpoints may bake projection inside the language model; handled by LoRA

    # Prepare for k-bit training (injects cast/gradient hooks)
    model = prepare_model_for_kbit_training(model)

    # Attach LoRA to language model blocks
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]  # standard Vicuna-style targets
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        target_modules=lora_targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Put model in train mode by default
    model.train()
    return model, processor, tokenizer
