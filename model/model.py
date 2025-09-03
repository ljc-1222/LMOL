# -*- coding: utf-8 -*-

from typing import Optional
import torch
from transformers import AutoProcessor, AutoImageProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

from configs.config import config


# ----------------------------
# Projector utilities
# ----------------------------
def _get_projector(model):
    if hasattr(model, "multi_modal_projector") and model.multi_modal_projector is not None:
        return "multi_modal_projector", model.multi_modal_projector
    if hasattr(model, "mm_projector") and model.mm_projector is not None:
        return "mm_projector", model.mm_projector
    base = getattr(model, "model", None) or getattr(model, "get_model", lambda: None)()
    if base is not None:
        if hasattr(base, "multi_modal_projector") and base.multi_modal_projector is not None:
            return "model.multi_modal_projector", base.multi_modal_projector
        if hasattr(base, "mm_projector") and base.mm_projector is not None:
            return "model.mm_projector", base.mm_projector
    return "", None


def _extract_vision_hidden_size(model) -> Optional[int]:
    vt = getattr(model, "vision_tower", None)
    if vt is None:
        return None
    if hasattr(vt, "vision_tower"):
        vt = vt.vision_tower
    for attr in ["hidden_size", "embed_dim", "vision_hidden_size"]:
        if hasattr(vt, attr):
            val = getattr(vt, attr)
            if isinstance(val, int):
                return val
    cfg = getattr(vt, "config", None)
    if cfg is not None:
        for attr in ["hidden_size", "embed_dim"]:
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if isinstance(val, int):
                    return val
    return None


def _infer_hidden_dim(model, projector) -> Optional[int]:
    """Infer text hidden size via multiple fallbacks."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        hs = getattr(cfg, "hidden_size", None)
        if isinstance(hs, int):
            return hs
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            hs = getattr(text_cfg, "hidden_size", None)
            if isinstance(hs, int):
                return hs
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "embed_tokens"):
        emb = base.embed_tokens
        emb_dim = getattr(emb, "embedding_dim", None)
        if isinstance(emb_dim, int):
            return emb_dim
        if hasattr(emb, "weight") and emb.weight is not None:
            return emb.weight.shape[1]
    if projector is not None:
        for name in ["linear_2", "linear_1"]:
            layer = getattr(projector, name, None)
            if layer is not None:
                for attr in ["out_features", "out_feature", "features_out"]:
                    val = getattr(layer, attr, None)
                    if isinstance(val, int):
                        return val
                w = getattr(layer, "weight", None)
                if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
                    return w.shape[0]
    return None


def _replace_projector_linear4bit(projector: torch.nn.Module, vision_dim: int, hidden_dim: int, dtype=torch.bfloat16):
    """Replace 4-bit projector linear layers with nn.Linear using dequantized (or probed) weights."""
    for name in ["linear_1", "linear_2"]:
        if not hasattr(projector, name):
            continue
        layer = getattr(projector, name)
        if not isinstance(layer, bnb.nn.Linear4bit):
            continue
        if name == "linear_1":
            in_f, out_f = vision_dim, hidden_dim
        else:
            in_f, out_f = hidden_dim, hidden_dim
        with torch.no_grad():
            if hasattr(layer.weight, "dequantize"):
                w_full = layer.weight.dequantize()
            else:
                w_full = layer.weight.float()

            # Attempt to reshape/transposed-reshape; probe if packed
            if w_full.numel() == out_f * in_f:
                if w_full.dim() == 2 and w_full.shape == (out_f, in_f):
                    weight_matrix = w_full
                elif w_full.dim() == 2 and w_full.shape == (in_f, out_f):
                    weight_matrix = w_full.t()
                else:
                    weight_matrix = w_full.view(out_f, in_f)
            elif w_full.numel() * 2 == out_f * in_f:
                # Probe by feeding identity
                eye = torch.eye(in_f, device=w_full.device, dtype=dtype)
                outputs = []
                batch = 512 if in_f > 512 else in_f
                was_train = layer.training
                layer.eval()
                for s in range(0, in_f, batch):
                    e = min(s + batch, in_f)
                    out_chunk = layer(eye[s:e])
                    outputs.append(out_chunk.detach())
                if was_train:
                    layer.train()
                weight_matrix = torch.cat(outputs, dim=0).transpose(0, 1).contiguous().to(torch.float32)
            else:
                raise ValueError(
                    f"Unexpected weight size for {name}: {w_full.shape}, numel={w_full.numel()}, expected {out_f*in_f}"
                )

            if weight_matrix.shape != (out_f, in_f):
                raise ValueError(f"Final shape mismatch for {name}: {weight_matrix.shape} vs ({out_f},{in_f})")

            new_lin = torch.nn.Linear(in_f, out_f, bias=(layer.bias is not None), dtype=dtype, device=weight_matrix.device)
            new_lin.weight.copy_(weight_matrix.to(dtype))
            if layer.bias is not None:
                b = layer.bias.detach().float()
                if b.numel() != out_f:
                    raise ValueError(f"Bias size mismatch for {name}: {b.numel()} vs {out_f}")
                new_lin.bias.copy_(b.to(dtype))
        setattr(projector, name, new_lin)


# ----------------------------
# Freeze/Train helpers
# ----------------------------
def _freeze_all(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False

def _mark_lora_trainable(module: torch.nn.Module) -> None:
    # PEFT injects LoRA params, typically named with "lora_"
    for n, p in module.named_parameters():
        if "lora_" in n or "lora_A" in n or "lora_B" in n:
            p.requires_grad = True

def _mark_projector_trainable(projector: torch.nn.Module) -> None:
    for p in projector.parameters():
        p.requires_grad = True


# ----------------------------
# Public API
# ----------------------------
def model_generator():
    # 4-bit loading config (QLoRA)
    bnb_config = None
    if config.USE_4BIT:
        compute_dtype = torch.bfloat16 if config.BNB_4BIT_COMPUTE_DTYPE == "bfloat16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.BNB_DOUBLE_QUANT,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        )

    # Load LLaVA
    model = LlavaForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # IMPORTANT: disable cache to avoid checkpointing warning
    model.config.use_cache = False

    # Enable gradient checkpointing with explicit use_reentrant flag
    model.gradient_checkpointing_enable()

    # Processor + Fast image processor
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    # Try to upgrade to fast image processor to remove "slow image processor" warning
    try:
        fast_img_proc = AutoImageProcessor.from_pretrained(config.MODEL_ID, use_fast=True)
        if hasattr(processor, "image_processor") and processor.image_processor is not None:
            processor.image_processor = fast_img_proc
    except Exception:
        # Fall back silently if fast variant not available
        pass

    # Freeze vision tower explicitly
    if hasattr(model, "vision_tower") and model.vision_tower is not None:
        _freeze_all(model.vision_tower)

    # Prepare for k-bit training (PEFT utility)
    model = prepare_model_for_kbit_training(model)

    # Replace projector 4-bit Linear with fp32 nn.Linear and make it trainable
    proj_name, projector = _get_projector(model)
    if projector is not None:
        vision_dim = _extract_vision_hidden_size(model)
        hidden_dim = getattr(model.config, "hidden_size", None)
        if not isinstance(hidden_dim, int):
            hidden_dim = _infer_hidden_dim(model, projector)
        if vision_dim is None or hidden_dim is None:
            raise ValueError(f"Cannot determine vision_dim ({vision_dim}) or hidden_dim ({hidden_dim}) for projector replacement.")
        _replace_projector_linear4bit(projector, vision_dim=vision_dim, hidden_dim=hidden_dim, dtype=torch.bfloat16)

    # Apply LoRA to attention projections only
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_cfg = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        target_modules=lora_targets,
        task_type="CAUSAL_LM",
        modules_to_save=["mm_projector", "multi_modal_projector"],
    )
    model = get_peft_model(model, peft_cfg)

    # Ensure tokenizer has pad token
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Final freeze policy: everything off, then enable LoRA + projector
    _freeze_all(model)
    _mark_lora_trainable(model)
    if projector is not None:
        _mark_projector_trainable(projector)

    model.train()
    return model, processor, tok

def build_inference_base():
    """Return a base LLaVA model prepared like training (quantized + projector replacement) but WITHOUT creating new LoRA adapters.
    Used for evaluation: afterwards load saved PEFT adapter weights via PeftModel.from_pretrained.
    """
    bnb_config = None
    if config.USE_4BIT:
        compute_dtype = torch.bfloat16 if config.BNB_4BIT_COMPUTE_DTYPE == "bfloat16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.BNB_DOUBLE_QUANT,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        )
    model = LlavaForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    # Replace projector 4-bit layers similarly to training path
    proj_name, projector = _get_projector(model)
    if projector is not None:
        vision_dim = _extract_vision_hidden_size(model)
        hidden_dim = getattr(model.config, "hidden_size", None)
        if not isinstance(hidden_dim, int):
            hidden_dim = _infer_hidden_dim(model, projector)
        if vision_dim is not None and hidden_dim is not None:
            _replace_projector_linear4bit(projector, vision_dim, hidden_dim, dtype=torch.bfloat16)
    return model
