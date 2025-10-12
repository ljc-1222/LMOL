# -*- coding: utf-8 -*-
"""
LMOL Evaluation Core Logic

This module provides the core evaluation functionality for the LMOL project:
- Model loading and inference
- Evaluation pipeline
- Performance monitoring

Key Features:
- Robust model loading with fallback strategies
- Efficient inference with caching
- Comprehensive evaluation metrics
- Memory management and cleanup
"""

import os
import gc
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from tqdm import tqdm
from PIL import Image

# Fix HuggingFace tokenizers parallelism warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.config import config
from model import build_inference_base
from data import read_pairs_csv, PairRecord, basic_image_loader


class RepetitionStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria to detect and stop repetitive generation patterns."""
    
    def __init__(self, tokenizer, max_repetition_length: int = 10):
        self.tokenizer = tokenizer
        self.max_repetition_length = max_repetition_length
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        # Get the last generated tokens
        generated_tokens = input_ids[0].tolist()
        
        # Check for repetitive patterns
        if len(generated_tokens) >= self.max_repetition_length:
            # Look for repeating patterns in the last tokens
            last_tokens = generated_tokens[-self.max_repetition_length:]
            
            # Check for simple repetition (same token repeated)
            if len(set(last_tokens)) == 1:
                return True
                
            # Check for 2-token repetition patterns
            if len(last_tokens) >= 4:
                pattern = last_tokens[-2:]
                if all(last_tokens[i:i+2] == pattern for i in range(0, len(last_tokens)-1, 2)):
                    return True
                    
            # Check for 3-token repetition patterns
            if len(last_tokens) >= 6:
                pattern = last_tokens[-3:]
                if all(last_tokens[i:i+3] == pattern for i in range(0, len(last_tokens)-2, 3)):
                    return True
        
        return False


def _tokens_per_image(processor) -> int:
    """Calculate number of tokens per image based on patch size."""
    patch_size = getattr(getattr(processor, 'image_processor', {}), 'patch_size', 14)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    if isinstance(patch_size, dict):
        patch_size = patch_size.get('height', 14)
    size_cfg = getattr(getattr(processor, 'image_processor', {}), 'size', {})
    if isinstance(size_cfg, dict):
        h = size_cfg.get('height') or size_cfg.get('shortest_edge') or config.IMAGE_SIZE
        w = size_cfg.get('width') or size_cfg.get('shortest_edge') or config.IMAGE_SIZE
    else:
        h = w = config.IMAGE_SIZE
    return (h // patch_size) * (w // patch_size)


def _build_prompt_prefix(tokens_per_image: int) -> str:
    """Build prompt with image placeholders and question."""
    placeholder = '<image>' * (tokens_per_image * 2)
    return placeholder + ', ' + config.QUESTION_TEXT


def resolve_model_path(fold_path: Path, model_type: str) -> Path:
    """Resolve the model path based on the specified type (best, last, or directly in fold)."""
    if model_type in ("best", "last"):
        model_path = fold_path / model_type
        if model_path.exists():
            return model_path
        print(f"Warning: {model_type} directory not found in {fold_path}, falling back to fold directory")
    
    # Check if the fold directory itself contains a model
    if (fold_path / "adapter_config.json").exists():
        return fold_path
    
    # Try both best and last as fallbacks
    for fallback in ("best", "last"):
        fallback_path = fold_path / fallback
        if fallback_path.exists():
            print(f"Using {fallback} model as fallback")
            return fallback_path
    
    # If nothing else works, just return the fold path and let the loader handle potential errors
    return fold_path


def load_single_fold(fold_path: Path, model_type: str = "best") -> Tuple[PeftModel, AutoProcessor]:
    """Load a single fold model for evaluation."""
    model_path = resolve_model_path(fold_path, model_type)
    print(f"[Load] {fold_path.name} ({model_type}) from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    tok = processor.tokenizer
    if (tok.pad_token is None):
        tok.pad_token = tok.eos_token
    base = build_inference_base()
    model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  params: total={total:,} trainable={trainable:,} ({trainable/total:.6f})")
    return model, processor


@torch.no_grad()
def classify_pair(model, tokenizer, prompt_ids, prompt_attn, pixel_pair) -> str:
    """Classify a single image pair."""
    # Create custom stopping criteria to prevent repetitive generation
    repetition_stopper = RepetitionStoppingCriteria(tokenizer, max_repetition_length=8)
    stopping_criteria = StoppingCriteriaList([repetition_stopper])
    
    gen_ids = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_attn,
        pixel_values=pixel_pair,
        max_new_tokens=5,        # Increased for better generation
        do_sample=True,          # Enable sampling for better control
        temperature=0.1,         # Low temperature for deterministic-like output
        top_p=0.9,               # Nucleus sampling
        repetition_penalty=1.5,  # Stronger penalty for repetitive tokens
        no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria,
    )
    gen_only = gen_ids[0, prompt_ids.size(1):]
    
    # Use token-level parsing for robust matching (no need to decode to text)
    return _parse_answer(gen_only, tokenizer=tokenizer)


@torch.no_grad()
def _parse_answer(text_or_tokens, tokenizer=None) -> str:
    """Parse generated answer and return canonical label using flexible matching."""
    # If we were given tokens, try multiple matching strategies
    if tokenizer is not None and hasattr(text_or_tokens, '__len__') and not isinstance(text_or_tokens, str):
        gen_list = text_or_tokens.tolist() if isinstance(text_or_tokens, torch.Tensor) else list(text_or_tokens)
        
        # Remove EOS and PAD tokens for cleaner processing
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        gen_list = [t for t in gen_list if t not in [eos_id, pad_id]]
        
        # Strategy 1: Exact token sequence matching (original method)
        seqs = _make_label_token_seqs(tokenizer)
        earliest = None
        best = None
        for label, seq in seqs.items():
            pos = _find_subsequence(gen_list, seq)
            if pos >= 0 and (earliest is None or pos < earliest):
                earliest = pos
                best = label
        
        if best is not None:
            return best
        
        # Strategy 2: Flexible token matching - look for individual answer words
        # This handles cases where the model generates different tokens for the same words
        decoded_gen = tokenizer.decode(gen_list, skip_special_tokens=True).strip().lower()
        
        # Check for answer keywords in the generated text
        if 'first' in decoded_gen and 'second' not in decoded_gen and 'similar' not in decoded_gen:
            return config.ANSWER_FIRST
        elif 'second' in decoded_gen and 'first' not in decoded_gen and 'similar' not in decoded_gen:
            return config.ANSWER_SECOND
        elif 'similar' in decoded_gen and 'first' not in decoded_gen and 'second' not in decoded_gen:
            return config.ANSWER_SIMILAR
        
        # Strategy 3: Partial token matching - check if any individual tokens match
        # This handles cases where only part of the answer tokens are generated
        vocab = tokenizer.get_vocab()
        
        # Get all possible tokens for each answer word
        first_tokens = set()
        second_tokens = set()
        similar_tokens = set()
        
        # Find all tokens that decode to "first", "second", "similar"
        for token_text, token_id in vocab.items():
            decoded_token = tokenizer.decode([token_id], skip_special_tokens=True).lower().strip()
            if decoded_token == 'first':
                first_tokens.add(token_id)
            elif decoded_token == 'second':
                second_tokens.add(token_id)
            elif decoded_token == 'similar':
                similar_tokens.add(token_id)
        
        # Check if any generated tokens match our answer tokens
        gen_set = set(gen_list)
        
        if gen_set.intersection(first_tokens) and not gen_set.intersection(second_tokens) and not gen_set.intersection(similar_tokens):
            return config.ANSWER_FIRST
        elif gen_set.intersection(second_tokens) and not gen_set.intersection(first_tokens) and not gen_set.intersection(similar_tokens):
            return config.ANSWER_SECOND
        elif gen_set.intersection(similar_tokens) and not gen_set.intersection(first_tokens) and not gen_set.intersection(second_tokens):
            return config.ANSWER_SIMILAR
        
        # Strategy 4: Text-based fallback (original fallback)
        try:
            decoded = tokenizer.decode(gen_list, skip_special_tokens=True).strip()
            if decoded:
                print(f"[Fallback] All token matching failed, using text decode: '{decoded[:50]}{'...' if len(decoded) > 50 else ''}'")
                # Simple text matching as last resort
                decoded_lower = decoded.lower()
                if 'first' in decoded_lower:
                    return config.ANSWER_FIRST
                if 'second' in decoded_lower:
                    return config.ANSWER_SECOND
                if 'similar' in decoded_lower:
                    return config.ANSWER_SIMILAR
        except Exception as e:
            print(f"[Error] All parsing strategies failed: {e}")
    
    # If we got here, something went wrong - default to Similar
    return config.ANSWER_SIMILAR


@torch.no_grad()
def _make_label_token_seqs(tokenizer):
    """Build representative token sequences for each canonical answer."""
    seqs = {}
    for label in (config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR):
        try:
            toks = tokenizer(label, add_special_tokens=False).input_ids
        except Exception:
            toks = tokenizer.encode(label, add_special_tokens=False)
        seqs[label] = toks
        # Debug: Print token sequences for verification
        print(f"[TokenSeq] '{label}' -> {toks}")
    return seqs


def _find_subsequence(haystack: list, needle: list) -> int:
    """Find the first occurrence of needle subsequence in haystack."""
    if not needle:
        return -1
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1


@torch.no_grad()
def evaluate_fold(fold_name: str, model, processor, eval_csv: Path, fold_path: Path, n_samples: int = None) -> Tuple[int, int, List[str], List[str]]:
    """Evaluate a single fold model."""
    start_ts = time.perf_counter()
    print(f"[Time] {fold_name} eval start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pair_records = read_pairs_csv(str(eval_csv))
    
    # If n_samples is specified, override the default N_EVAL_SAMPLE
    sample_size = n_samples if n_samples is not None else 1000  # Default sample size
    
    if sample_size and sample_size > 0 and len(pair_records) > sample_size:
        print(f"[Sample] Using {sample_size}/{len(pair_records)} pairs ({sample_size/len(pair_records):.1%})")
        rng = random.Random(config.SEED + hash(fold_name) % 100000)
        pair_records = rng.sample(pair_records, sample_size)
    total = len(pair_records)
    correct = 0

    tpi = _tokens_per_image(processor)
    # Build prompt without an extra trailing space so tokens align with training collator
    prompt_text = _build_prompt_prefix(tpi)
    tokenizer = processor.tokenizer
    prompt_tok = tokenizer(prompt_text, add_special_tokens=True, return_tensors='pt')
    device = next(model.parameters()).device
    prompt_ids = prompt_tok.input_ids.to(device)
    prompt_attn = prompt_tok.attention_mask.to(device)

    image_cache: Dict[str, Image.Image] = {}
    pixel_cache: Dict[str, torch.Tensor] = {}

    # 用於蒐集真實與預測標籤
    y_true: List[str] = []
    y_pred: List[str] = []

    def _resolve(p: str) -> str:
        if os.path.exists(p):
            return p
        base = os.path.basename(p)
        cand = os.path.join(config.IMAGE_DIR, base)
        if os.path.exists(cand):
            return cand
        return p

    def get_pil(path: str) -> Image.Image:
        if path not in image_cache:
            image_cache[path] = basic_image_loader(path)
        return image_cache[path]

    def get_pixel(path: str) -> torch.Tensor:
        if path not in pixel_cache:
            img = get_pil(path)
            pv = processor.image_processor([img], return_tensors='pt').pixel_values[0]
            pixel_cache[path] = pv.cpu()
        return pixel_cache[path]

    prog = tqdm(pair_records, desc=f"{fold_name} eval", dynamic_ncols=True, unit='pair')
    for idx, pr in enumerate(prog):
        p1 = _resolve(pr.img1)
        p2 = _resolve(pr.img2)
        t1 = get_pixel(p1)
        t2 = get_pixel(p2)
        pixel_pair = torch.stack([t1, t2], dim=0).to(device, dtype=torch.bfloat16)
        pred_label = classify_pair(model, tokenizer, prompt_ids, prompt_attn, pixel_pair)
        y_true.append(pr.label)
        y_pred.append(pred_label)
        if pred_label == pr.label:
            correct += 1
        if (idx + 1) % 200 == 0:
            prog.set_postfix(acc=f"{correct/(idx+1):.4f}")
    acc = correct / total if total else 0.0
    dur = time.perf_counter() - start_ts
    print(f"[Result] {fold_name}: accuracy={acc:.6f} ({correct}/{total}) | time={dur/60:.2f} min")
    print(f"[Cache] images={len(image_cache)} pixel_tensors={len(pixel_cache)}")

    return correct, total, y_true, y_pred


def free_model(model):
    """Free model memory and clean up GPU cache."""
    try:
        model.to('cpu')
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
