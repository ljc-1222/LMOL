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
    
    # ===== VALIDATION BLOCK =====
    # Validate tokenizer configuration to ensure consistency
    print(f"[Validate] Tokenizer configuration:")
    
    # Check vocabulary size
    vocab_size = len(tok.get_vocab())
    expected_vocab_size = 32002  # LLaVA vocab size (LLaMA-2 32000 + 2 special tokens)
    print(f"  - Vocabulary size: {vocab_size}")
    if vocab_size != expected_vocab_size:
        print(f"  ⚠️  WARNING: Expected {expected_vocab_size}, got {vocab_size}")
    
    # Check EOS token
    if tok.eos_token_id is None:
        raise ValueError("EOS token not configured in loaded tokenizer")
    print(f"  - EOS token: {tok.eos_token} (ID: {tok.eos_token_id})")
    
    # Check PAD token (set to EOS if missing)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"  - PAD token: Set to EOS token (ID: {tok.pad_token_id})")
    else:
        print(f"  - PAD token: {tok.pad_token} (ID: {tok.pad_token_id})")
    
    # Check BOS token
    if tok.bos_token_id is not None:
        print(f"  - BOS token: {tok.bos_token} (ID: {tok.bos_token_id})")
    
    # Validate answer tokenization (with leading space to match collator)
    print(f"[Validate] Answer token sequences (with leading space):")
    for answer in [config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR]:
        # CRITICAL: Add leading space to match collator tokenization
        toks_cap = tok(' ' + answer, add_special_tokens=False).input_ids
        toks_lower = tok(' ' + answer.lower(), add_special_tokens=False).input_ids
        print(f"  - ' {answer}': {toks_cap}")
        print(f"  - ' {answer.lower()}': {toks_lower}")
    
    print(f"[Validate] ✓ Tokenizer validation complete")
    # ===== END VALIDATION BLOCK =====
    
    base = build_inference_base()
    model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  params: total={total:,} trainable={trainable:,} ({trainable/total:.6f})")
    return model, processor


@torch.no_grad()
def classify_pair(model, tokenizer, prompt_ids, prompt_attn, pixel_pair, use_constrained_generation=True) -> str:
    """
    Classify a single image pair.
    
    Args:
        model: LMOL model
        tokenizer: HuggingFace tokenizer
        prompt_ids: Input prompt token IDs
        prompt_attn: Attention mask
        pixel_pair: Pixel values for image pair
        use_constrained_generation: If True, force model to only output valid answer tokens
        
    Returns:
        Predicted answer: "First", "Second", or "Similar"
    """
    # NEW: Use constrained generation to guarantee valid outputs
    if use_constrained_generation:
        try:
            from utils.constrained_generation import AnswerConstraintProcessor
            
            # Create processor to restrict outputs to 3 answer tokens
            processor = AnswerConstraintProcessor(tokenizer, verbose=False)
            
            # Generate with constraint - only 1 token needed
            gen_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attn,
                pixel_values=pixel_pair,
                max_new_tokens=1,           # Only need 1 token for classification
                do_sample=False,             # Greedy decoding for deterministic results
                logits_processor=[processor], # FORCE valid outputs only
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # Extract generated token
            generated_token = gen_ids[0, -1].item()
            
            # Convert to answer string (guaranteed to be valid)
            answer = processor.get_answer_from_token_id(generated_token)
            if answer is None:
                # This should NEVER happen with constrained generation
                raise RuntimeError(f"Constrained generation failed! Generated invalid token: {generated_token}")
            
            return answer
        except ImportError:
            print("[WARN] Constrained generation not available, falling back to unconstrained generation")
            # Fall through to unconstrained generation
        except Exception as e:
            print(f"[WARN] Constrained generation failed: {e}, falling back to unconstrained generation")
            # Fall through to unconstrained generation
    
    # OLD: Original generation without constraints (kept for backwards compatibility)
    else:
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
    """
    Parse generated answer and return canonical label.
    
    This simplified version expects clean token generation matching training.
    If parsing fails, it logs the failure explicitly for debugging.
    
    Args:
        text_or_tokens: Generated tokens (torch.Tensor or list)
        tokenizer: Tokenizer for decoding (required)
        
    Returns:
        Canonical answer label ("First", "Second", or "Similar")
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for answer parsing")
    
    # Convert to list and remove special tokens
    if isinstance(text_or_tokens, torch.Tensor):
        gen_list = text_or_tokens.tolist()
    else:
        gen_list = list(text_or_tokens)
    
    # Remove EOS and PAD tokens
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    gen_list = [t for t in gen_list if t not in [eos_id, pad_id]]
    
    if not gen_list:
        print("[PARSE_WARN] Empty token sequence after filtering")
        return config.ANSWER_SIMILAR
    
    # Get expected token sequences for each answer (includes case variants)
    answer_seqs = _make_label_token_seqs(tokenizer)
    
    # Strategy 1: Exact prefix matching (most reliable)
    # Check if generated tokens start with any expected answer sequence variant
    for label, expected_seq_variants in answer_seqs.items():
        for expected_seq in expected_seq_variants:
            if len(gen_list) >= len(expected_seq):
                if gen_list[:len(expected_seq)] == expected_seq:
                    return label
    
    # Strategy 2: Subsequence matching (for cases where answer is not at start)
    # This handles generation artifacts like extra tokens before the answer
    for label, expected_seq_variants in answer_seqs.items():
        for expected_seq in expected_seq_variants:
            pos = _find_subsequence(gen_list, expected_seq)
            if pos >= 0:
                if pos <= 3:  # Only accept if answer appears within first 3 tokens
                    print(f"[PARSE_INFO] Found '{label}' at position {pos} (not at start)")
                    return label
    
    # Parsing failed - log for debugging
    decoded = tokenizer.decode(gen_list, skip_special_tokens=True).strip()
    print(f"[PARSE_FAIL] Generated: '{decoded}' | Tokens: {gen_list[:10]}")
    print(f"[PARSE_FAIL] Expected one of: {list(answer_seqs.keys())}")
    
    # Return default (most common class to minimize false positives)
    return config.ANSWER_SIMILAR


@torch.no_grad()
def _make_label_token_seqs(tokenizer) -> dict:
    """
    Build token sequences for each canonical answer.
    
    This function creates token sequences for both capitalized and lowercase versions
    of each answer to support flexible matching during evaluation.
    
    IMPORTANT: The collator constructs sequences as: prompt + ' ' + answer
    So we must tokenize answers WITH a leading space to match training labels.
    
    Returns:
        Dictionary mapping answer labels to list of token sequence variants
        Example: {"First": [[29871, 3824]], ...} where 29871 is space token
    """
    seqs = {}
    for label in (config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR):
        # CRITICAL: Add leading space to match collator tokenization
        # The collator constructs: prompt + ' ' + answer_text
        # Without the space, tokenization is different and matching will fail
        variants = [' ' + label, ' ' + label.lower()]  # Both " First" and " first"
        token_variants = []
        for variant in variants:
            try:
                # Tokenize answer without special tokens
                toks = tokenizer(variant, add_special_tokens=False).input_ids
            except Exception:
                toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks not in token_variants:  # Avoid duplicates
                token_variants.append(toks)
        seqs[label] = token_variants
    return seqs


def _find_subsequence(haystack: list, needle: list) -> int:
    """
    Find the first occurrence of needle subsequence in haystack.
    
    Args:
        haystack: List to search in
        needle: Subsequence to find
        
    Returns:
        Index of first occurrence, or -1 if not found
    """
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
        # Match training dtype for consistency (training uses float32)
        # Ensure both tensors are on the same device before stacking
        t1 = t1.to(device, dtype=torch.float32)
        t2 = t2.to(device, dtype=torch.float32)
        pixel_pair = torch.stack([t1, t2], dim=0)
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
