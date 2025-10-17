# -*- coding: utf-8 -*-
"""
LMOL Consistency Evaluation Module

This module provides consistency evaluation functionality for the LMOL project:
- Tests both (P1, P2, Question) and (P2, P1, Question) for each sample
- Measures consistency between forward and reverse predictions
- Calculates accuracy for overall and consistent samples only

Key Features:
- Consistency measurement between forward and reverse predictions
- Detailed accuracy reporting for different sample groups
- Memory-efficient evaluation with caching
- Comprehensive logging and progress tracking
"""

import os
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, NamedTuple
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from tqdm import tqdm
from PIL import Image

# Fix HuggingFace tokenizers parallelism warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from configs.config import config
from model import build_inference_base
from data import read_pairs_csv, PairRecord, basic_image_loader
from .evaluator import classify_pair, _make_label_token_seqs, _build_prompt_prefix, _tokens_per_image


@dataclass
class ConsistencyResult:
    """Results from consistency evaluation."""
    total_samples: int
    consistent_samples: int
    overall_correct: int
    consistent_correct: int
    overall_accuracy: float
    consistent_accuracy: float
    consistency_rate: float
    y_true: List[str]
    y_pred_forward: List[str]
    y_pred_reverse: List[str]
    is_consistent: List[bool]
    evaluation_time: float


class RepetitionStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria to detect and stop repetitive generation patterns."""
    
    def __init__(self, tokenizer, max_repetition_length: int = 10):
        self.tokenizer = tokenizer
        self.max_repetition_length = max_repetition_length
        
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        # Get the last generated tokens
        generated_tokens = input_ids[0].tolist()
        
        # Check for repetitive patterns
        if len(generated_tokens) < self.max_repetition_length:
            return False
            
        # Look for repeated subsequences
        last_tokens = generated_tokens[-self.max_repetition_length:]
        for i in range(1, len(last_tokens) // 2 + 1):
            if last_tokens[-i:] == last_tokens[-2*i:-i]:
                return True
                
        return False


def is_consistent_prediction(pred1: str, pred2: str) -> bool:
    """
    Check if two predictions are logically consistent.
    
    For beauty comparison task:
    - If (P1, P2) predicts "First", then (P2, P1) should predict "Second"
    - If (P1, P2) predicts "Second", then (P2, P1) should predict "First"  
    - If (P1, P2) predicts "Similar", then (P2, P1) should predict "Similar"
    
    Args:
        pred1: Prediction for (P1, P2, Question)
        pred2: Prediction for (P2, P1, Question)
        
    Returns:
        True if predictions are consistent, False otherwise
    """
    if pred1 == config.ANSWER_SIMILAR and pred2 == config.ANSWER_SIMILAR:
        return True
    elif pred1 == config.ANSWER_FIRST and pred2 == config.ANSWER_SECOND:
        return True
    elif pred1 == config.ANSWER_SECOND and pred2 == config.ANSWER_FIRST:
        return True
    else:
        return False


def evaluate_consistency_fold(
    model: PeftModel,
    tokenizer: AutoProcessor,
    pair_records: List[PairRecord],
    fold_name: str,
    device: torch.device
) -> ConsistencyResult:
    """
    Evaluate a single fold with consistency checking.
    
    For each pair (P1, P2, Question), this function:
    1. Evaluates (P1, P2, Question) -> prediction1
    2. Evaluates (P2, P1, Question) -> prediction2  
    3. Checks if predictions are consistent
    4. Records accuracy for overall and consistent samples
    
    Args:
        model: Loaded PEFT model
        tokenizer: Tokenizer/processor
        pair_records: List of pair records to evaluate
        fold_name: Name of the fold for logging
        device: Device to run inference on
        
    Returns:
        ConsistencyResult with detailed metrics
    """
    print(f"[Consistency] Starting evaluation for {fold_name}")
    print(f"[Consistency] Total samples: {len(pair_records)}")
    
    start_ts = time.perf_counter()
    
    # Prepare prompt
    processor = tokenizer
    tpi = _tokens_per_image(processor)
    prompt_text = _build_prompt_prefix(tpi)
    tokenizer_obj = processor.tokenizer
    prompt_tok = tokenizer_obj(prompt_text, add_special_tokens=True, return_tensors='pt')
    prompt_ids = prompt_tok.input_ids.to(device)
    prompt_attn = prompt_tok.attention_mask.to(device)
    
    # Image and pixel caching for efficiency
    image_cache = {}
    pixel_cache = {}
    
    def _resolve(p: str) -> str:
        """Resolve image path with fallback strategies."""
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

    # Initialize tracking variables
    total_samples = len(pair_records)
    consistent_samples = 0
    overall_correct = 0
    consistent_correct = 0
    
    y_true = []
    y_pred_forward = []
    y_pred_reverse = []
    is_consistent = []
    
    prog = tqdm(pair_records, desc=f"{fold_name} consistency eval", dynamic_ncols=True, unit='pair')
    
    for idx, pr in enumerate(prog):
        # Get image paths
        p1 = _resolve(pr.img1)
        p2 = _resolve(pr.img2)
        
        # Load and prepare images
        t1 = get_pixel(p1)
        t2 = get_pixel(p2)
        t1 = t1.to(device=device, dtype=torch.float32)
        t2 = t2.to(device=device, dtype=torch.float32)
        
        # Forward evaluation: (P1, P2, Question)
        pixel_pair_forward = torch.stack([t1, t2], dim=0)
        pred_forward = classify_pair(model, tokenizer_obj, prompt_ids, prompt_attn, pixel_pair_forward)
        
        # Reverse evaluation: (P2, P1, Question)  
        pixel_pair_reverse = torch.stack([t2, t1], dim=0)
        pred_reverse = classify_pair(model, tokenizer_obj, prompt_ids, prompt_attn, pixel_pair_reverse)
        
        # Check consistency
        consistent = is_consistent_prediction(pred_forward, pred_reverse)
        
        # Record results
        y_true.append(pr.label)
        y_pred_forward.append(pred_forward)
        y_pred_reverse.append(pred_reverse)
        is_consistent.append(consistent)
        
        # Update counters
        if consistent:
            consistent_samples += 1
            # For consistent samples, check if forward prediction matches ground truth
            if pred_forward == pr.label:
                consistent_correct += 1
        
        # Overall accuracy: check if forward prediction matches ground truth
        if pred_forward == pr.label:
            overall_correct += 1
        
        # Update progress
        if (idx + 1) % 100 == 0:
            current_consistency = consistent_samples / (idx + 1)
            current_overall_acc = overall_correct / (idx + 1)
            current_consistent_acc = consistent_correct / consistent_samples if consistent_samples > 0 else 0.0
            prog.set_postfix(
                cons_rate=f"{current_consistency:.3f}",
                overall_acc=f"{current_overall_acc:.3f}",
                cons_acc=f"{current_consistent_acc:.3f}"
            )
    
    # Calculate final metrics
    overall_accuracy = overall_correct / total_samples if total_samples > 0 else 0.0
    consistent_accuracy = consistent_correct / consistent_samples if consistent_samples > 0 else 0.0
    consistency_rate = consistent_samples / total_samples if total_samples > 0 else 0.0
    evaluation_time = time.perf_counter() - start_ts
    
    # Print detailed results
    print(f"[Consistency Result] {fold_name}:")
    print(f"  Total samples: {total_samples}")
    print(f"  Consistent samples: {consistent_samples} ({consistency_rate:.3f})")
    print(f"  Overall accuracy: {overall_accuracy:.6f} ({overall_correct}/{total_samples})")
    print(f"  Consistent accuracy: {consistent_accuracy:.6f} ({consistent_correct}/{consistent_samples})")
    print(f"  Evaluation time: {evaluation_time/60:.2f} min")
    print(f"  Cache: images={len(image_cache)} pixel_tensors={len(pixel_cache)}")
    
    return ConsistencyResult(
        total_samples=total_samples,
        consistent_samples=consistent_samples,
        overall_correct=overall_correct,
        consistent_correct=consistent_correct,
        overall_accuracy=overall_accuracy,
        consistent_accuracy=consistent_accuracy,
        consistency_rate=consistency_rate,
        y_true=y_true,
        y_pred_forward=y_pred_forward,
        y_pred_reverse=y_pred_reverse,
        is_consistent=is_consistent,
        evaluation_time=evaluation_time
    )


def load_single_fold_consistency(run_dir: Path, fold_name: str, model_type: str) -> Tuple[PeftModel, AutoProcessor, torch.device]:
    """
    Load model and tokenizer for consistency evaluation.
    
    This is a wrapper around the original load_single_fold function to ensure
    compatibility with the consistency evaluation pipeline.
    """
    from .evaluator import load_single_fold
    fold_path = run_dir / fold_name
    model, tokenizer = load_single_fold(fold_path, model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer, device


def free_model_consistency(model: PeftModel, tokenizer: AutoProcessor):
    """
    Free model and tokenizer resources for consistency evaluation.
    
    This is a wrapper around the original free_model function to ensure
    compatibility with the consistency evaluation pipeline.
    """
    from .evaluator import free_model
    free_model(model, tokenizer)
