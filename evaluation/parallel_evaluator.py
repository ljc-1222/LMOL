# -*- coding: utf-8 -*-
"""
LMOL Parallel Evaluation Module

This module provides parallel evaluation capabilities for the LMOL project:
- Batch processing for improved GPU utilization
- Multi-GPU support for fold-level parallelism
- Concurrent image loading and preprocessing
- Memory-efficient parallel inference

Key Features:
- Batch inference for better GPU utilization
- Multi-GPU fold parallelism
- Concurrent data loading
- Progress tracking for parallel operations
"""

import os
import gc
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from peft import PeftModel
from tqdm import tqdm
from PIL import Image
import numpy as np

# Fix HuggingFace tokenizers parallelism warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.config import config
from model import build_inference_base
from data import read_pairs_csv, PairRecord, basic_image_loader


class ParallelEvaluator:
    """
    Parallel evaluator for LMOL models with batch processing and multi-GPU support.
    """
    
    def __init__(self, 
                 batch_size: int = 8,
                 max_workers: int = 4,
                 use_multi_gpu: bool = False,
                 device_ids: Optional[List[int]] = None):
        """
        Initialize parallel evaluator.
        
        Args:
            batch_size: Number of pairs to process in each batch
            max_workers: Maximum number of worker threads for data loading
            use_multi_gpu: Whether to use multiple GPUs for different folds
            device_ids: List of GPU device IDs to use
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_multi_gpu = use_multi_gpu
        self.device_ids = device_ids or list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
        
    def _tokens_per_image(self, processor) -> int:
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

    def _build_prompt_prefix(self, tokens_per_image: int) -> str:
        """Build prompt with image placeholders and question."""
        placeholder = '<image>' * (tokens_per_image * 2)
        return placeholder + ', ' + config.QUESTION_TEXT

    def _resolve_image_path(self, p: str) -> str:
        """Resolve image path with fallback to data directory."""
        if os.path.exists(p):
            return p
        base = os.path.basename(p)
        cand = os.path.join(config.IMAGE_DIR, base)
        if os.path.exists(cand):
            return cand
        return p

    def _load_image_batch(self, paths: List[str]) -> List[Image.Image]:
        """Load a batch of images in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            resolved_paths = [self._resolve_image_path(p) for p in paths]
            images = list(executor.map(basic_image_loader, resolved_paths))
        return images

    def _preprocess_batch(self, images: List[Image.Image], processor) -> torch.Tensor:
        """Preprocess a batch of images."""
        pixel_values = processor.image_processor(images, return_tensors='pt').pixel_values
        return pixel_values

    @torch.no_grad()
    def _classify_batch(self, 
                       model, 
                       tokenizer, 
                       prompt_ids, 
                       prompt_attn, 
                       pixel_pairs: torch.Tensor) -> List[str]:
        """
        Classify a batch of image pairs.
        
        Args:
            model: The model to use for inference
            tokenizer: Tokenizer for text processing
            prompt_ids: Pre-computed prompt token IDs
            prompt_attn: Pre-computed prompt attention mask
            pixel_pairs: Batch of pixel values for image pairs
            
        Returns:
            List of predicted labels
        """
        batch_size = pixel_pairs.shape[0]
        device = next(model.parameters()).device
        
        # Expand prompt for batch processing
        batch_prompt_ids = prompt_ids.expand(batch_size, -1)
        batch_prompt_attn = prompt_attn.expand(batch_size, -1)
        
        # Generate predictions for the batch
        gen_ids = model.generate(
            input_ids=batch_prompt_ids,
            attention_mask=batch_prompt_attn,
            pixel_values=pixel_pairs,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Parse answers for each sample in the batch
        predictions = []
        for i in range(batch_size):
            gen_only = gen_ids[i, prompt_ids.size(1):]
            pred = self._parse_answer(gen_only, tokenizer=tokenizer)
            predictions.append(pred)
            
        return predictions

    @torch.no_grad()
    def _parse_answer(self, text_or_tokens, tokenizer=None) -> str:
        """Parse generated answer and return canonical label using flexible matching."""
        # If we were given tokens, try multiple matching strategies
        if tokenizer is not None and hasattr(text_or_tokens, '__len__') and not isinstance(text_or_tokens, str):
            gen_list = text_or_tokens.tolist() if isinstance(text_or_tokens, torch.Tensor) else list(text_or_tokens)
            
            # Remove EOS and PAD tokens for cleaner processing
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id
            gen_list = [t for t in gen_list if t not in [eos_id, pad_id]]
            
            # Strategy 1: Exact token sequence matching (original method)
            seqs = self._make_label_token_seqs(tokenizer)
            earliest = None
            best = None
            for label, seq in seqs.items():
                pos = self._find_subsequence(gen_list, seq)
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
            
            # Strategy 4: Text-based fallback
            try:
                decoded = tokenizer.decode(gen_list, skip_special_tokens=True).strip().lower()
                text = decoded
            except Exception:
                text = ""
        else:
            text = (text_or_tokens or "").strip().lower()

        # Final text-based matching
        if 'first' in text and 'second' not in text and 'similar' not in text:
            return config.ANSWER_FIRST
        if 'second' in text and 'first' not in text and 'similar' not in text:
            return config.ANSWER_SECOND
        if 'similar' in text and 'first' not in text and 'second' not in text:
            return config.ANSWER_SIMILAR
        if '1' in text and '2' not in text:
            return config.ANSWER_FIRST
        if '2' in text and '1' not in text:
            return config.ANSWER_SECOND
        return config.ANSWER_SIMILAR

    @torch.no_grad()
    def _make_label_token_seqs(self, tokenizer):
        """Build representative token sequences for each canonical answer."""
        seqs = {}
        for label in (config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR):
            try:
                toks = tokenizer(label, add_special_tokens=False).input_ids
            except Exception:
                toks = tokenizer.encode(label, add_special_tokens=False)
            seqs[label] = toks
        return seqs

    def _find_subsequence(self, haystack: list, needle: list) -> int:
        """Find the first occurrence of needle subsequence in haystack."""
        if not needle:
            return -1
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            if haystack[i:i+n] == needle:
                return i
        return -1

    def evaluate_fold_parallel(self, 
                             fold_name: str, 
                             model, 
                             processor, 
                             eval_csv: Path, 
                             fold_path: Path, 
                             n_samples: int = None) -> Tuple[int, int, List[str], List[str]]:
        """
        Evaluate a single fold model with parallel processing.
        
        Args:
            fold_name: Name of the fold being evaluated
            model: The model to evaluate
            processor: Image processor
            eval_csv: Path to evaluation CSV file
            fold_path: Path to fold directory
            n_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Tuple of (correct_count, total_count, true_labels, predicted_labels)
        """
        start_ts = time.perf_counter()
        print(f"[Time] {fold_name} parallel eval start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        pair_records = read_pairs_csv(str(eval_csv))
        
        # Sample data if specified
        sample_size = n_samples if n_samples is not None else 1000
        if sample_size and sample_size > 0 and len(pair_records) > sample_size:
            print(f"[Sample] Using {sample_size}/{len(pair_records)} pairs ({sample_size/len(pair_records):.1%})")
            rng = random.Random(config.SEED + hash(fold_name) % 100000)
            pair_records = rng.sample(pair_records, sample_size)
        
        total = len(pair_records)
        correct = 0
        
        # Setup model and prompt
        tpi = self._tokens_per_image(processor)
        prompt_text = self._build_prompt_prefix(tpi)
        tokenizer = processor.tokenizer
        prompt_tok = tokenizer(prompt_text, add_special_tokens=True, return_tensors='pt')
        device = next(model.parameters()).device
        prompt_ids = prompt_tok.input_ids.to(device)
        prompt_attn = prompt_tok.attention_mask.to(device)
        
        # Initialize result lists
        y_true: List[str] = []
        y_pred: List[str] = []
        
        # Process in batches
        num_batches = (total + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=total, desc=f"{fold_name} parallel eval", unit='pair') as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total)
                batch_records = pair_records[start_idx:end_idx]
                
                # Prepare batch data
                batch_img1_paths = [pr.img1 for pr in batch_records]
                batch_img2_paths = [pr.img2 for pr in batch_records]
                batch_labels = [pr.label for pr in batch_records]
                
                # Load images in parallel
                batch_img1 = self._load_image_batch(batch_img1_paths)
                batch_img2 = self._load_image_batch(batch_img2_paths)
                
                # Preprocess images
                pixel1 = self._preprocess_batch(batch_img1, processor)
                pixel2 = self._preprocess_batch(batch_img2, processor)
                
                # Stack pairs: [batch_size, 2, channels, height, width]
                pixel_pairs = torch.stack([pixel1, pixel2], dim=1).to(device, dtype=torch.bfloat16)
                
                # Classify batch
                batch_predictions = self._classify_batch(
                    model, tokenizer, prompt_ids, prompt_attn, pixel_pairs
                )
                
                # Update results
                for i, (true_label, pred_label) in enumerate(zip(batch_labels, batch_predictions)):
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    if true_label == pred_label:
                        correct += 1
                
                # Update progress
                pbar.update(len(batch_records))
                pbar.set_postfix(acc=f"{correct/(start_idx + len(batch_records)):.4f}")
        
        acc = correct / total if total else 0.0
        dur = time.perf_counter() - start_ts
        print(f"[Result] {fold_name} parallel: accuracy={acc:.6f} ({correct}/{total}) | time={dur/60:.2f} min")
        
        return correct, total, y_true, y_pred


def evaluate_folds_parallel(fold_dirs: List[Path], 
                           eval_csvs: List[Path], 
                           model_type: str = "best",
                           batch_size: int = 8,
                           n_samples: int = None) -> Tuple[int, int, List[str], List[str]]:
    """
    Evaluate multiple folds in parallel using multiple GPUs.
    
    Args:
        fold_dirs: List of fold directories
        eval_csvs: List of evaluation CSV files
        model_type: Type of model to evaluate
        batch_size: Batch size for each fold
        n_samples: Number of samples per fold
        
    Returns:
        Tuple of (total_correct, total_samples, all_true_labels, all_predicted_labels)
    """
    from .evaluator import load_single_fold, free_model
    
    if not torch.cuda.is_available() or len(fold_dirs) == 1:
        # Fallback to sequential evaluation
        print("[Info] Using sequential evaluation (single GPU or single fold)")
        grand_correct = 0
        grand_total = 0
        grand_y_true = []
        grand_y_pred = []
        
        for fold_idx, (fold_path, eval_csv) in enumerate(zip(fold_dirs, eval_csvs), 1):
            evaluator = ParallelEvaluator(batch_size=batch_size)
            model, processor = load_single_fold(fold_path, model_type)
            
            c, t, y_true, y_pred = evaluator.evaluate_fold_parallel(
                fold_path.name, model, processor, eval_csv, fold_path, n_samples
            )
            
            grand_correct += c
            grand_total += t
            grand_y_true.extend(y_true)
            grand_y_pred.extend(y_pred)
            
            del processor
            free_model(model)
        
        return grand_correct, grand_total, grand_y_true, grand_y_pred
    
    # Multi-GPU parallel evaluation
    print(f"[Info] Using parallel evaluation with {len(fold_dirs)} GPUs")
    
    def evaluate_single_fold(args):
        """Worker function for parallel fold evaluation."""
        fold_path, eval_csv, device_id, batch_size, n_samples = args
        
        # Set device for this process
        torch.cuda.set_device(device_id)
        
        try:
            evaluator = ParallelEvaluator(batch_size=batch_size)
            model, processor = load_single_fold(fold_path, model_type)
            
            # Move model to specific device
            model = model.to(f'cuda:{device_id}')
            
            result = evaluator.evaluate_fold_parallel(
                fold_path.name, model, processor, eval_csv, fold_path, n_samples
            )
            
            del processor
            free_model(model)
            
            return result
            
        except Exception as e:
            print(f"[Error] Failed to evaluate {fold_path.name} on GPU {device_id}: {e}")
            return 0, 0, [], []
    
    # Prepare arguments for parallel processing
    device_ids = list(range(min(len(fold_dirs), torch.cuda.device_count())))
    args_list = [
        (fold_path, eval_csv, device_ids[i % len(device_ids)], batch_size, n_samples)
        for i, (fold_path, eval_csv) in enumerate(zip(fold_dirs, eval_csvs))
    ]
    
    # Execute parallel evaluation
    with ProcessPoolExecutor(max_workers=len(fold_dirs)) as executor:
        results = list(executor.map(evaluate_single_fold, args_list))
    
    # Aggregate results
    grand_correct = sum(result[0] for result in results)
    grand_total = sum(result[1] for result in results)
    grand_y_true = []
    grand_y_pred = []
    
    for result in results:
        grand_y_true.extend(result[2])
        grand_y_pred.extend(result[3])
    
    return grand_correct, grand_total, grand_y_true, grand_y_pred
