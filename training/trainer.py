# -*- coding: utf-8 -*-
"""
LMOL Custom Trainer Implementation

This module implements the WeightedSwapConsistencyTrainer, a custom trainer
that implements the novel LMOL training approach with swap consistency regularization.

Key Features:
- Weighted cross-entropy loss with class-specific weights
- Swap consistency regularization using symmetric KL divergence
- Dynamic consistency weight scheduling for training stability
- Clean, minimal logging output
- Comprehensive loss tracking and monitoring

Loss Formula:
    total_loss = weighted_ce_loss + λ(t) * consistency_loss

Where:
- weighted_ce_loss: Cross-entropy with class weights and swap sample weights
- consistency_loss: Symmetric KL divergence between (A,B) and (B,A) predictions
- λ(t): Dynamic consistency weight that increases over training
"""

# Standard library imports
import math
import os
from typing import List, Dict, Any, Optional

# Third-party imports
import torch
from transformers import Trainer, TrainerCallback

# Fix HuggingFace tokenizers parallelism warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.config import config
from utils.constants import IGNORE_INDEX
from utils.memory_manager import MemoryManager, get_memory_manager


class WeightedSwapConsistencyTrainer(Trainer):
    """
    Custom trainer implementing weighted cross-entropy loss with swap consistency regularization.
    
    This trainer implements the novel LMOL training approach:
    - Weighted cross-entropy loss with class-specific weights
    - Swap consistency regularization using symmetric KL divergence
    - Dynamic consistency weight scheduling for training stability
    - Clean, minimal logging output
    
    Loss Formula:
        total_loss = weighted_ce_loss + λ(t) * consistency_loss
    
    Where:
    - weighted_ce_loss: Cross-entropy with class weights and swap sample weights
    - consistency_loss: Symmetric KL divergence between (A,B) and (B,A) predictions
    - λ(t): Dynamic consistency weight that increases over training
    
    Key Features:
    - Token-level loss computation with proper masking
    - Robust consistency loss computation with pair alignment
    - Dynamic weight scheduling for improved training stability
    - Single-line logging format for easy monitoring
    - Automatic best model saving based on training loss
    """
    def __init__(self, tokenizer=None, w_sim=config.WSIM, cons_weight=config.CONS_WEIGHT,
                 swap_ce_weight=getattr(config, 'SWAP_CE_WEIGHT', 1.0),
                 forward_strategy=config.FORWARD_STRATEGY,
                 use_dynamic_consistency=False, cons_weight_start=5.0, cons_weight_end=20.0,
                 cons_weight_ramp_ratio=0.5, total_steps=None, 
                 enable_memory_cleanup=getattr(config, 'ENABLE_MEMORY_CLEANUP', True), 
                 memory_cleanup_frequency=getattr(config, 'MEMORY_CLEANUP_FREQUENCY', 10),
                 **kwargs):
        """
        Initialize the WeightedSwapConsistencyTrainer.
        
        Args:
            tokenizer: HuggingFace tokenizer for text processing
            w_sim: Weight for 'Similar' class in cross-entropy loss
            cons_weight: Base consistency weight for swap consistency regularization
            swap_ce_weight: Weight multiplier for cross-entropy on swapped samples
            forward_strategy: Forward pass strategy ('double' for swap doubling)
            use_dynamic_consistency: Enable dynamic consistency weight scheduling
            cons_weight_start: Starting consistency weight (lower for stability)
            cons_weight_end: Final consistency weight (higher for regularization)
            cons_weight_ramp_ratio: Fraction of training to reach final weight
            total_steps: Total training steps for scheduling
            enable_memory_cleanup: Enable intelligent memory cleanup during training
            memory_cleanup_frequency: How often to perform cleanup (every N batches)
            **kwargs: Additional arguments passed to Trainer
        """
        if tokenizer is not None:
            kwargs["processing_class"] = kwargs.get("processing_class", tokenizer)
        super().__init__(**kwargs)
        
        # Remove the default callbacks that print verbose dict logs
        # We use our own clean logging format, but keep ProgressCallback for tqdm
        from transformers.trainer_callback import PrinterCallback, DefaultFlowCallback, ProgressCallback
        
        # Remove callbacks more aggressively
        self.remove_callback(PrinterCallback)
        self.remove_callback(DefaultFlowCallback)
        
        # Double-check and remove any remaining DefaultFlowCallback or PrinterCallback instances
        callbacks_to_remove = []
        for callback in self.callback_handler.callbacks:
            # Remove any callback that might print logs, but keep ProgressCallback for tqdm
            if isinstance(callback, (DefaultFlowCallback, PrinterCallback)):
                callbacks_to_remove.append(callback)
        
        for callback in callbacks_to_remove:
            self.remove_callback(callback)
        
        # Set logging level to suppress verbose output
        import logging
        logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
        
        # Add flag to track if we should suppress prints from callbacks
        self._suppress_callback_prints = True
        
        
        # Store loss configuration
        self.w_sim = float(w_sim)
        self.cons_weight = float(cons_weight)
        self.swap_ce_weight = float(swap_ce_weight)
        self.forward_strategy = forward_strategy
        
        # Dynamic consistency weight configuration
        self.use_dynamic_consistency = use_dynamic_consistency
        self.cons_weight_start = float(cons_weight_start)
        self.cons_weight_end = float(cons_weight_end)
        self.cons_weight_ramp_ratio = float(cons_weight_ramp_ratio)
        self.total_steps = total_steps
        
        # Buffers for accumulating metrics within a batch
        self.batch_metrics_buffer = {
            'loss': [],
            'ce_loss': [],
            'cons_loss': [],
            'grad_norm': []
        }
        
        # Track accumulation steps to know when to log
        self._accumulation_counter = 0
        
        # Track first batch to suppress verbose model output
        self._first_batch_complete = False
        
        # Memory management configuration
        self.enable_memory_cleanup = enable_memory_cleanup
        self.memory_cleanup_frequency = memory_cleanup_frequency
        
        # Initialize memory manager with minimal overhead
        if self.enable_memory_cleanup:
            self.memory_manager = MemoryManager(
                enable_monitoring=False,  # Disable monitoring to reduce overhead
                cleanup_frequency=memory_cleanup_frequency,
                aggressive_cleanup=False,
                log_memory_usage=False  # Disable logging to reduce overhead
            )
            # Memory cleanup initialization message disabled to reduce console output
            # print(f"[MEMORY] Lightweight memory cleanup enabled (frequency: every {memory_cleanup_frequency} batches)")
        else:
            self.memory_manager = None
        
        # Setup class token IDs for consistency loss computation
        # Note: This must be called after super().__init__() to ensure tokenizer is available
        self._setup_class_token_ids()
        
        # Apply torch.compile optimization if enabled
        if getattr(config, 'USE_TORCH_COMPILE', False):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception as e:
                print(f"[WARN] torch.compile failed: {e}, continuing without compilation")
        
    def _setup_class_token_ids(self):
        """
        Setup class token IDs for consistency loss computation.
        
        Determines the token IDs for "First", "Second", and "Similar" answers
        to enable robust consistency loss computation between swapped pairs.
        """
        # Get token sequences for canonical answers
        first_text = "First"
        second_text = "Second"
        similar_text = "Similar"
        toks_first = self.tokenizer(first_text, add_special_tokens=False).input_ids
        toks_second = self.tokenizer(second_text, add_special_tokens=False).input_ids
        toks_similar = self.tokenizer(similar_text, add_special_tokens=False).input_ids
        
        # Try vocabulary lookup for single-word tokens first
        vocab = self.tokenizer.get_vocab()
        first_id = vocab.get("first", vocab.get(" first", -1))
        second_id = vocab.get("second", vocab.get(" second", -1))
        similar_id = vocab.get("similar", vocab.get(" similar", -1))
        
        if first_id != -1 and second_id != -1 and similar_id != -1:
            # Use single-word tokens if available
            self.cls_first_id = first_id
            self.cls_second_id = second_id
            self.cls_sim_id = similar_id
        else:
            # Fallback to first token from representative full answer strings
            full_first = self.tokenizer("The first person is more attractive.", add_special_tokens=False).input_ids
            full_second = self.tokenizer("The second person is more attractive.", add_special_tokens=False).input_ids
            full_similar = self.tokenizer("The two people are similarly attractive.", add_special_tokens=False).input_ids
            self.cls_first_id = full_first[0]
            self.cls_second_id = full_second[0]
            self.cls_sim_id = full_similar[0]
    
    @property
    def tokenizer(self):
        """Get the tokenizer from processing class."""
        return self.processing_class
    
    def get_current_consistency_weight(self, step: int) -> float:
        """
        Calculate current consistency weight based on training progress.
        
        Implements dynamic consistency weight scheduling:
        - Starts at CONS_WEIGHT_START for early training stability
        - Gradually increases to CONS_WEIGHT_END for stronger regularization
        - Uses linear interpolation over CONS_WEIGHT_RAMP_RATIO of training
        
        Args:
            step: Current training step
            
        Returns:
            Current consistency weight to apply
        """
        if not self.use_dynamic_consistency or self.total_steps is None:
            return self.cons_weight
        
        # Calculate progress (0.0 to 1.0)
        progress = min(step / (self.total_steps * self.cons_weight_ramp_ratio), 1.0)
        
        # Linear interpolation from start to end weight
        current_weight = self.cons_weight_start + (self.cons_weight_end - self.cons_weight_start) * progress
        
        return current_weight
    
    @staticmethod
    def _answer_start_positions(labels: torch.Tensor) -> torch.Tensor:
        """
        Find the first non-ignore index position for each sample.
        
        Args:
            labels: Label tensor with IGNORE_INDEX for ignored positions
            
        Returns:
            Tensor of start positions for each sample
        """
        N, L = labels.shape
        device = labels.device
        starts = torch.zeros(N, dtype=torch.long, device=device)
        for i in range(N):
            nz = torch.nonzero(labels[i] != IGNORE_INDEX, as_tuple=False)
            starts[i] = nz[0, 0] if nz.numel() else 0
        return starts
    
    @staticmethod
    def _get_answer_ranges(labels: torch.Tensor) -> torch.Tensor:
        """
        Get (start, end) indices of answer token ranges for each sample.
        
        Args:
            labels: Label tensor with IGNORE_INDEX for ignored positions
            
        Returns:
            Tuple of (start_indices, end_indices) tensors
        """
        N, L = labels.shape
        device = labels.device
        starts = torch.zeros(N, dtype=torch.long, device=device)
        ends = torch.zeros(N, dtype=torch.long, device=device)
        for i in range(N):
            nz = torch.nonzero(labels[i] != IGNORE_INDEX, as_tuple=False)
            if nz.numel():
                starts[i] = nz[0, 0]
                ends[i] = nz[-1, 0] + 1  # exclusive end
            else:
                ends[i] = 0  # empty range
        return starts, ends
    
    def log(self, logs: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Custom logging method for clean, minimal training output.
        
        Overrides the default Trainer logging to provide:
        - Single-line logging format with key metrics
        - Metrics averaged across gradient accumulation steps
        - Clean formatting without verbose dict outputs
        
        Args:
            logs: Dictionary of metrics to log
            step: Current training step (auto-detected if None)
        """
        if step is None:
            step = getattr(self.state, 'global_step', 0)
        
        # Filter out unwanted keys to prevent them from being printed by default mechanisms
        # Keep only the keys we need for our custom logging
        filtered_logs = {}
        for key in ['loss', 'ce_loss', 'cons_loss', 'grad_norm']:
            if key in logs:
                filtered_logs[key] = logs[key]
                self.batch_metrics_buffer[key].append(logs[key])
        
        # Increment accumulation counter
        self._accumulation_counter += 1
        
        # Only log when we've accumulated metrics from all gradient accumulation steps
        if self._accumulation_counter < self.args.gradient_accumulation_steps:
            return
        
        # Reset counter for next batch
        self._accumulation_counter = 0
        
        # Calculate effective batch number (number of optimizer steps completed)
        effective_batch_num = step
        
        # Compute averaged metrics across all accumulation steps in this batch
        avg_loss = sum(self.batch_metrics_buffer['loss']) / len(self.batch_metrics_buffer['loss']) if self.batch_metrics_buffer['loss'] else 0.0
        avg_ce_loss = sum(self.batch_metrics_buffer['ce_loss']) / len(self.batch_metrics_buffer['ce_loss']) if self.batch_metrics_buffer['ce_loss'] else 0.0
        avg_cons_loss = sum(self.batch_metrics_buffer['cons_loss']) / len(self.batch_metrics_buffer['cons_loss']) if self.batch_metrics_buffer['cons_loss'] else 0.0
        # Use the last grad_norm (final accumulated gradient) rather than average
        avg_grad_norm = self.batch_metrics_buffer['grad_norm'][-1] if self.batch_metrics_buffer['grad_norm'] else 0.0
        
        # Clear buffers for next batch
        for key in self.batch_metrics_buffer:
            self.batch_metrics_buffer[key].clear()
        
        # Lightweight memory cleanup at batch end - only every N batches
        if (self.enable_memory_cleanup and self.memory_manager is not None and 
            effective_batch_num % self.memory_cleanup_frequency == 0):
            # Only do GPU cache cleanup, not tensor cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        
        # Get learning rates for both parameter groups
        lr_proj = 0.0
        lr_lora = 0.0
        
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            try:
                # param_groups[0] = projector, param_groups[1] = LoRA
                if len(self.optimizer.param_groups) >= 1:
                    lr_proj = self.optimizer.param_groups[0]['lr']
                if len(self.optimizer.param_groups) >= 2:
                    lr_lora = self.optimizer.param_groups[1]['lr']
            except Exception:
                # Fallback to logs if optimizer access fails
                lr_lora = logs.get('learning_rate', 0.0)
        
        # Calculate epoch progress as float
        # HuggingFace Trainer's state.epoch is an integer starting at 0
        # We need to calculate the actual epoch progress including fractional parts
        if hasattr(self.state, 'epoch') and self.state.epoch is not None:
            base_epoch = int(self.state.epoch)
            # Calculate steps per epoch
            steps_per_epoch = len(self.get_train_dataloader())
            if steps_per_epoch > 0:
                # Calculate fractional epoch progress within current epoch
                fractional_progress = (effective_batch_num % steps_per_epoch) / steps_per_epoch
                epoch_progress = base_epoch + fractional_progress
            else:
                epoch_progress = float(base_epoch)
        else:
            epoch_progress = 0.0
        
        print(f"Batch {effective_batch_num:4d} | Epoch {epoch_progress:.5f} | "
              f"Loss: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, Cons: {avg_cons_loss:.4f}) | "
              f"LR: Proj = {lr_proj:.2e}, LoRA = {lr_lora:.2e} | Grad: {avg_grad_norm:.2e}")
        
        # Prepare callback logs for internal state tracking (don't print)
        callback_logs = {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'cons_loss': avg_cons_loss,
            'grad_norm': avg_grad_norm,
            'epoch': epoch_progress,
            'learning_rate': lr_lora,
            'step': effective_batch_num,
        }
        
        # Update state and trigger callbacks WITHOUT any verbose output
        # We handle all printing ourselves above, so just update internal state silently
        if hasattr(self, 'state'):
            self.state.log_history.append(callback_logs)
        
        # Call callback handler - temporarily suppress stdout to prevent unwanted dict printing
        # This prevents the transformers library from printing the logs dictionary
        # while still allowing our SaveBestTrainingLossCallback to print notifications
        if hasattr(self, 'control') and hasattr(self, 'callback_handler'):
            # Capture stdout to suppress unwanted prints, but save our callback prints
            import sys
            import io
            
            # Create a custom stdout that only suppresses dictionary-like output
            class FilteredStdout:
                def __init__(self, original_stdout):
                    self.original_stdout = original_stdout
                    self.buffer = ""
                
                def write(self, text):
                    # If it looks like a dictionary with our keys, suppress it
                    if ('{' in text and 'loss' in text and 'ce_loss' in text and 
                        'cons_loss' in text and 'grad_norm' in text):
                        # Skip this output - it's the unwanted dict print
                        return
                    # Otherwise, pass through to original stdout
                    self.original_stdout.write(text)
                
                def flush(self):
                    self.original_stdout.flush()
            
            old_stdout = sys.stdout
            sys.stdout = FilteredStdout(old_stdout)
            try:
                self.control = self.callback_handler.on_log(
                    self.args, self.state, self.control, callback_logs
                )
            finally:
                sys.stdout = old_stdout
    
    def _log(self, logs: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Override internal _log method to use custom logging.
        
        Args:
            logs: Dictionary of metrics to log
            step: Current training step
        """
        # Don't call the parent's log method to avoid printing dictionary logs
        # We handle all logging ourselves in the custom log method
        pass
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the composite loss for LMOL training.
        
        This is the core method implementing the LMOL loss function:
        
        total_loss = weighted_ce_loss + λ(t) * consistency_loss
        
        Where:
        - weighted_ce_loss: Cross-entropy with class weights and swap sample weights
        - consistency_loss: Symmetric KL divergence between (A,B) and (B,A) predictions
        - λ(t): Dynamic consistency weight that increases over training
        
        Key Features:
        1. Token-level loss computation with proper masking
        2. Class weighting (Similar class gets WSIM weight)
        3. Swap sample weighting for different CE loss on swapped samples
        4. Robust consistency loss with pair alignment validation
        5. Dynamic consistency weight scheduling
        6. Comprehensive loss tracking and logging
        
        Args:
            model: The LMOL model
            inputs: Batch inputs containing input_ids, attention_mask, labels, pixel_values, label_ids, pair_ids
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (unused)
            
        Returns:
            Loss tensor (and optionally model outputs)
        """
        label_ids: torch.Tensor | None = inputs.pop("label_ids", None)
        pair_ids: torch.Tensor | None = inputs.pop("pair_ids", None)
        
        # Get model outputs (suppress stdout on first batch to avoid verbose transformers messages)
        if not self._first_batch_complete:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                outputs = model(**inputs)
            finally:
                sys.stdout = old_stdout
                self._first_batch_complete = True
        else:
            outputs = model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        labels = inputs["labels"]  # [batch_size, seq_len] where IGNORE_INDEX are ignored positions
        
        # Create a mask for swapped samples (if SWAP_DOUBLE=True)
        swap_mask = None
        if pair_ids is not None and config.SWAP_DOUBLE and self.swap_ce_weight != 1.0:
            # Determine which samples are swapped (odd indices in consecutive pairs)
            N = pair_ids.size(0)
            swap_mask = torch.zeros(N, dtype=torch.bool, device=pair_ids.device)
            
            # Find samples that are second in each pair (swapped samples)
            if N >= 2:
                for i in range(0, N-1, 2):
                    if pair_ids[i] == pair_ids[i+1]:  # Same pair ID means they're a pair
                        swap_mask[i+1] = True  # Mark the second sample as swapped
                
                # Note: Odd batch sizes skip consistency silently (rare with standard configs)
        
        # First check if there are any valid labels
        valid_mask = (labels != IGNORE_INDEX)
        if not valid_mask.any():
            # If no valid labels, fall back to original HF CE loss
            base_ce = outputs.loss
            self._last_base_ce = float(base_ce.item())
            self._last_ce = float(base_ce.item())
            self._last_weight = 1.0
            ce = base_ce
        else:
            # Create weights for each sample
            batch_size = labels.shape[0]
            weights = torch.ones(batch_size, dtype=torch.float32, device=labels.device)
            
            # Apply class weights (Similar=2 gets WSIM weight)
            if label_ids is not None:
                weights[label_ids == 2] = float(config.WSIM)
                
                # Apply swap weights if needed
                if swap_mask is not None:
                    weights[swap_mask] *= self.swap_ce_weight
                
                # Record weight info
                self._last_weight = float(weights.mean().item()) if weights.numel() > 0 else 1.0
            
            # Compute per-token CE loss using PyTorch's built-in function
            # This is more numerically stable than manual log_softmax + gather
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            
            # Apply sample weights to each token
            token_weights = None
            if weights is not None:
                # Map sample weights to token weights
                batch_size, seq_len = labels.shape
                sample_indices = torch.div(torch.arange(batch_size * seq_len, device=labels.device), 
                                         seq_len, rounding_mode='floor')
                token_weights = weights.gather(dim=0, index=sample_indices)
                token_weights = token_weights * (flat_labels != IGNORE_INDEX).float()
                
                # Normalize weights
                weight_sum = token_weights.sum()
                if weight_sum > 0:
                    token_weights = token_weights / weight_sum
            
            # Compute loss with PyTorch's cross_entropy
            ce_loss = torch.nn.functional.cross_entropy(
                flat_logits, 
                flat_labels,
                ignore_index=-100,
                reduction='none'
            )
            
            # Apply weights if needed
            if token_weights is not None and token_weights.sum() > 0:
                ce = (ce_loss * token_weights).sum()
            else:
                ce = ce_loss.mean()
            
            # Store base CE (unweighted) for logging
            self._last_base_ce = float(ce_loss[flat_labels != -100].mean().item())
            self._last_ce = float(ce.item())
            
            # Calculate and store CE ratio between original and swapped samples
            if swap_mask is not None and swap_mask.any():
                orig_indices = torch.nonzero(~swap_mask, as_tuple=True)[0]
                swap_indices = torch.nonzero(swap_mask, as_tuple=True)[0]
                
                if orig_indices.numel() > 0 and swap_indices.numel() > 0:
                    # Get per-sample losses (vectorized)
                    per_token_loss = ce_loss.clone()
                    loss_per_sample = torch.zeros(batch_size, device=per_token_loss.device)
                    token_counts = torch.zeros(batch_size, device=per_token_loss.device)
                    
                    # Use scatter_add for more efficient accumulation
                    valid_indices = (flat_labels != -100).nonzero(as_tuple=True)[0]
                    sample_indices = torch.div(valid_indices, seq_len, rounding_mode='floor')
                    
                    for i, idx in enumerate(sample_indices):
                        loss_per_sample[idx] += per_token_loss[valid_indices[i]]
                        token_counts[idx] += 1
                    
                    # Normalize by token count per sample
                    token_counts = torch.clamp(token_counts, min=1)  # Avoid division by zero
                    loss_per_sample /= token_counts
                    
                    # Calculate mean loss for original and swapped samples
                    orig_loss = loss_per_sample[orig_indices].mean()
                    swap_loss = loss_per_sample[swap_indices].mean()
                    
                    # Calculate ratio with numerical stability
                    if swap_loss > 1e-6:
                        curr_ratio = float(orig_loss.item() / swap_loss.item())
                        self._last_ce_ratio = curr_ratio
                        
                        # Update EMA
                        if getattr(self, '_ema_ce_ratio', None) is None:
                            self._ema_ce_ratio = curr_ratio
                        else:
                            alpha = getattr(self, '_ema_alpha', 0.95)
                            self._ema_ce_ratio = alpha * self._ema_ce_ratio + (1 - alpha) * curr_ratio
                    else:
                        self._last_ce_ratio = float('nan')
        
        # Compute symmetric KL consistency loss - implement pairwise consistency
        cons_loss = torch.tensor(0.0, device=ce.device, dtype=ce.dtype)
        
        # Get current consistency weight (dynamic or static)
        current_step = getattr(self.state, 'global_step', 0)
        current_cons_weight = self.get_current_consistency_weight(current_step)
        
        if current_cons_weight > 0 and pair_ids is not None and self.model.training:
            N = logits.size(0)
            
            # Only proceed if we have an even number of samples
            if N % 2 == 0:
                # Check pair alignment - each pair should have the same ID
                pairs_aligned = True
                for i in range(0, N, 2):
                    if i+1 < N and pair_ids[i] != pair_ids[i+1]:
                        pairs_aligned = False
                        break
                
                if pairs_aligned:
                    # Get answer start positions more robustly
                    starts, ends = self._get_answer_ranges(labels)
                    
                    # For each sample, get the class token position (if available)
                    cls_positions = []
                    cls_ids_tensor = torch.tensor([self.cls_first_id, self.cls_second_id, self.cls_sim_id], 
                                            device=logits.device)
                    
                    for i in range(N):
                        # Check if we have a valid answer range
                        if ends[i] > starts[i]:
                            # Extract just the answer span
                            answer_logits = logits[i, starts[i]:ends[i]]
                            
                            # Find position with highest class activation (vectorized)
                            cls_logits = torch.stack([
                                answer_logits[:, self.cls_first_id],
                                answer_logits[:, self.cls_second_id],
                                answer_logits[:, self.cls_sim_id]
                            ], dim=-1)
                            
                            # Get max activation across class tokens per position
                            max_cls_activations, _ = cls_logits.max(dim=-1)
                            
                            # Find position with highest activation
                            if max_cls_activations.numel() > 0:
                                best_pos = max_cls_activations.argmax().item()
                                cls_positions.append(starts[i] + best_pos)
                            else:
                                cls_positions.append(starts[i])  # Default to first position
                        else:
                            cls_positions.append(starts[i])  # Default to first position
                    
                    # Convert to tensor
                    cls_positions = torch.tensor(cls_positions, device=logits.device)
                    
                    # Get class token logits for each sample
                    batch_indices = torch.arange(N, device=logits.device)
                    ans_logits = logits[batch_indices, cls_positions]  # (N, V)
                    
                    # Get logits for the three class tokens
                    cls_logits = ans_logits.index_select(-1, cls_ids_tensor)  # (N, 3)
                    
                    # Compute log probabilities for KL divergence
                    log_probs = torch.log_softmax(cls_logits, dim=-1)  # (N, 3)
                    probs = torch.exp(log_probs)  # (N, 3)
                    
                    # Split into original and swapped pairs
                    p = probs[0::2]  # Original pairs (A,B)
                    q = probs[1::2]  # Swapped pairs (B,A)
                    log_p = log_probs[0::2]
                    log_q = log_probs[1::2]
                    
                    # Create permutation indices: First<->Second, Similar stays
                    perm_indices = torch.tensor([1, 0, 2], device=probs.device)
                    
                    # Apply permutation to q (swap First and Second)
                    q_perm = q.index_select(-1, perm_indices)
                    log_q_perm = log_q.index_select(-1, perm_indices)
                    
                    # Compute symmetric KL divergence explicitly to avoid log_target pitfalls
                    # KL(p || q_perm) = sum_x p(x) * (log p(x) - log q_perm(x))
                    kl_pq = (p * (log_p - log_q_perm)).sum(dim=-1)

                    # KL(q_perm || p) = sum_x q_perm(x) * (log q_perm(x) - log p(x))
                    kl_qp = (q_perm * (log_q_perm - log_p)).sum(dim=-1)

                    # Symmetric KL
                    sym_kl = 0.5 * (kl_pq + kl_qp)
                    
                    # Compute final consistency loss (no debug printing)
                    cons_loss = sym_kl.mean()
                else:
                    # Pair IDs not aligned -> skip consistency loss silently
                    cons_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            else:
                # Odd batch size -> skip consistency loss silently
                cons_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        self._last_cons = float(cons_loss.item())
        
        # Combine losses with numerical stability check
        # If cons_loss is not finite, ignore it to avoid affecting training
        if torch.isfinite(cons_loss):
            loss = ce + float(current_cons_weight) * cons_loss
        else:
            loss = ce
            self._last_cons = 0.0
        
        # Enable gradient norm logging
        self._log_grad_norm = True
        
        # Calculate gradient norm manually for logging - only every few batches to reduce overhead
        grad_norm = 0.0
        current_step = getattr(self.state, 'global_step', 0)
        if hasattr(self, 'model') and self.model is not None and current_step % 5 == 0:
            try:
                total_norm = 0.0
                param_count = 0
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                if param_count > 0:
                    grad_norm = total_norm ** (1. / 2)
            except Exception:
                pass
        
        # Record exact loss values for logging
        self._last_total = float(loss.detach().cpu().item())
        self._last_grad_norm = grad_norm

        # Skip expensive tensor cleanup - let Python's garbage collector handle it
        # Manual tensor deletion is computationally expensive and unnecessary

        # Log detailed losses for training monitoring (total, weighted CE, consistency)
        # These will appear in Trainer logs and can be used by callbacks/console output.
        try:
            # Calculate weighted consistency loss for display
            current_step = getattr(self.state, 'global_step', 0)
            current_cons_weight = self.get_current_consistency_weight(current_step)
            weighted_cons_loss = float(self._last_cons) * float(current_cons_weight) if self._last_cons is not None else 0.0
            
            log_dict = {
                "loss": self._last_total,
                "ce_loss": float(self._last_ce) if self._last_ce is not None else float('nan'),
                "cons_loss": weighted_cons_loss,  # Show weighted consistency loss
                "cons_weight": float(current_cons_weight),  # Show current consistency weight
                "base_ce": float(self._last_base_ce) if self._last_base_ce is not None else float('nan'),
                "ce_weight": float(self._last_weight) if self._last_weight is not None else float('nan'),
                "grad_norm": float(self._last_grad_norm) if hasattr(self, '_last_grad_norm') else 0.0,
            }
            # Use Trainer.log to ensure consistent aggregation and reporting
            self.log(log_dict)
        except Exception:
            # Do not let logging failures interrupt training
            pass

        # Return combined loss. Training-time monitoring (per-sample generation and decode)
        # has been removed to keep training lightweight and ensure the original loss
        # reported by the Trainer is displayed.
        return (loss, outputs) if return_outputs else loss
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Hook called at the end of each epoch for memory cleanup.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control object
            **kwargs: Additional arguments
        """
        # Call parent method first
        super().on_epoch_end(args, state, control, **kwargs)
        
        # Memory cleanup at epoch end
        if self.enable_memory_cleanup and self.memory_manager is not None:
            epoch_num = getattr(state, 'epoch', 0)
            self.memory_manager.cleanup_epoch_end(int(epoch_num))
    
    def on_train_end(self, args, state, control, **kwargs):
        """
        Hook called at the end of training for final memory cleanup.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control object
            **kwargs: Additional arguments
        """
        # Call parent method first
        super().on_train_end(args, state, control, **kwargs)
        
        # Final memory cleanup
        if self.enable_memory_cleanup and self.memory_manager is not None:
            # Log final memory statistics
            summary = self.memory_manager.get_cleanup_summary()
            print(f"\n[MEMORY] Training Complete - Cleanup Summary:")
            print(f"  - Cleanup operations: {summary['cleanup_count']}")
            print(f"  - Peak memory usage: {summary['peak_memory_mb']:.1f}MB")
            print(f"  - Final memory usage: {summary['current_memory_mb']:.1f}MB")
            print(f"  - Memory trend: {summary['memory_trend']}")
            
            # Perform final cleanup
            self.memory_manager.perform_gpu_cleanup(aggressive=True)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        if self.enable_memory_cleanup and self.memory_manager is not None:
            return self.memory_manager.get_memory_stats()
        else:
            return {}
