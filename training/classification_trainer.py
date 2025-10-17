# -*- coding: utf-8 -*-
"""
Classification-based Trainer for LMOL

This module implements a proper classification approach for LMOL training:
- Model only sees the prompt (question)
- Model learns to classify the answer (First/Second/Similar)
- No answer leakage during training
- Proper accuracy computation

Key Features:
- Classification head approach instead of causal LM
- No answer tokens in input sequence
- Proper accuracy measurement during training
- Prevents model from cheating by seeing the answer
"""

import math
import os
from typing import List, Dict, Any, Optional, Union

import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback

# Fix HuggingFace tokenizers parallelism warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.config import config
from utils.constants import IGNORE_INDEX
from utils.memory_manager import MemoryManager, get_memory_manager
from utils.gpu_diag import GPUDeviceManager, get_canonical_device
from utils.grad_diag import setup_gradient_diagnostics, assert_all_requires_grad
from .health_monitor import HealthMonitoringCallback




class LMOLClassificationTrainer(Trainer):
    """
    Classification-based trainer for LMOL.
    
    This trainer implements a proper classification approach:
    1. Model only sees the prompt (no answer tokens)
    2. Model learns to classify the answer
    3. Proper accuracy computation during training
    4. No answer leakage issues
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
        Initialize the LMOL Classification Trainer.
        
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
            'grad_norm': [],
            'train_acc': [],
            'class_acc_first': [],
            'class_acc_second': [],
            'class_acc_similar': []
        }
        
        # Track accumulation steps to know when to log
        self._accumulation_counter = 0
        
        # Track first batch to suppress verbose model output
        self._first_batch_complete = False
        
        # Track actual batch number for logging (independent of HuggingFace state)
        self._actual_batch_num = 0
        
        # Memory management configuration
        self.enable_memory_cleanup = enable_memory_cleanup
        self.memory_cleanup_frequency = memory_cleanup_frequency
        
        # Initialize memory manager
        if self.enable_memory_cleanup:
            self.memory_manager = MemoryManager(
                enable_monitoring=False,
                cleanup_frequency=memory_cleanup_frequency,
                aggressive_cleanup=False,
                log_memory_usage=False
            )
        else:
            self.memory_manager = None
        
        # Initialize GPU device manager for monitoring
        self.device_manager = GPUDeviceManager(target_device=0)
        self.device = self.device_manager.get_device()
        self.gpu_log_interval = getattr(config, 'GPU_LOG_INTERVAL', 100)
        
        # No classification head needed - use first token logits directly
        
        # Add health monitoring callback
        if config.MONITOR_GRADIENT_NORMS or config.MONITOR_LOSS_ANOMALIES or config.LOG_PER_CLASS_ACCURACY:
            self.add_callback(HealthMonitoringCallback())
        
        # Initialize gradient diagnostics
        self.grad_diagnostics = None
        self.amp_scaler = None
        self.amp_dtype = getattr(config, 'AMP_DTYPE', 'bf16')
        
        # Gradient clipping configuration
        self.grad_clip = getattr(config, 'GRAD_CLIP', 0.0)
        
        # Set up AMP scaler for fp16
        if self.amp_dtype == 'fp16' and torch.cuda.is_available():
            self.amp_scaler = torch.cuda.amp.GradScaler()
            print(f"[INFO] Initialized AMP scaler for fp16")
        elif self.amp_dtype == 'bf16':
            print(f"[INFO] Using bf16 AMP (no scaler needed)")
        
        # Set up gradient diagnostics (disabled to reduce console output)
        grad_log_interval = 0  # Disable gradient logging
        clip_grad_norm = getattr(config, 'GRADIENT_CLIP_NORM', 0.0)
        detect_anomaly = getattr(config, 'DETECT_ANOMALY', False)
        
        if hasattr(self, 'model') and self.model is not None:
            self.grad_diagnostics = setup_gradient_diagnostics(
                model=self.model,
                log_interval=grad_log_interval,
                clip_grad_norm=clip_grad_norm,
                detect_anomaly=detect_anomaly
            )
            print(f"[INFO] Initialized gradient diagnostics")
            
    
    
    @property
    def tokenizer(self):
        """Get the tokenizer from processing class."""
        return self.processing_class
    
    def setup_gradient_diagnostics(self):
        """Set up gradient diagnostics after model is available."""
        if self.grad_diagnostics is None and hasattr(self, 'model') and self.model is not None:
            grad_log_interval = 0  # Disable gradient logging
            clip_grad_norm = getattr(config, 'GRADIENT_CLIP_NORM', 0.0)
            detect_anomaly = getattr(config, 'DETECT_ANOMALY', False)
            
            self.grad_diagnostics = setup_gradient_diagnostics(
                model=self.model,
                log_interval=grad_log_interval,
                clip_grad_norm=clip_grad_norm,
                detect_anomaly=detect_anomaly
            )
            print(f"[INFO] Set up gradient diagnostics")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the classification loss for LMOL training using first token logits.
        
        Args:
            model: The LMOL model (no classification head needed)
            inputs: Batch inputs containing input_ids, attention_mask, pixel_values, label_ids, pair_ids
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (unused)
            
        Returns:
            Loss tensor (and optionally model outputs)
        """
        # Set up gradient diagnostics if not already done
        self.setup_gradient_diagnostics()
        
        # Verify device placement on first batch
        if not hasattr(self, '_device_verified'):
            try:
                self.device_manager.verify_training_setup(model, inputs)
                self._device_verified = True
                print(f"[INFO] Training setup verified: model and batch on {self.device}")
            except RuntimeError as e:
                print(f"[ERROR] Training setup verification failed: {e}")
                raise
        
        # GPU monitoring
        current_step = getattr(self.state, 'global_step', 0)
        if (self.gpu_log_interval > 0 and 
            current_step > 0 and 
            current_step % self.gpu_log_interval == 0 and 
            self.device.type == "cuda"):
            self.device_manager.log_memory_usage(current_step, "GPU")
        
        label_ids: Optional[torch.Tensor] = inputs.pop("label_ids", None)
        pair_ids: Optional[torch.Tensor] = inputs.pop("pair_ids", None)
        
        # Ensure model is in training mode
        model.train()
        
        # Get model outputs with proper AMP handling
        inputs_without_labels = {k: v for k, v in inputs.items() if k != 'labels'}
        
        # Use autocast for mixed precision if enabled
        if self.amp_dtype == 'bf16' and torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**inputs_without_labels)
        elif self.amp_dtype == 'fp16' and torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs_without_labels)
        else:
            outputs = model(**inputs_without_labels)
        
        # Use last token logits for classification (where model should predict answer)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Map to 3 classes: First, Second, Similar
        # We need to find the token IDs for these words in the vocabulary
        first_token_id = self.tokenizer.encode(config.ANSWER_FIRST, add_special_tokens=False)[0]
        second_token_id = self.tokenizer.encode(config.ANSWER_SECOND, add_special_tokens=False)[0]
        similar_token_id = self.tokenizer.encode(config.ANSWER_SIMILAR, add_special_tokens=False)[0]
        
        # Extract logits for the three answer tokens
        classification_logits = torch.stack([
            last_token_logits[:, first_token_id],   # First
            last_token_logits[:, second_token_id],  # Second  
            last_token_logits[:, similar_token_id]  # Similar
        ], dim=1)  # [batch_size, 3]
        
        # Debug information (only print once to avoid spam) - MUTED
        if not hasattr(self, '_debug_printed'):
            # print(f"[INFO] Classification logits shape: {classification_logits.shape}")
            # print(f"[INFO] Label IDs shape: {label_ids.shape}")
            # print(f"[INFO] First token ID: {first_token_id}, Second token ID: {second_token_id}, Similar token ID: {similar_token_id}")
            self._debug_printed = True
        
        # Create class weights
        class_weights = torch.ones(3, dtype=classification_logits.dtype, device=classification_logits.device)
        class_weights[2] = float(self.w_sim)  # Similar class gets WSIM weight
        
        # Apply swap weights to samples (not classes)
        sample_weights = torch.ones(classification_logits.size(0), dtype=classification_logits.dtype, device=classification_logits.device)
        if pair_ids is not None and config.SWAP_DOUBLE and self.swap_ce_weight != 1.0:
            # Determine which samples are swapped (odd indices in consecutive pairs)
            N = pair_ids.size(0)
            swap_mask = torch.zeros(N, dtype=torch.bool, device=pair_ids.device)
            
            if N >= 2:
                for i in range(0, N-1, 2):
                    if pair_ids[i] == pair_ids[i+1]:  # Same pair ID means they're a pair
                        swap_mask[i+1] = True  # Mark the second sample as swapped
                
                sample_weights[swap_mask] = self.swap_ce_weight
        
        # Validate inputs for loss computation
        if not torch.isfinite(classification_logits).all():
            raise ValueError("Classification logits contain NaN/Inf values")
        
        if not torch.all((label_ids >= 0) & (label_ids < 3)):
            raise ValueError(f"Invalid label_ids: {label_ids.tolist()}. Must be in range [0, 2]")
        
        if not torch.isfinite(class_weights).all():
            raise ValueError("Class weights contain NaN/Inf values")
        
        if not torch.isfinite(sample_weights).all():
            raise ValueError("Sample weights contain NaN/Inf values")
        
        # Compute cross-entropy loss with numerical stability
        ce_loss_per_sample = torch.nn.functional.cross_entropy(
            classification_logits,  # [batch_size, 3]
            label_ids,              # [batch_size,] with values 0, 1, or 2
            weight=class_weights,   # [3,] class weights
            reduction='none'        # Get per-sample loss
        )
        
        # Validate loss computation
        if not torch.isfinite(ce_loss_per_sample).all():
            raise ValueError("Cross-entropy loss contains NaN/Inf values")
        
        # Apply sample weights
        ce_loss = (ce_loss_per_sample * sample_weights).mean()
        
        # Final loss validation
        if not torch.isfinite(ce_loss):
            raise ValueError(f"Final CE loss is not finite: {ce_loss.item()}")
        
        if ce_loss.item() < 0:
            raise ValueError(f"Negative CE loss detected: {ce_loss.item()}")
        
        if ce_loss.item() > 100:  # Arbitrary large threshold
            print(f"[WARNING] Very large CE loss: {ce_loss.item():.2f}")
        
        # Store for logging (detach only for logging, not for loss computation)
        self._last_ce = float(ce_loss.detach().item())
        self._last_weight = float(sample_weights.mean().detach().item())
        
        # Compute training accuracy (now meaningful!)
        with torch.no_grad():
            preds = classification_logits.argmax(dim=-1)  # [batch_size,]
            correct = (preds == label_ids).sum().item()
            total = label_ids.size(0)
            train_acc = correct / total if total > 0 else 0.0
            self._last_train_acc = train_acc
            
            # Compute per-class accuracy
            from collections import defaultdict
            class_correct = defaultdict(int)
            class_total = defaultdict(int)
            for p, t in zip(preds.tolist(), label_ids.tolist()):
                class_total[t] += 1
                if p == t:
                    class_correct[t] += 1
            
            self._last_class_acc = {
                'first': class_correct[0] / max(class_total[0], 1),
                'second': class_correct[1] / max(class_total[1], 1),
                'similar': class_correct[2] / max(class_total[2], 1),
            }
        
        # Compute consistency loss (simplified for classification)
        cons_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
        
        # Get current consistency weight
        current_step = getattr(self.state, 'global_step', 0)
        current_cons_weight = self.get_current_consistency_weight(current_step)
        
        if current_cons_weight > 0 and pair_ids is not None and self.model.training:
            N = classification_logits.size(0)
            
            # Only proceed if we have an even number of samples
            if N % 2 == 0:
                # Check pair alignment
                pairs_aligned = True
                for i in range(0, N, 2):
                    if i+1 < N and pair_ids[i] != pair_ids[i+1]:
                        pairs_aligned = False
                        break
                
                # Debug: Print pair alignment status occasionally
                if not hasattr(self, '_pair_debug_printed') and not pairs_aligned:
                    print(f"[DEBUG] Pairs not aligned: pair_ids = {pair_ids.tolist()}")
                    self._pair_debug_printed = True
                
                if pairs_aligned:
                    # Get probabilities
                    probs = torch.softmax(classification_logits, dim=-1)  # [N, 3]
                    
                    # Split into original and swapped pairs
                    p = probs[0::2]  # Original pairs (A,B)
                    q = probs[1::2]  # Swapped pairs (B,A)
                    
                    # Create permutation indices: First<->Second, Similar stays
                    perm_indices = torch.tensor([1, 0, 2], device=probs.device)
                    q_perm = q.index_select(-1, perm_indices)
                    
                    # Compute symmetric KL divergence with numerical stability
                    eps = 1e-8
                    log_p = torch.log(torch.clamp(p, min=eps))  # Clamp to avoid log(0)
                    log_q_perm = torch.log(torch.clamp(q_perm, min=eps))
                    
                    kl_pq = (p * (log_p - log_q_perm)).sum(dim=-1)
                    kl_qp = (q_perm * (log_q_perm - log_p)).sum(dim=-1)
                    sym_kl = 0.5 * (kl_pq + kl_qp)
                    
                    # Debug: Print consistency loss details occasionally
                    if not hasattr(self, '_cons_debug_printed'):
                        print(f"[DEBUG] Consistency calculation: p.shape={p.shape}, q.shape={q.shape}")
                        print(f"[DEBUG] p[0]={p[0].tolist()}, q[0]={q[0].tolist()}, q_perm[0]={q_perm[0].tolist()}")
                        print(f"[DEBUG] sym_kl={sym_kl.tolist()}")
                        self._cons_debug_printed = True
                    
                    # Validate consistency loss
                    if not torch.isfinite(sym_kl).all():
                        print(f"[WARNING] Consistency loss contains NaN/Inf, skipping")
                        cons_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
                    else:
                        cons_loss = sym_kl.mean()
                        
                        # Additional validation
                        if not torch.isfinite(cons_loss):
                            print(f"[WARNING] Mean consistency loss is not finite, setting to 0")
                            cons_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
                        elif cons_loss.item() < 0:
                            print(f"[WARNING] Negative consistency loss: {cons_loss.item():.6f}")
                        elif cons_loss.item() > 10:  # Arbitrary threshold
                            print(f"[WARNING] Very large consistency loss: {cons_loss.item():.2f}")
        
        self._last_cons = float(cons_loss.detach().item())
        
        # Combine losses with validation
        if torch.isfinite(cons_loss) and current_cons_weight > 0:
            total_loss = ce_loss + float(current_cons_weight) * cons_loss
            
            # Validate combined loss
            if not torch.isfinite(total_loss):
                print(f"[WARNING] Combined loss is not finite, using CE loss only")
                loss = ce_loss
                self._last_cons = 0.0
            else:
                loss = total_loss
        else:
            loss = ce_loss
            self._last_cons = 0.0
        
        # Final loss validation
        if not torch.isfinite(loss):
            raise ValueError(f"Final combined loss is not finite: {loss.item()}")
        
        if loss.item() < 0:
            raise ValueError(f"Negative combined loss detected: {loss.item()}")
        
        if loss.item() > 1000:  # Very large threshold
            print(f"[WARNING] Extremely large combined loss: {loss.item():.2f}")
        
        # Gradient norm will be computed in optimizer_step after backward pass
        # Use previously computed value or initialize to 0.0
        if not hasattr(self, '_last_grad_norm'):
            self._last_grad_norm = 0.0
        # CRITICAL FIX: Don't detach loss before backward pass - this severs the gradient graph!
        # Store loss value for logging but keep the tensor for backward pass
        self._last_total = float(loss.detach().item())  # Only detach for logging, not for computation
        
        # Calculate weighted consistency loss for meaningful logging
        # Note: self._last_cons is the raw consistency loss, so we multiply by current_cons_weight
        weighted_cons_loss = self._last_cons * current_cons_weight if current_cons_weight > 0 else 0.0
        
        # Log metrics with explicit loss breakdown for debugging
        # Call our custom log method to ensure batch accuracy logging is displayed
        log_dict = {
            "loss": self._last_total,
            "ce_loss": self._last_ce,
            "cons_loss": weighted_cons_loss,  # Show weighted consistency loss, not raw
            "grad_norm": self._last_grad_norm,
            "train_acc": self._last_train_acc,
            "class_acc_first": self._last_class_acc.get('first', 0.0),
            "class_acc_second": self._last_class_acc.get('second', 0.0),
            "class_acc_similar": self._last_class_acc.get('similar', 0.0),
        }
        
        self.log(log_dict)
        
        return (loss, outputs) if return_outputs else loss
    
    # Remove custom training_step - let parent class handle backward pass
    # The issue was that we were detaching the loss before the parent could use it
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training step to ensure proper gradient handling and AMP.
        
        Args:
            model: The model to train
            inputs: Input batch
            num_items_in_batch: Number of items in batch (unused, for HuggingFace compatibility)
        """
        # Set up gradient diagnostics if not already done
        self.setup_gradient_diagnostics()
        
        
        # Zero gradients with proper method
        self.optimizer.zero_grad(set_to_none=True)
        
        # Compute loss
        loss = self.compute_loss(model, inputs)
        
        # Backward pass with AMP scaling
        if self.amp_scaler is not None:
            # FP16 with scaler
            self.amp_scaler.scale(loss).backward()
        else:
            # BF16 or no AMP - direct backward
            loss.backward()
        
        
        # Compute gradient norm and apply clipping
        if self.grad_diagnostics is not None:
            grad_norm = self.grad_diagnostics.apply_gradient_clipping(self.model)
        else:
            # Fallback gradient norm computation
            grad_norm = 0.0
            if hasattr(self, 'model') and self.model is not None:
                try:
                    total_norm = 0.0
                    param_count = 0
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param_norm = param.grad.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    
                    if param_count > 0:
                        grad_norm = total_norm ** (1. / 2)
                except (AttributeError, RuntimeError) as e:
                    grad_norm = 0.0
        
        # Apply additional gradient clipping if specified
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        
        # Store gradient norm for logging
        self._last_grad_norm = grad_norm
        
        # Note: Optimizer step is handled by the parent HuggingFace Trainer class
        # We don't call optimizer.step() here to avoid double-stepping
        
        # Gradient statistics logging disabled to reduce console output
        
        
        return loss.detach()
    
    def get_current_consistency_weight(self, step: int) -> float:
        """Calculate current consistency weight based on training progress."""
        if not self.use_dynamic_consistency or self.total_steps is None:
            return self.cons_weight
        
        progress = min(step / (self.total_steps * self.cons_weight_ramp_ratio), 1.0)
        current_weight = self.cons_weight_start + (self.cons_weight_end - self.cons_weight_start) * progress
        return current_weight
    
    def cleanup(self):
        """Clean up resources."""
        if self.grad_diagnostics is not None:
            self.grad_diagnostics.cleanup()
            self.grad_diagnostics = None
        
    
    def log(self, logs: Dict[str, float], step: Optional[int] = None) -> None:
        """Custom logging method for clean, minimal training output with DDP support."""
        # Filter out unwanted keys
        filtered_logs = {}
        for key in ['loss', 'ce_loss', 'cons_loss', 'grad_norm', 'train_acc', 'class_acc_first', 'class_acc_second', 'class_acc_similar']:
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
        
        # Increment our own batch counter (independent of HuggingFace state)
        self._actual_batch_num += 1
        effective_batch_num = self._actual_batch_num
        
        # Compute averaged metrics - prevent empty buffer issues
        if not self.batch_metrics_buffer['loss']:
            print(f"[WARNING] Empty loss buffer detected at step {effective_batch_num}, skipping logging")
            return
            
        avg_loss = sum(self.batch_metrics_buffer['loss']) / len(self.batch_metrics_buffer['loss'])
        avg_ce_loss = sum(self.batch_metrics_buffer['ce_loss']) / len(self.batch_metrics_buffer['ce_loss']) if self.batch_metrics_buffer['ce_loss'] else 0.0
        avg_cons_loss = sum(self.batch_metrics_buffer['cons_loss']) / len(self.batch_metrics_buffer['cons_loss']) if self.batch_metrics_buffer['cons_loss'] else 0.0
        avg_grad_norm = self.batch_metrics_buffer['grad_norm'][-1] if self.batch_metrics_buffer['grad_norm'] else 0.0
        # FIXED: Average accuracy across all gradient accumulation steps, not just take the last one
        train_acc = sum(self.batch_metrics_buffer['train_acc']) / len(self.batch_metrics_buffer['train_acc']) if self.batch_metrics_buffer.get('train_acc') else 0.0
        
        # FIXED: Average class accuracies across all gradient accumulation steps
        class_acc_first = sum(self.batch_metrics_buffer['class_acc_first']) / len(self.batch_metrics_buffer['class_acc_first']) if self.batch_metrics_buffer.get('class_acc_first') else 0.0
        class_acc_second = sum(self.batch_metrics_buffer['class_acc_second']) / len(self.batch_metrics_buffer['class_acc_second']) if self.batch_metrics_buffer.get('class_acc_second') else 0.0
        class_acc_similar = sum(self.batch_metrics_buffer['class_acc_similar']) / len(self.batch_metrics_buffer['class_acc_similar']) if self.batch_metrics_buffer.get('class_acc_similar') else 0.0
        
        # Clear buffers for next batch
        for key in self.batch_metrics_buffer:
            self.batch_metrics_buffer[key].clear()
        
        # Get learning rates
        lr_proj = 0.0
        lr_lora = 0.0
        
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            try:
                if len(self.optimizer.param_groups) >= 1:
                    lr_proj = self.optimizer.param_groups[0]['lr']
                if len(self.optimizer.param_groups) >= 2:
                    lr_lora = self.optimizer.param_groups[1]['lr']
            except (AttributeError, KeyError, IndexError) as e:
                # Fallback to logs if optimizer access fails
                lr_lora = logs.get('learning_rate', 0.0)
        
        # Calculate epoch progress
        if hasattr(self.state, 'epoch') and self.state.epoch is not None:
            base_epoch = int(self.state.epoch)
            steps_per_epoch = len(self.get_train_dataloader())
            if steps_per_epoch > 0:
                fractional_progress = (effective_batch_num % steps_per_epoch) / steps_per_epoch
                epoch_progress = base_epoch + fractional_progress
            else:
                epoch_progress = float(base_epoch)
        else:
            epoch_progress = 0.0
        
        # Enhanced logging with per-class accuracy
        if True:  # Always print in single-process mode
            # Use averaged class accuracies instead of last step only
            class_acc_str = f" | Class Acc: F={class_acc_first:.3f}, S={class_acc_second:.3f}, Sim={class_acc_similar:.3f}"
            
            print(f"Batch {effective_batch_num:4d} | Epoch {epoch_progress:.5f} | "
                  f"Loss: {avg_loss:.3e} (CE: {avg_ce_loss:.3e}, WCons: {avg_cons_loss:.3e}) | "
                  f"Acc: {train_acc:.4f}{class_acc_str} | "
                  f"LR: Proj = {lr_proj:.2e}, LoRA = {lr_lora:.2e} | Grad: {avg_grad_norm:.2e}")
        
        # Prepare callback logs with all available metrics
        callback_logs = {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'cons_loss': avg_cons_loss,
            'grad_norm': avg_grad_norm,
            'epoch': epoch_progress,
            'learning_rate': lr_lora,
            'lr_projection': lr_proj,
            'lr_lora': lr_lora,
            'train_acc': train_acc,
            'train_acc_first': class_acc_first,
            'train_acc_second': class_acc_second,
            'train_acc_similar': class_acc_similar,
            'step': effective_batch_num,
        }
        
        # Update state and trigger callbacks
        if hasattr(self, 'state'):
            self.state.log_history.append(callback_logs)
        
        # Call callback handler
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
                    else:
                        # Allow other output (like our callback notifications)
                        self.original_stdout.write(text)
                
                def flush(self):
                    self.original_stdout.flush()
            
            # Temporarily replace stdout
            old_stdout = sys.stdout
            sys.stdout = FilteredStdout(old_stdout)
            try:
                self.control = self.callback_handler.on_log(
                    self.args, self.state, self.control, callback_logs
                )
            finally:
                # Restore original stdout
                sys.stdout = old_stdout
