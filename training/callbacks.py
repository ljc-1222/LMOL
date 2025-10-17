# -*- coding: utf-8 -*-
"""
LMOL Training Callbacks

This module provides training callbacks for the LMOL project:
- SaveBestTrainingLossCallback: Automatic model saving based on training loss
- Additional callbacks for monitoring and logging

Key Features:
- Desktop notifications for model saves
- Comprehensive metadata tracking
- Robust error handling for save operations
- Automatic best and last model saving
"""

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainerCallback

from utils.csv_logger import CSVTrainLogger


class SaveBestTrainingLossCallback(TrainerCallback):
    """
    Callback to save the best model based on training loss.
    
    This callback monitors training loss and automatically saves:
    - Best model: Saved to <fold_dir>/best when loss improves (any improvement)
    - Last model: Saved to <fold_dir>/last at the end of each epoch
    
    Features:
    - Desktop notifications for model saves (if notify-send available)
    - Comprehensive metadata tracking
    - Robust error handling for save operations
    """
    def __init__(self, best_dir: Path, processor, tokenizer, csv_logger: Optional[CSVTrainLogger] = None):
        """
        Initialize the callback.
        
        Args:
            best_dir: Directory to save best model
            processor: Image processor for saving
            tokenizer: Tokenizer for saving
            csv_logger: Optional CSV logger for training metrics
        """
        self.best_dir = best_dir
        self.fold_dir = best_dir.parent  # Parent directory of best_dir (the fold directory)
        self.best_loss = float("inf")
        self.last_saved_epoch = -1
        self._processor = processor
        self._tokenizer = tokenizer
        self._first_model_saved = False
        self.csv_logger = csv_logger
        self._last_log_time = time.time()

    def _notify(self, title: str, message: str) -> None:
        """
        Send desktop notification if available, otherwise print to console.
        
        Args:
            title: Notification title
            message: Notification message
        """
        try:
            if shutil.which("notify-send"):
                subprocess.run(["notify-send", title, message], check=False)
            else:
                print(f"[Notify] {title}: {message}")
        except Exception:
            print(f"[Notify] {title}: {message}")

    def _is_valid_loss(self, loss: float) -> bool:
        """
        Validate that a loss value is reasonable for model saving.
        
        Args:
            loss: Loss value to validate
            
        Returns:
            True if loss is valid for saving, False otherwise
        """
        # Check for invalid loss values
        if loss <= 0.0:
            return False
        
        # Check for unreasonably small loss (likely numerical error)
        if loss < 1e-6:
            return False
            
        # Check for NaN or infinite values
        if not (0 < loss < float('inf')):
            return False
            
        return True

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Handle training log events for model saving and CSV logging.
        
        Monitors training loss and saves models when:
        - First model (best model) - always save the first model
        - Loss improves (best model) - any improvement counts
        - At the end of each epoch (last model)
        
        Also logs training metrics to CSV if csv_logger is provided.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            logs: Log dictionary containing metrics
            **kwargs: Additional keyword arguments
        """
        # Suppress any print statements from the logs dict itself
        # (this prevents transformers' default behavior of printing logs)
        if logs is None or "loss" not in logs:
            return
        
        loss = float(logs["loss"])
        is_main = getattr(args, "local_rank", -1) in (-1, 0)
        
        if not is_main:
            return
        
        # Validate loss value - prevent saving models with invalid loss
        if not self._is_valid_loss(loss):
            print(f"[WARNING] Skipping model save due to invalid loss: {loss}")
            return
        
        # Calculate batch time for performance monitoring
        current_time = time.time()
        batch_time_ms = (current_time - self._last_log_time) * 1000
        self._last_log_time = current_time
        
        # Get memory usage if available
        memory_usage_mb = 0.0
        if torch.cuda.is_available():
            try:
                memory_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        
        # Use the epoch from logs if available (float progress), otherwise fallback to state.epoch
        current_epoch = logs.get('epoch', state.epoch) if logs else (state.epoch if state.epoch is not None else 0)
        
        # Log to CSV if logger is available
        if self.csv_logger is not None:
            # Prepare metrics for CSV logging
            csv_metrics = {
                'loss': logs.get('loss'),
                'ce_loss': logs.get('ce_loss'),
                'cons_loss': logs.get('cons_loss'),
                'cons_weight': logs.get('cons_weight'),
                'base_ce': logs.get('base_ce'),
                'ce_weight': logs.get('ce_weight'),
                'grad_norm': logs.get('grad_norm'),
                'learning_rate': logs.get('learning_rate'),
                'lr_projection': logs.get('lr_projection'),
                'lr_lora': logs.get('lr_lora'),
                'train_acc': logs.get('train_acc'),
                'train_acc_first': logs.get('train_acc_first'),
                'train_acc_second': logs.get('train_acc_second'),
                'train_acc_similar': logs.get('train_acc_similar'),
                'ce_ratio': logs.get('ce_ratio'),
                'ema_ce_ratio': logs.get('ema_ce_ratio'),
                'memory_usage_mb': memory_usage_mb,
                'batch_time_ms': batch_time_ms
            }
            
            # Log to CSV
            self.csv_logger.log_metrics(
                csv_metrics,
                step=state.global_step,
                epoch=current_epoch
            )
            
        # Check for best model after each batch
        if state.global_step > 0:
            batch_num = state.global_step
            
            # Save first model as best model (even if it's not the best)
            if not self._first_model_saved:
                self.best_loss = loss
                self._save_best(kwargs.get("model"), state)
                self._first_model_saved = True
            
            # Save best model if loss improves (any improvement)
            elif loss < self.best_loss:
                self.best_loss = loss
                self._save_best(kwargs.get("model"), state)
            
            # Save last model at the end of each epoch
            # Check if we've completed a full epoch (integer epoch boundary)
            epoch_int = int(current_epoch)
            if epoch_int > self.last_saved_epoch and epoch_int > 0:
                self._save_last(kwargs.get("model"), state)
                self.last_saved_epoch = epoch_int

    def _save_best(self, model, state):
        """
        Save the best model with comprehensive metadata.
        
        Args:
            model: Model to save
            state: Training state
        """
        # Get epoch progress from the latest log entry if available
        current_epoch = 0.0
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            current_epoch = state.log_history[-1].get('epoch', state.epoch or 0)
        else:
            current_epoch = state.epoch if state.epoch is not None else 0
        
        current_step = state.global_step
        
        # Detailed logging is now handled in the calling method
        
        self.best_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        try:
            model.save_pretrained(str(self.best_dir), safe_serialization=False)
        except Exception as e:
            torch.save(model.state_dict(), self.best_dir / "pytorch_model.bin")
        
        # Save processor and tokenizer
        for obj in (self._processor, self._tokenizer):
            try:
                obj.save_pretrained(self.best_dir)
            except Exception:
                pass
        
        # Save metadata - validate loss before saving
        if not self._is_valid_loss(self.best_loss):
            print(f"[ERROR] Attempted to save best model with invalid loss: {self.best_loss}")
            return
            
        metadata = {
            "best_loss": round(self.best_loss, 6),
            "global_step": state.global_step,
            "epoch": state.epoch,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        
        try:
            (self.best_dir / "best_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            pass
        
        # Send desktop notification
        self._notify(
            "New Best Model Saved!", 
            f"Loss: {self.best_loss:.6f} | Epoch: {current_epoch:.5f} | Step: {current_step:,}"
        )

    def _save_last(self, model, state):
        """
        Save the last model for checkpointing as last_{epoch}.
        
        Args:
            model: Model to save
            state: Training state
        """
        # Get current epoch as integer
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        last_dir = self.fold_dir / f"last_{current_epoch}"
        
        last_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(last_dir), safe_serialization=False)
        except Exception:
            torch.save(model.state_dict(), last_dir / "pytorch_model.bin")
        
        # Save processor and tokenizer
        for obj in (self._processor, self._tokenizer):
            try:
                obj.save_pretrained(last_dir)
            except Exception:
                pass
        
        # Get loss from log history and validate
        last_loss = float(state.log_history[-1].get("loss", 0)) if state.log_history else 0.0
        
        # Save metadata - validate loss before saving
        if not self._is_valid_loss(last_loss):
            print(f"[ERROR] Attempted to save last model with invalid loss: {last_loss}")
            return
            
        (last_dir / "last_meta.json").write_text(json.dumps({
            "loss": round(last_loss, 6),
            "global_step": state.global_step,
            "epoch": state.epoch,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
