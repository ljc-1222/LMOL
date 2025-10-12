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
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainerCallback


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
    def __init__(self, best_dir: Path, processor, tokenizer):
        """
        Initialize the callback.
        
        Args:
            best_dir: Directory to save best model
            processor: Image processor for saving
            tokenizer: Tokenizer for saving
        """
        self.best_dir = best_dir
        self.last_dir = best_dir.parent / "last"
        self.best_loss = float("inf")
        self.last_saved_epoch = -1
        self._processor = processor
        self._tokenizer = tokenizer
        self._first_model_saved = False

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

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Handle training log events for model saving.
        
        Monitors training loss and saves models when:
        - First model (best model) - always save the first model
        - Loss improves (best model) - any improvement counts
        - At the end of each epoch (last model)
        
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
            
        # Check for best model after each batch
        if state.global_step > 0:
            batch_num = state.global_step
            # Use the epoch from logs if available (float progress), otherwise fallback to state.epoch
            current_epoch = logs.get('epoch', state.epoch) if logs else (state.epoch if state.epoch is not None else 0)
            
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
        
        # Save metadata
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
        Save the last model for checkpointing.
        
        Args:
            model: Model to save
            state: Training state
        """
        self.last_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(self.last_dir), safe_serialization=False)
        except Exception:
            torch.save(model.state_dict(), self.last_dir / "pytorch_model.bin")
        
        # Save processor and tokenizer
        for obj in (self._processor, self._tokenizer):
            try:
                obj.save_pretrained(self.last_dir)
            except Exception:
                pass
        
        # Save metadata
        (self.last_dir / "last_meta.json").write_text(json.dumps({
            "loss": round(float(state.log_history[-1].get("loss", 0)), 6),
            "global_step": state.global_step,
            "epoch": state.epoch,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
