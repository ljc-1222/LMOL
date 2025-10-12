# -*- coding: utf-8 -*-
"""
LMOL Memory Management Utilities

This module provides comprehensive memory management utilities for efficient
GPU memory usage during training while maintaining training correctness.

Key Features:
- Intelligent memory cleanup at different training phases
- Memory usage monitoring and logging
- Safe tensor cleanup that preserves gradients
- Configurable cleanup strategies
- Memory leak detection and prevention
"""

import gc
import torch
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
import psutil
import os


class MemoryManager:
    """
    Intelligent memory manager for LMOL training.
    
    Provides comprehensive memory management with different cleanup strategies
    for various training phases while ensuring training correctness.
    """
    
    def __init__(self, 
                 enable_monitoring: bool = True,
                 cleanup_frequency: int = 10,
                 aggressive_cleanup: bool = False,
                 log_memory_usage: bool = True):
        """
        Initialize the memory manager.
        
        Args:
            enable_monitoring: Enable memory usage monitoring
            cleanup_frequency: How often to perform cleanup (every N batches)
            aggressive_cleanup: Use more aggressive cleanup strategies
            log_memory_usage: Log memory usage statistics
        """
        self.enable_monitoring = enable_monitoring
        self.cleanup_frequency = cleanup_frequency
        self.aggressive_cleanup = aggressive_cleanup
        self.log_memory_usage = log_memory_usage
        
        # Track cleanup statistics
        self.cleanup_count = 0
        self.last_memory_usage = 0
        self.peak_memory_usage = 0
        
        # Memory usage history for trend analysis
        self.memory_history: List[float] = []
        
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        stats = {}
        
        # GPU memory stats
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'gpu_max_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
            })
        
        # CPU memory stats
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        stats.update({
            'cpu_rss_mb': memory_info.rss / 1024**2,
            'cpu_vms_mb': memory_info.vms / 1024**2,
        })
        
        return stats
    
    def log_memory_stats(self, phase: str = "unknown") -> None:
        """
        Log current memory statistics.
        
        Args:
            phase: Current training phase for context
        """
        if not self.log_memory_usage:
            return
            
        stats = self.get_memory_stats()
        
        # Update peak memory tracking
        current_gpu_memory = stats.get('gpu_allocated_mb', 0)
        if current_gpu_memory > self.peak_memory_usage:
            self.peak_memory_usage = current_gpu_memory
        
        # Store in history for trend analysis
        self.memory_history.append(current_gpu_memory)
        if len(self.memory_history) > 100:  # Keep last 100 measurements
            self.memory_history.pop(0)
        
        # Memory monitoring disabled to reduce console output
        # print(f"[MEMORY] {phase}: GPU={current_gpu_memory:.1f}MB "
        #       f"(Peak={self.peak_memory_usage:.1f}MB) "
        #       f"CPU={stats['cpu_rss_mb']:.1f}MB")
    
    def safe_cleanup_tensors(self, 
                           tensors: Union[List[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor],
                           preserve_gradients: bool = True) -> None:
        """
        Safely clean up tensors while preserving gradients if needed.
        
        Args:
            tensors: Tensors to clean up (single tensor, list, or dict)
            preserve_gradients: Whether to preserve gradient information
        """
        if tensors is None:
            return
            
        # Handle different input types
        if isinstance(tensors, torch.Tensor):
            tensor_list = [tensors]
        elif isinstance(tensors, dict):
            tensor_list = list(tensors.values())
        elif isinstance(tensors, list):
            tensor_list = tensors
        else:
            return
        
        for tensor in tensor_list:
            if tensor is None:
                continue
                
            # Only clean up if tensor doesn't require gradients or we're not preserving them
            if not preserve_gradients or not tensor.requires_grad:
                try:
                    del tensor
                except Exception:
                    pass
    
    def cleanup_intermediate_tensors(self, **kwargs) -> None:
        """
        Lightweight cleanup of intermediate tensors.
        
        Args:
            **kwargs: Tensors to clean up
        """
        # Only clean up tensors that are explicitly passed and safe to delete
        safe_tensors = []
        for name, tensor in kwargs.items():
            if tensor is not None and not tensor.requires_grad:
                safe_tensors.append(tensor)
        
        # Simple cleanup without complex logic
        for tensor in safe_tensors:
            try:
                del tensor
            except Exception:
                pass
    
    def cleanup_consistency_tensors(self, **kwargs) -> None:
        """
        Lightweight cleanup of consistency loss tensors.
        
        Args:
            **kwargs: Tensors to clean up
        """
        # Only clean up tensors that are explicitly passed and safe to delete
        safe_tensors = []
        for name, tensor in kwargs.items():
            if tensor is not None and not tensor.requires_grad:
                safe_tensors.append(tensor)
        
        # Simple cleanup without complex logic
        for tensor in safe_tensors:
            try:
                del tensor
            except Exception:
                pass
    
    def perform_gpu_cleanup(self, aggressive: bool = None) -> None:
        """
        Perform GPU memory cleanup.
        
        Args:
            aggressive: Override aggressive_cleanup setting
        """
        if not torch.cuda.is_available():
            return
            
        aggressive = aggressive if aggressive is not None else self.aggressive_cleanup
        
        # Standard cleanup
        torch.cuda.empty_cache()
        
        if aggressive:
            # More aggressive cleanup
            torch.cuda.ipc_collect()
            gc.collect()
    
    def should_cleanup(self, batch_num: int) -> bool:
        """
        Determine if cleanup should be performed based on batch number.
        
        Args:
            batch_num: Current batch number
            
        Returns:
            True if cleanup should be performed
        """
        return batch_num % self.cleanup_frequency == 0
    
    def cleanup_batch_end(self, batch_num: int, phase: str = "batch_end") -> None:
        """
        Perform cleanup at the end of a batch.
        
        Args:
            batch_num: Current batch number
            phase: Current training phase
        """
        if self.should_cleanup(batch_num):
            self.perform_gpu_cleanup()
            self.cleanup_count += 1
            
            if self.enable_monitoring:
                self.log_memory_stats(f"{phase}_cleanup")
    
    def cleanup_epoch_end(self, epoch_num: int) -> None:
        """
        Perform cleanup at the end of an epoch.
        
        Args:
            epoch_num: Current epoch number
        """
        # Always perform cleanup at epoch end
        self.perform_gpu_cleanup(aggressive=True)
        
        if self.enable_monitoring:
            self.log_memory_stats(f"epoch_{epoch_num}_end")
    
    def cleanup_fold_end(self, fold_num: int) -> None:
        """
        Perform cleanup at the end of a fold.
        
        Args:
            fold_num: Current fold number
        """
        # Most aggressive cleanup between folds
        self.perform_gpu_cleanup(aggressive=True)
        gc.collect()
        
        if self.enable_monitoring:
            self.log_memory_stats(f"fold_{fold_num}_end")
    
    @contextmanager
    def memory_context(self, context_name: str = "operation"):
        """
        Context manager for memory monitoring during operations.
        
        Args:
            context_name: Name of the operation being monitored
        """
        if self.enable_monitoring:
            initial_stats = self.get_memory_stats()
            initial_gpu = initial_stats.get('gpu_allocated_mb', 0)
        
        try:
            yield self
        finally:
            if self.enable_monitoring:
                final_stats = self.get_memory_stats()
                final_gpu = final_stats.get('gpu_allocated_mb', 0)
                memory_delta = final_gpu - initial_gpu
                
                # Memory monitoring disabled to reduce console output
                # if abs(memory_delta) > 10:  # Only log significant changes
                #     print(f"[MEMORY] {context_name}: Δ={memory_delta:+.1f}MB "
                #           f"(Final={final_gpu:.1f}MB)")
    
    def get_cleanup_summary(self) -> Dict[str, Any]:
        """
        Get summary of cleanup operations and memory usage.
        
        Returns:
            Dictionary with cleanup statistics
        """
        current_stats = self.get_memory_stats()
        
        return {
            'cleanup_count': self.cleanup_count,
            'peak_memory_mb': self.peak_memory_usage,
            'current_memory_mb': current_stats.get('gpu_allocated_mb', 0),
            'memory_trend': self._calculate_memory_trend(),
            'cleanup_frequency': self.cleanup_frequency,
            'aggressive_cleanup': self.aggressive_cleanup,
        }
    
    def _calculate_memory_trend(self) -> str:
        """
        Calculate memory usage trend from recent history.
        
        Returns:
            String describing the memory trend
        """
        if len(self.memory_history) < 10:
            return "insufficient_data"
        
        recent_history = self.memory_history[-10:]
        if len(recent_history) < 2:
            return "stable"
        
        # Simple trend calculation
        first_half = sum(recent_history[:5]) / 5
        second_half = sum(recent_history[5:]) / 5
        
        if second_half > first_half * 1.05:
            return "increasing"
        elif second_half < first_half * 0.95:
            return "decreasing"
        else:
            return "stable"


# Global memory manager instance
memory_manager = MemoryManager()


def get_memory_manager() -> MemoryManager:
    """
    Get the global memory manager instance.
    
    Returns:
        Global MemoryManager instance
    """
    return memory_manager


def cleanup_model_memory(model: torch.nn.Module) -> None:
    """
    Comprehensive model memory cleanup.
    
    Args:
        model: Model to clean up
    """
    try:
        # Move model to CPU
        model.to('cpu')
    except Exception:
        pass
    
    try:
        # Delete model
        del model
    except Exception:
        pass
    
    # Force garbage collection
    gc.collect()
    
    # Clean GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
