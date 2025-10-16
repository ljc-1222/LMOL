# -*- coding: utf-8 -*-
"""
Training Health Monitoring Utilities

This module provides comprehensive monitoring capabilities for LMOL training:
- Per-class accuracy tracking
- Gradient norm monitoring
- Loss anomaly detection
- Early stopping implementation
- Training health reporting

Key Features:
- Real-time monitoring of training metrics
- Automatic detection of training issues
- Comprehensive logging and reporting
- Early stopping to prevent overfitting
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from collections import defaultdict, deque

from configs.config import config


class TrainingHealthMonitor:
    """
    Comprehensive training health monitoring system.
    
    This class tracks various training metrics and detects potential issues
    such as overfitting, gradient explosion, vanishing gradients, and loss anomalies.
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """
        Initialize the health monitor.
        
        Args:
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        
        # Tracking variables
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.epoch_losses = deque(maxlen=10)
        self.gradient_norms = deque(maxlen=100)
        self.per_class_accuracies = defaultdict(lambda: deque(maxlen=50))
        
        # Anomaly detection
        self.loss_anomaly_count = 0
        self.gradient_anomaly_count = 0
        
    def update_epoch_loss(self, loss: float) -> bool:
        """
        Update epoch loss and check for early stopping.
        
        Args:
            loss: Current epoch loss
            
        Returns:
            True if training should continue, False if early stopping triggered
        """
        self.epoch_losses.append(loss)
        
        # Check for improvement
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return self.patience_counter < self.patience
    
    def update_gradient_norm(self, grad_norm: float) -> bool:
        """
        Update gradient norm and check for anomalies.
        
        Args:
            grad_norm: Current gradient norm
            
        Returns:
            True if gradient norm is healthy, False if anomaly detected
        """
        self.gradient_norms.append(grad_norm)
        
        # Check for gradient explosion (silent)
        if grad_norm > config.GRADIENT_NORM_THRESHOLD:
            self.gradient_anomaly_count += 1
            return False
            
        # Check for vanishing gradients (silent)
        if grad_norm < config.GRADIENT_NORM_MIN_THRESHOLD:
            self.gradient_anomaly_count += 1
            return False
            
        return True
    
    def update_per_class_accuracy(self, class_accuracies: Dict[str, float]):
        """
        Update per-class accuracy tracking.
        
        Args:
            class_accuracies: Dictionary mapping class names to accuracies
        """
        for class_name, accuracy in class_accuracies.items():
            self.per_class_accuracies[class_name].append(accuracy)
    
    def check_loss_anomaly(self, loss: float) -> bool:
        """
        Check for loss anomalies (NaN, Inf, too small).
        
        Args:
            loss: Current loss value
            
        Returns:
            True if loss is healthy, False if anomaly detected
        """
        if math.isnan(loss):
            print(f"[LOSS_ERROR] Loss is NaN!")
            self.loss_anomaly_count += 1
            return False
            
        if math.isinf(loss):
            print(f"[LOSS_ERROR] Loss is Inf!")
            self.loss_anomaly_count += 1
            return False
            
        if loss < config.MIN_REASONABLE_LOSS:
            # Loss is suspiciously small (silent)
            self.loss_anomaly_count += 1
            return False
            
        return True
    
    def get_health_report(self) -> Dict[str, any]:
        """
        Generate comprehensive health report.
        
        Returns:
            Dictionary containing health metrics and recommendations
        """
        report = {
            'training_healthy': True,
            'issues': [],
            'recommendations': [],
            'metrics': {}
        }
        
        # Check gradient health
        if self.gradient_norms:
            avg_grad_norm = sum(self.gradient_norms) / len(self.gradient_norms)
            max_grad_norm = max(self.gradient_norms)
            min_grad_norm = min(self.gradient_norms)
            
            report['metrics']['gradient_norm'] = {
                'average': avg_grad_norm,
                'maximum': max_grad_norm,
                'minimum': min_grad_norm
            }
            
            if max_grad_norm > config.GRADIENT_NORM_THRESHOLD:
                report['issues'].append('Gradient explosion detected')
                report['recommendations'].append('Reduce learning rate')
                report['training_healthy'] = False
                
            if min_grad_norm < config.GRADIENT_NORM_MIN_THRESHOLD:
                report['issues'].append('Vanishing gradients detected')
                report['recommendations'].append('Increase learning rate or check model architecture')
                report['training_healthy'] = False
        
        # Check loss health
        if self.epoch_losses:
            recent_losses = list(self.epoch_losses)[-5:]  # Last 5 epochs
            avg_loss = sum(recent_losses) / len(recent_losses)
            loss_std = math.sqrt(sum((x - avg_loss) ** 2 for x in recent_losses) / len(recent_losses))
            
            report['metrics']['loss'] = {
                'average': avg_loss,
                'std': loss_std,
                'trend': 'stable' if loss_std < 0.1 else 'unstable'
            }
            
            if avg_loss < config.MIN_REASONABLE_LOSS:
                report['issues'].append('Loss is suspiciously small')
                report['recommendations'].append('Check for overfitting or numerical issues')
                report['training_healthy'] = False
        
        # Check per-class accuracy balance
        if self.per_class_accuracies:
            class_avg_accs = {}
            for class_name, accs in self.per_class_accuracies.items():
                if accs:
                    class_avg_accs[class_name] = sum(accs) / len(accs)
            
            report['metrics']['per_class_accuracy'] = class_avg_accs
            
            # Check for class imbalance in accuracy
            if len(class_avg_accs) >= 2:
                acc_values = list(class_avg_accs.values())
                max_acc = max(acc_values)
                min_acc = min(acc_values)
                acc_diff = max_acc - min_acc
                
                if acc_diff > 0.3:  # 30% difference
                    report['issues'].append('Significant class accuracy imbalance')
                    report['recommendations'].append('Check class weights and data balance')
                    report['training_healthy'] = False
        
        # Check anomaly counts
        if self.loss_anomaly_count > 0:
            report['issues'].append(f'{self.loss_anomaly_count} loss anomalies detected')
            report['training_healthy'] = False
            
        if self.gradient_anomaly_count > 0:
            report['issues'].append(f'{self.gradient_anomaly_count} gradient anomalies detected')
            report['training_healthy'] = False
        
        return report


class HealthMonitoringCallback(TrainerCallback):
    """
    Trainer callback for comprehensive health monitoring.
    
    This callback integrates with HuggingFace Trainer to provide
    real-time monitoring of training health metrics.
    """
    
    def __init__(self):
        """Initialize the health monitoring callback."""
        self.monitor = TrainingHealthMonitor(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA
        )
        self.step_count = 0
        
    def on_log(self, args: TrainingArguments, state: TrainerState, 
               control: TrainerControl, logs: Dict[str, float], **kwargs) -> TrainerControl:
        """
        Monitor training logs for health metrics.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            logs: Current training logs
            **kwargs: Additional arguments
            
        Returns:
            Updated training control
        """
        if not config.MONITOR_GRADIENT_NORMS and not config.MONITOR_LOSS_ANOMALIES:
            return control
            
        self.step_count += 1
        
        # Monitor gradient norms (silent)
        if config.MONITOR_GRADIENT_NORMS and 'grad_norm' in logs:
            grad_norm = logs['grad_norm']
            self.monitor.update_gradient_norm(grad_norm)
        
        # Monitor loss anomalies (silent)
        if config.MONITOR_LOSS_ANOMALIES and 'loss' in logs:
            loss = logs['loss']
            self.monitor.check_loss_anomaly(loss)
        
        # Monitor per-class accuracy
        if config.LOG_PER_CLASS_ACCURACY:
            class_accs = {}
            for key in ['train_acc_first', 'train_acc_second', 'train_acc_similar']:
                if key in logs:
                    class_name = key.replace('train_acc_', '')
                    class_accs[class_name] = logs[key]
            
            if class_accs:
                self.monitor.update_per_class_accuracy(class_accs)
        
        return control
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, **kwargs) -> TrainerControl:
        """
        Check for early stopping at epoch end.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            **kwargs: Additional arguments
            
        Returns:
            Updated training control
        """
        if not config.USE_EARLY_STOPPING:
            return control
            
        # Get current epoch loss
        if hasattr(state, 'log_history') and state.log_history:
            recent_logs = state.log_history[-1]
            if 'loss' in recent_logs:
                current_loss = recent_logs['loss']
                
                # Check for early stopping
                if not self.monitor.update_epoch_loss(current_loss):
                    print(f"[EARLY_STOPPING] Stopping training at epoch {state.epoch}")
                    print(f"[EARLY_STOPPING] Best loss: {self.monitor.best_loss:.6f}")
                    print(f"[EARLY_STOPPING] Current loss: {current_loss:.6f}")
                    control.should_training_stop = True
        
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, **kwargs) -> TrainerControl:
        """
        Generate final health report at training end.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            **kwargs: Additional arguments
            
        Returns:
            Updated training control
        """
        print("\n" + "="*60)
        print("TRAINING HEALTH REPORT")
        print("="*60)
        
        report = self.monitor.get_health_report()
        
        print(f"Training Healthy: {'✓' if report['training_healthy'] else '✗'}")
        
        if report['issues']:
            print("\nIssues Detected:")
            for issue in report['issues']:
                print(f"  - {issue}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print("\nMetrics:")
        for metric_name, metric_data in report['metrics'].items():
            print(f"  {metric_name}: {metric_data}")
        
        print("="*60)
        
        return control
