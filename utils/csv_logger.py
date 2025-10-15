# -*- coding: utf-8 -*-
"""
LMOL CSV Training Logger

This module provides CSV logging functionality for training metrics:
- Real-time logging of training metrics to CSV files
- Scientific notation formatting (.6e) for numerical values
- Automatic file management and rotation
- Integration with training callbacks

Key Features:
- Scientific notation formatting for all numerical values
- Automatic CSV header management
- Thread-safe logging operations
- Integration with training callbacks
"""

import csv
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


class CSVTrainLogger:
    """
    CSV logger for training metrics with scientific notation formatting.
    
    This logger writes training metrics to CSV files using scientific notation
    (.6e format) for all numerical values, ensuring consistent formatting
    and easy parsing for analysis tools.
    """
    
    def __init__(self, log_dir: Union[str, Path], filename: str = "training_log.csv"):
        """
        Initialize CSV training logger.
        
        Args:
            log_dir: Directory to save CSV log files
            filename: Name of the CSV log file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.log_path = self.log_dir / filename
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Track if header has been written
        self._header_written = False
        
        # Initialize CSV file with headers
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with standard training metric headers."""
        if not self.log_path.exists():
            # Define standard training metric headers
            headers = [
                'timestamp',
                'step',
                'epoch',
                'loss',
                'ce_loss',
                'cons_loss',
                'cons_weight',
                'base_ce',
                'ce_weight',
                'grad_norm',
                'learning_rate',
                'lr_projection',
                'lr_lora',
                'train_acc',
                'train_acc_first',
                'train_acc_second',
                'train_acc_similar',
                'ce_ratio',
                'ema_ce_ratio',
                'memory_usage_mb',
                'batch_time_ms'
            ]
            
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            self._header_written = True
    
    def _format_value(self, value: Any) -> str:
        """
        Format a value using scientific notation (.6e format) for numbers.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string value
        """
        if value is None:
            return ''
        elif isinstance(value, (int, float)):
            if isinstance(value, int):
                return f"{value:d}"
            else:
                # Use scientific notation for floats
                return f"{value:.6e}"
        else:
            return str(value)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, 
                   epoch: Optional[float] = None, timestamp: Optional[datetime] = None):
        """
        Log training metrics to CSV file.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step number
            epoch: Current epoch (can be fractional)
            timestamp: Timestamp for the log entry (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Prepare row data with all standard fields
        row_data = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],  # Include milliseconds
            'step': step,
            'epoch': epoch,
            'loss': metrics.get('loss'),
            'ce_loss': metrics.get('ce_loss'),
            'cons_loss': metrics.get('cons_loss'),
            'cons_weight': metrics.get('cons_weight'),
            'base_ce': metrics.get('base_ce'),
            'ce_weight': metrics.get('ce_weight'),
            'grad_norm': metrics.get('grad_norm'),
            'learning_rate': metrics.get('learning_rate'),
            'lr_projection': metrics.get('lr_projection'),
            'lr_lora': metrics.get('lr_lora'),
            'train_acc': metrics.get('train_acc'),
            'train_acc_first': metrics.get('train_acc_first'),
            'train_acc_second': metrics.get('train_acc_second'),
            'train_acc_similar': metrics.get('train_acc_similar'),
            'ce_ratio': metrics.get('ce_ratio'),
            'ema_ce_ratio': metrics.get('ema_ce_ratio'),
            'memory_usage_mb': metrics.get('memory_usage_mb'),
            'batch_time_ms': metrics.get('batch_time_ms')
        }
        
        # Thread-safe writing
        with self._lock:
            try:
                with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Format all values using scientific notation
                    formatted_row = [self._format_value(row_data[field]) for field in [
                        'timestamp', 'step', 'epoch', 'loss', 'ce_loss', 'cons_loss',
                        'cons_weight', 'base_ce', 'ce_weight', 'grad_norm', 'learning_rate',
                        'lr_projection', 'lr_lora', 'train_acc', 'train_acc_first',
                        'train_acc_second', 'train_acc_similar', 'ce_ratio', 'ema_ce_ratio',
                        'memory_usage_mb', 'batch_time_ms'
                    ]]
                    writer.writerow(formatted_row)
            except Exception as e:
                print(f"[CSV_LOG_ERROR] Failed to write metrics: {e}")
    
    def log_custom_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None,
                          epoch: Optional[float] = None, timestamp: Optional[datetime] = None):
        """
        Log custom metrics that may not be in the standard format.
        
        This method allows logging additional metrics beyond the standard training
        metrics, appending them as additional columns to the CSV.
        
        Args:
            metrics: Dictionary of custom metrics to log
            step: Training step number
            epoch: Current epoch (can be fractional)
            timestamp: Timestamp for the log entry (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Prepare basic row data
        row_data = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'step': step,
            'epoch': epoch
        }
        
        # Add custom metrics
        row_data.update(metrics)
        
        # Thread-safe writing
        with self._lock:
            try:
                # Check if we need to update headers for new fields
                if self.log_path.exists():
                    with open(self.log_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        existing_headers = next(reader, [])
                else:
                    existing_headers = []
                
                # Find new fields that need to be added to headers
                new_fields = [field for field in row_data.keys() if field not in existing_headers]
                
                if new_fields:
                    # Update headers
                    updated_headers = existing_headers + new_fields
                    # Read existing data
                    existing_data = []
                    if self.log_path.exists():
                        with open(self.log_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            existing_data = list(reader)
                    
                    # Write updated CSV with new headers
                    with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=updated_headers)
                        writer.writeheader()
                        for row in existing_data:
                            # Fill missing fields with empty strings
                            for field in new_fields:
                                if field not in row:
                                    row[field] = ''
                            writer.writerow(row)
                
                # Append new row
                with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Format all values using scientific notation
                    formatted_row = [self._format_value(row_data.get(field, '')) for field in 
                                   (existing_headers + new_fields)]
                    writer.writerow(formatted_row)
                    
            except Exception as e:
                print(f"[CSV_LOG_ERROR] Failed to write custom metrics: {e}")
    
    def get_log_path(self) -> Path:
        """Get the path to the CSV log file."""
        return self.log_path
    
    def close(self):
        """Close the logger (no-op for file-based logging)."""
        pass


def create_training_logger(output_dir: Union[str, Path], fold_name: str = "fold1") -> CSVTrainLogger:
    """
    Create a CSV training logger for a specific fold.
    
    Args:
        output_dir: Base output directory for the training run
        fold_name: Name of the fold (e.g., "fold1", "fold2")
        
    Returns:
        Configured CSVTrainLogger instance
    """
    log_dir = Path(output_dir) / fold_name
    return CSVTrainLogger(log_dir, "training_log.csv")


def create_timestamped_logger(base_dir: Union[str, Path], timestamp: Optional[str] = None) -> CSVTrainLogger:
    """
    Create a CSV training logger with timestamped directory.
    
    Args:
        base_dir: Base directory for logs
        timestamp: Timestamp string (defaults to current time)
        
    Returns:
        Configured CSVTrainLogger instance
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = Path(base_dir) / timestamp
    return CSVTrainLogger(log_dir, "training_log.csv")
