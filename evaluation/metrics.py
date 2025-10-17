# -*- coding: utf-8 -*-
"""
LMOL Evaluation Metrics

This module provides evaluation metrics and visualization for the LMOL project:
- Confusion matrix plotting and saving
- Additional evaluation metrics
- Visualization utilities

Key Features:
- Confusion matrix generation and saving
- Comprehensive metric calculation
- Visualization utilities for evaluation results
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from configs.config import config


def calculate_per_class_accuracy(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Calculate per-class accuracy for evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary mapping class names to their accuracies
    """
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1
    
    # Calculate accuracy for each class
    class_accuracies = {}
    for class_name in [config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR]:
        total = class_total.get(class_name, 0)
        correct = class_correct.get(class_name, 0)
        accuracy = correct / total if total > 0 else 0.0
        class_accuracies[class_name] = accuracy
    
    return class_accuracies


def plot_and_save_cm(y_true: List[str], y_pred: List[str], out_path: Path, title: str = "Confusion Matrix"):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        out_path: Output path for the plot
        title: Title for the plot
    """
    # Include all three possible answers to avoid missing-label issues
    labels = [config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
