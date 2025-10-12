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
from typing import List

from configs.config import config


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
