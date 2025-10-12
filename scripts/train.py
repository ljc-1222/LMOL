# -*- coding: utf-8 -*-
"""
LMOL Training Entry Point

This script provides the main entry point for LMOL training.
It imports and runs the training main function from the training module.

Usage:
    python scripts/train.py

The training will:
- Load configuration from configs/config.py
- Create timestamped output directory
- Train 5-fold cross-validation models
- Save best and last models for each fold
- Generate comprehensive training logs and metadata
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Add the project root to the Python path to allow absolute imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from training.main import training_main
    training_main()
