# -*- coding: utf-8 -*-
"""
LMOL Data Generation Entry Point

This script provides the main entry point for LMOL data generation.
It imports and runs the data generation pipeline.

Usage:
    python scripts/generate_data.py

The data generation will:
- Load labels from config.LABELS_PATH
- Generate balanced tri-class pairs
- Create 5-fold cross-validation splits
- Generate training and evaluation datasets
- Save CSV files for training
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Add the project root to the Python path to allow absolute imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from data.generator import main
    main()
