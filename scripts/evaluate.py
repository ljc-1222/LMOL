# -*- coding: utf-8 -*-
"""
LMOL Evaluation Entry Point

This script provides the main entry point for LMOL evaluation.
It imports and runs the evaluation main function from the evaluation module.

Usage:
    python scripts/evaluate.py [options]

Options:
    --samples, -n: Number of pairs to evaluate per fold (default: 1000, 0 for all)
    --run-dir: Specific run directory to evaluate (default: latest)
    --model-type, -m: Model type to evaluate ('best', 'last', or 'fold')

The evaluation will:
- Load trained models from checkpoints
- Evaluate on test datasets
- Generate confusion matrices
- Report accuracy metrics
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Add the project root to the Python path to allow absolute imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from evaluation.main import evaluation_main
    evaluation_main()
