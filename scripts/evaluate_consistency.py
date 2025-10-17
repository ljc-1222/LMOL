#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMOL Consistency Evaluation Script

This script evaluates model consistency by testing both (P1, P2, Question) and (P2, P1, Question)
for each sample and measuring the consistency between predictions.

Usage:
    python scripts/evaluate_consistency.py --samples 1000 --model-type best
    python scripts/evaluate_consistency.py --samples 1000 --model-type last --run-dir /path/to/run
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from configs.config import config
from evaluation.consistency_evaluator import (
    load_single_fold_consistency, 
    evaluate_consistency_fold, 
    free_model_consistency
)
from evaluation.metrics import plot_and_save_cm


def _latest_run_dir(root: Path) -> Path:
    """Find the latest run directory."""
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    cand = [p for p in root.iterdir() if p.is_dir()]
    if not cand:
        raise FileNotFoundError(f"No run directories inside {root}")
    cand.sort()
    return cand[-1]


def get_eval_csv_for_fold(fold_idx: int) -> Path:
    """Get evaluation CSV path for a specific fold."""
    try:
        rel = config.EVAL_PAIRS_CSVS[fold_idx - 1]
    except IndexError:
        raise IndexError(f"No eval CSV configured for fold {fold_idx}")
    path = Path(__file__).resolve().parent.parent / rel
    if not path.exists():
        raise FileNotFoundError(f"Eval CSV not found for fold {fold_idx}: {path}")
    return path


def consistency_evaluation_main():
    """Main consistency evaluation function."""
    print(f"[INFO] Running consistency evaluation with PyTorch's default single-process handling")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate model consistency on beauty comparison task")
    parser.add_argument("--samples", "-n", type=int, default=1000, 
                      help=f"Number of pairs to evaluate per fold (default: 1000)")
    parser.add_argument("--run-dir", type=str, default=None,
                      help="Specific run directory to evaluate (default: latest)")
    parser.add_argument("--model-type", "-m", type=str, choices=["best", "last", "fold"], default="best",
                      help="Model type to evaluate: 'best', 'last', or 'fold' (default: best)")
    args = parser.parse_args()
    
    run_start = time.perf_counter()
    print(f"[Time] Script start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup checkpoint root
    ckpt_root = Path(__file__).resolve().parent.parent / config.OUTPUT_DIR
    
    # Allow specifying a custom run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        print(f"Using specified run directory: {run_dir}")
    else:
        run_dir = _latest_run_dir(ckpt_root)
        print(f"Using latest run directory: {run_dir}")

    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('fold')])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories in {run_dir}")

    # Display evaluation configuration
    n_samples = args.samples
    print(f"[Config] Consistency evaluation with {n_samples} pairs per fold")
    print(f"[Config] Model type: {args.model_type}")
    print(f"[Config] Run directory: {run_dir}")

    # Track overall results
    grand_total_samples = 0
    grand_consistent_samples = 0
    grand_overall_correct = 0
    grand_consistent_correct = 0
    grand_y_true = []
    grand_y_pred_forward = []
    grand_y_pred_reverse = []
    grand_is_consistent = []

    # Evaluate each fold
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        fold_idx = int(fold_name.replace('fold', ''))
        
        print(f"\n[Fold] Processing {fold_name}...")
        
        try:
            # Get evaluation CSV for this fold
            eval_csv = get_eval_csv_for_fold(fold_idx)
            print(f"[Data] Loading evaluation data from {eval_csv}")
            
            # Load pair records
            from data import read_pairs_csv
            pair_records = read_pairs_csv(eval_csv)
            if n_samples > 0:
                pair_records = pair_records[:n_samples]
            
            print(f"[Data] Loaded {len(pair_records)} pairs for evaluation")
            
            # Load model and tokenizer
            print(f"[Model] Loading {args.model_type} model for {fold_name}...")
            model, tokenizer, device = load_single_fold_consistency(run_dir, fold_name, args.model_type)
            
            # Run consistency evaluation
            result = evaluate_consistency_fold(model, tokenizer, pair_records, fold_name, device)
            
            # Accumulate results
            grand_total_samples += result.total_samples
            grand_consistent_samples += result.consistent_samples
            grand_overall_correct += result.overall_correct
            grand_consistent_correct += result.consistent_correct
            grand_y_true.extend(result.y_true)
            grand_y_pred_forward.extend(result.y_pred_forward)
            grand_y_pred_reverse.extend(result.y_pred_reverse)
            grand_is_consistent.extend(result.is_consistent)
            
            # Save confusion matrix for this fold
            try:
                from evaluation.metrics import calculate_per_class_accuracy
                class_accuracies = calculate_per_class_accuracy(result.y_true, result.y_pred_forward)
                
                # Determine the model folder based on model_type
                if args.model_type in ("best", "last"):
                    model_folder = fold_dir / args.model_type
                else:  # "fold" - use fold directory directly
                    model_folder = fold_dir
                
                # Ensure the model folder exists
                model_folder.mkdir(parents=True, exist_ok=True)
                
                # Save confusion matrix for forward predictions
                out_cm_forward = model_folder / f"confusion_matrix_consistency_forward_{args.model_type}.png"
                plot_and_save_cm(
                    result.y_true, 
                    result.y_pred_forward, 
                    out_cm_forward, 
                    title=f"{fold_name} Consistency Evaluation - Forward Predictions ({args.model_type})"
                )
                print(f"[Plot] Saved forward confusion matrix to {out_cm_forward}")
                
                # Save confusion matrix for reverse predictions
                out_cm_reverse = model_folder / f"confusion_matrix_consistency_reverse_{args.model_type}.png"
                plot_and_save_cm(
                    result.y_true, 
                    result.y_pred_reverse, 
                    out_cm_reverse, 
                    title=f"{fold_name} Consistency Evaluation - Reverse Predictions ({args.model_type})"
                )
                print(f"[Plot] Saved reverse confusion matrix to {out_cm_reverse}")
                
            except Exception as e:
                print(f"[Warn] Failed to plot/save confusion matrix for {fold_name}: {e}")
            
            # Free model resources
            free_model_consistency(model, tokenizer)
            
        except Exception as e:
            print(f"[Error] Failed to evaluate {fold_name}: {e}")
            continue

    # Calculate overall results
    if grand_total_samples > 0:
        overall_accuracy = grand_overall_correct / grand_total_samples
        consistent_accuracy = grand_consistent_correct / grand_consistent_samples if grand_consistent_samples > 0 else 0.0
        consistency_rate = grand_consistent_samples / grand_total_samples
        
        print(f"\n[Overall Results]")
        print(f"  Total samples: {grand_total_samples}")
        print(f"  Consistent samples: {grand_consistent_samples} ({consistency_rate:.3f})")
        print(f"  Overall accuracy: {overall_accuracy:.6f} ({grand_overall_correct}/{grand_total_samples})")
        print(f"  Consistent accuracy: {consistent_accuracy:.6f} ({grand_consistent_correct}/{grand_consistent_samples})")
        
        # Save overall confusion matrix
        try:
            overall_cm_forward = run_dir / f"confusion_matrix_consistency_overall_forward_{args.model_type}.png"
            plot_and_save_cm(
                grand_y_true, 
                grand_y_pred_forward, 
                overall_cm_forward, 
                title=f"Overall Consistency Evaluation - Forward Predictions ({args.model_type})"
            )
            print(f"[Plot] Saved overall forward confusion matrix to {overall_cm_forward}")
            
            overall_cm_reverse = run_dir / f"confusion_matrix_consistency_overall_reverse_{args.model_type}.png"
            plot_and_save_cm(
                grand_y_true, 
                grand_y_pred_reverse, 
                overall_cm_reverse, 
                title=f"Overall Consistency Evaluation - Reverse Predictions ({args.model_type})"
            )
            print(f"[Plot] Saved overall reverse confusion matrix to {overall_cm_reverse}")
            
        except Exception as e:
            print(f"[Warn] Failed to plot/save overall confusion matrix: {e}")
    
    total_time = time.perf_counter() - run_start
    print(f"\n[Time] Total evaluation time: {total_time/60:.2f} min")
    print(f"[Time] Script end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    consistency_evaluation_main()
