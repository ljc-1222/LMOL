# -*- coding: utf-8 -*-
"""
LMOL Evaluation Main Orchestration

This module provides the main evaluation orchestration for the LMOL project:
- Command-line argument parsing
- Evaluation coordination across folds
- Results aggregation and reporting

Key Features:
- Flexible evaluation configuration
- Comprehensive results reporting
- Confusion matrix generation
- Performance monitoring
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List

from configs.config import config
# Lazy imports to avoid dependency issues
# from .evaluator import load_single_fold, evaluate_fold, free_model
# from .metrics import plot_and_save_cm


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


def evaluation_main():
    """Main evaluation function."""
    # Lazy imports to avoid dependency issues
    from .evaluator import load_single_fold, evaluate_fold, free_model
    from .metrics import plot_and_save_cm
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate models on beauty comparison task")
    parser.add_argument("--samples", "-n", type=int, default=None, 
                      help=f"Number of pairs to evaluate per fold (default: 1000, 0 for all)")
    parser.add_argument("--run-dir", type=str, default=None,
                      help="Specific run directory to evaluate (default: latest)")
    parser.add_argument("--model-type", "-m", type=str, choices=["best", "last", "fold"], default="best",
                      help="Model type to evaluate: 'best', 'last', or 'fold' (using fold dir directly) (default: best)")
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
    if n_samples == 0:
        print(f"[Config] Evaluating ALL pairs in each CSV")
    elif n_samples is not None:
        print(f"[Config] Evaluating {n_samples} pairs per fold")
    else:
        print(f"[Config] Using default sample size: 1000 pairs per fold")
    
    print(f"[Config] Using {args.model_type} model for evaluation")

    grand_correct = 0
    grand_total = 0

    # 用於彙總所有 folds 的標籤
    grand_y_true: List[str] = []
    grand_y_pred: List[str] = []

    for fold_idx, fold_path in enumerate(fold_dirs, start=1):
        eval_csv = get_eval_csv_for_fold(fold_idx)
        print(f"Evaluating {fold_path.name} on {eval_csv}")
        model, processor = load_single_fold(fold_path, args.model_type)
        # 注意：evaluate_fold 現在回傳 (correct, total, y_true, y_pred)
        c, t, y_true, y_pred = evaluate_fold(fold_path.name, model, processor, eval_csv, fold_path, n_samples)
        grand_correct += c
        grand_total += t
        grand_y_true.extend(y_true)
        grand_y_pred.extend(y_pred)
        del processor
        free_model(model)
        
        # 繪製並儲存此 fold 的 confusion matrix（儲存在選定的模型目錄）
        try:
            # Determine the model folder based on model_type
            if args.model_type in ("best", "last"):
                model_folder = fold_path / args.model_type
            else:  # "fold" - use fold directory directly
                model_folder = fold_path
            
            # Ensure the model folder exists
            model_folder.mkdir(parents=True, exist_ok=True)
            
            out_cm = model_folder / "confusion_matrix.png"
            plot_and_save_cm(y_true, y_pred, out_cm, title=f"{fold_path.name} ({args.model_type}) Confusion Matrix")
            print(f"[Plot] Saved confusion matrix to {out_cm}")
        except Exception as e:
            print(f"[Warn] Failed to plot/save confusion matrix for {fold_path.name}: {e}")
    
    if grand_total:
        print(f"[Overall] accuracy={grand_correct/grand_total:.6f} ({grand_correct}/{grand_total})")
    
    # 繪製並儲存整體 confusion matrix（儲存在 run_dir）
    try:
        overall_cm_path = run_dir / "confusion_overall.png"
        plot_and_save_cm(grand_y_true, grand_y_pred, overall_cm_path, title="Overall Confusion Matrix")
        print(f"[Plot] Saved overall confusion matrix to {overall_cm_path}")
    except Exception as e:
        print(f"[Warn] Failed to plot/save overall confusion matrix: {e}")

    total_dur = time.perf_counter() - run_start
    print(f"[Time] Script end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | total={total_dur/60:.2f} min")


if __name__ == '__main__':
    evaluation_main()
