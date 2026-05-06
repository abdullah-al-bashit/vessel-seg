#!/usr/bin/env python3
"""
Generate publication-ready cross-validation summary from training logs.

Usage:
    python summarize_cv.py <job_id>
    python summarize_cv.py /path/to/logs/train_JOBID.out
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def parse_fold_results(log_path: str) -> Dict:
    """Parse fold-wise CV results from training log."""
    with open(log_path) as f:
        content = f.read()

    results = {
        'folds': {},
        'best_fold': None,
        'test_dice': [],
        'test_loss': None,
    }

    # Extract fold summaries: "Fold X best val loss: Y (epoch Z)"
    fold_pattern = r'Fold (\d+) best val loss: ([\d.]+) \(epoch (\d+)\)'
    for match in re.finditer(fold_pattern, content):
        fold_num = int(match.group(1))
        val_loss = float(match.group(2))
        best_epoch = int(match.group(3))
        results['folds'][fold_num] = {
            'best_epoch': best_epoch,
            'val_loss': val_loss,
        }

    # Extract which fold was chosen: "Evaluating fold X (epoch Y, val loss Z)"
    eval_pattern = r'Evaluating fold (\d+) \(epoch (\d+), val loss ([\d.]+)\) on held-out test set'
    eval_match = re.search(eval_pattern, content)
    if eval_match:
        results['best_fold'] = int(eval_match.group(1))

    # Extract test loss: "Test loss X.XXXX"
    test_pattern = r'Test loss ([\d.]+)'
    test_match = re.search(test_pattern, content)
    if test_match:
        results['test_loss'] = float(test_match.group(1))

    # Extract per-image test Dice: "dice = X.XXXX"
    dice_pattern = r'dice = ([\d.]+)'
    for match in re.finditer(dice_pattern, content):
        results['test_dice'].append(float(match.group(1)))

    # Extract CV summary: "N-fold CV val loss: X.XXXX ± Y.YYYY"
    cv_pattern = r'(\d+)-fold CV val loss: ([\d.]+) ± ([\d.]+)'
    cv_match = re.search(cv_pattern, content)
    if cv_match:
        results['n_folds'] = int(cv_match.group(1))
        results['cv_mean'] = float(cv_match.group(2))
        results['cv_std'] = float(cv_match.group(3))

    return results


def print_publication_summary(results: Dict) -> str:
    """Generate publication-ready summary table."""
    summary = []
    summary.append("\n" + "="*80)
    summary.append("CROSS-VALIDATION RESULTS SUMMARY")
    summary.append("="*80 + "\n")

    # Fold-wise results table
    summary.append("Fold-wise Validation Performance:")
    summary.append("-" * 60)
    summary.append(f"{'Fold':<6} {'Best Epoch':<12} {'Val Loss':<12} {'Status':<15}")
    summary.append("-" * 60)

    for fold_num in sorted(results['folds'].keys()):
        fold_data = results['folds'][fold_num]
        is_best = "✓ SELECTED" if fold_num == results['best_fold'] else ""
        summary.append(
            f"{fold_num:<6} {fold_data['best_epoch']:<12} "
            f"{fold_data['val_loss']:<12.4f} {is_best:<15}"
        )

    # Cross-validation summary
    if 'cv_mean' in results:
        summary.append("\n" + "-" * 60)
        summary.append(
            f"Cross-Validation Mean ± Std: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}"
        )

    # Test set performance
    summary.append("\n" + "="*80)
    summary.append("TEST SET PERFORMANCE")
    summary.append("="*80)

    if results['test_loss']:
        summary.append(f"Test Loss: {results['test_loss']:.4f}")

    if results['test_dice']:
        test_dice_arr = np.array(results['test_dice'])
        summary.append(f"\nPer-image Dice Coefficient:")
        summary.append(f"  Mean:   {test_dice_arr.mean():.4f}")
        summary.append(f"  Std:    {test_dice_arr.std():.4f}")
        summary.append(f"  Min:    {test_dice_arr.min():.4f}")
        summary.append(f"  Max:    {test_dice_arr.max():.4f}")
        summary.append(f"  Median: {np.median(test_dice_arr):.4f}")
        summary.append(f"  N images: {len(results['test_dice'])}")

        # Top-k accuracy (Dice ≥ threshold)
        summary.append(f"\nDice Coefficient Thresholds (Top-k Accuracy):")
        for threshold in [0.80, 0.85, 0.90, 0.95]:
            count = (test_dice_arr >= threshold).sum()
            pct = 100 * count / len(test_dice_arr)
            summary.append(f"  Dice ≥ {threshold}: {count}/{len(test_dice_arr)} ({pct:.1f}%)")

        # Statistical summary
        summary.append(f"\nStatistical Summary:")
        ci = stats.t.interval(0.95, len(test_dice_arr)-1,
                              loc=test_dice_arr.mean(),
                              scale=stats.sem(test_dice_arr))
        summary.append(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        summary.append(f"  SEM (Std Error Mean): {stats.sem(test_dice_arr):.4f}")
        summary.append(f"  Q1 (25th percentile): {np.percentile(test_dice_arr, 25):.4f}")
        summary.append(f"  Q3 (75th percentile): {np.percentile(test_dice_arr, 75):.4f}")
        summary.append(f"  IQR (Interquartile Range): {np.percentile(test_dice_arr, 75) - np.percentile(test_dice_arr, 25):.4f}")

    # Model selection
    summary.append("\n" + "="*80)
    summary.append("MODEL SELECTION")
    summary.append("="*80)
    if results['best_fold'] and results['best_fold'] in results['folds']:
        summary.append(f"Selected Fold: {results['best_fold']}")
        summary.append(
            f"Best Epoch: {results['folds'][results['best_fold']]['best_epoch']}"
        )
        summary.append(
            f"Validation Loss: {results['folds'][results['best_fold']]['val_loss']:.4f}"
        )
        summary.append("Checkpoint: best_model.pth (all fold-specific checkpoints deleted)")
    else:
        summary.append("(No fold selection data found in logs)")

    summary.append("\n" + "="*80 + "\n")
    return "\n".join(summary)


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_cv.py <job_id_or_log_path>")
        sys.exit(1)

    arg = sys.argv[1]

    # Resolve log path
    if arg.startswith('/'):
        log_path = arg
    else:
        # Assume it's a job ID
        log_path = f"/home/a.bashit/vessel_seg/logs/train_{arg}.out"

    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    # Parse and summarize
    results = parse_fold_results(str(log_path))

    if not results['folds']:
        print(f"Warning: No fold results found in {log_path}")
        return results

    summary = print_publication_summary(results)
    print(summary)

    # Save to file
    summary_path = log_path.parent / f"{log_path.stem}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}\n")

    return results


if __name__ == '__main__':
    main()
