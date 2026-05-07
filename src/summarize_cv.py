"""Cross-validation summary — called directly from train.py at end of training."""

import numpy as np
from scipy import stats


def print_cv_summary(folds, best_fold, best_loss_folds, best_epoch_folds,
                     n_folds, te_loss, dice_per_image, ckpt_dir):
    """
    Print publication-ready CV summary directly from in-memory training results.

    Args:
        folds:            number of folds actually trained
        best_fold:        1-based index of selected fold
        best_loss_folds:  list of best val loss per fold
        best_epoch_folds: list of best epoch per fold
        n_folds:          total folds in the CV split (from config)
        te_loss:          scalar test loss
        dice_per_image:   list of per-image Dice scores on test set
        ckpt_dir:         checkpoint directory (for display)
    """
    lines = []
    sep = "=" * 70

    lines += ["\n" + sep, "CROSS-VALIDATION RESULTS SUMMARY", sep + "\n"]

    # Fold-wise table
    lines += ["Fold-wise Validation Performance:", "-" * 50,
              f"{'Fold':<6} {'Best Epoch':<12} {'Val Loss':<12} {'Status':<15}",
              "-" * 50]
    for i, (loss, epoch) in enumerate(zip(best_loss_folds, best_epoch_folds)):
        fold_num = i + 1
        status = "✓ SELECTED" if fold_num == best_fold else ""
        lines.append(f"{fold_num:<6} {epoch:<12} {loss:<12.4f} {status:<15}")

    # CV summary
    cv_mean = float(np.mean(best_loss_folds))
    cv_std  = float(np.std(best_loss_folds))
    lines += ["\n" + "-" * 50,
              f"Cross-Validation Mean ± Std: {cv_mean:.4f} ± {cv_std:.4f}"]

    # Test set
    lines += ["\n" + sep, "TEST SET PERFORMANCE", sep]
    lines.append(f"Test Loss: {te_loss:.4f}")

    if dice_per_image:
        arr = np.array(dice_per_image)
        lines += [
            f"\nPer-image Dice Coefficient:",
            f"  Mean:     {arr.mean():.4f}",
            f"  Std:      {arr.std():.4f}",
            f"  Min:      {arr.min():.4f}",
            f"  Max:      {arr.max():.4f}",
            f"  Median:   {np.median(arr):.4f}",
            f"  N images: {len(arr)}",
            f"\nDice Threshold Accuracy:",
        ]
        for thr in [0.80, 0.85, 0.90, 0.95]:
            n = (arr >= thr).sum()
            lines.append(f"  Dice ≥ {thr}: {n}/{len(arr)} ({100*n/len(arr):.1f}%)")

        ci = stats.t.interval(0.95, len(arr) - 1, loc=arr.mean(), scale=stats.sem(arr))
        lines += [
            f"\nStatistical Summary:",
            f"  95% CI:  [{ci[0]:.4f}, {ci[1]:.4f}]",
            f"  SEM:     {stats.sem(arr):.4f}",
            f"  IQR:     {np.percentile(arr, 75) - np.percentile(arr, 25):.4f}",
        ]

    # Model selection
    lines += ["\n" + sep, "MODEL SELECTION", sep,
              f"Selected Fold:     {best_fold}",
              f"Best Epoch:        {best_epoch_folds[best_fold - 1]}",
              f"Validation Loss:   {best_loss_folds[best_fold - 1]:.4f}",
              f"Checkpoint:        {ckpt_dir}/best_model.pth",
              "\n" + sep + "\n"]

    summary = "\n".join(lines)
    print(summary)
    return summary
