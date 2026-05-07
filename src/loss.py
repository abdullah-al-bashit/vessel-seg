import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tversky Loss ───────────────────────────────────────────────────────────────

def tversky_loss(pred, target, weight=None, alpha=0.3, beta=0.7, eps=1e-6):
    """
    Tversky loss — generalisation of Dice with asymmetric FP/FN weighting.
    alpha=0.3, beta=0.7 penalises FN 2.3× more than FP, directly targeting
    missed thin vessels (the dominant failure mode observed in W&B FN panels).

    L = 1 - TP / (TP + α·FP + β·FN)

    TN (correctly predicted background) is intentionally absent from the denominator.
    Including TN would dilute the loss — vessel pixels are rare (~10–20% of the image),
    so a large TN count would dominate the denominator and make the loss insensitive
    to both FP and FN mistakes. Excluding TN keeps the loss focused purely on the
    foreground (vessel) prediction quality, exactly like standard Dice.
      α=0.5, β=0.5 → standard Dice
      α<β          → penalise FN more (recover missed vessels)

    pred:   (B, 1, H, W)  sigmoid probabilities
    target: (B, 1, H, W)  binary float
    weight: (B, 1, H, W)  Hanning gate (or None)
    """
    # Flatten (B, 1, H, W) → (B, H*W); contiguous() required before view()
    # when the tensor may be non-contiguous after upstream ops (e.g. flip).
    pred   = pred.contiguous().view(pred.size(0), -1)    # (B, H*W)
    target = target.contiguous().view(target.size(0), -1)  # (B, H*W)

    if weight is not None:
        # Gate each pixel by its Hanning weight so tile-edge pixels
        # contribute less to the loss than center pixels.
        w      = weight.contiguous().view(weight.size(0), -1)  # (B, H*W)
        pred   = pred * w    # (B, H*W)
        target = target * w  # (B, H*W)

    tp    = (pred * target).sum(dim=1)               # (B,) true positives
    fp    = (pred * (1 - target)).sum(dim=1)          # (B,) false positives
    fn    = ((1 - pred) * target).sum(dim=1)          # (B,) false negatives
    return (1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()  # scalar


# ── clDice Loss ────────────────────────────────────────────────────────────────

def soft_erode(img):
    """Soft morphological erosion via max-pool."""
    # Negate → max-pool → negate implements min-pool (erosion) without a native op.
    # Two separable passes (3×1, 1×3) approximate a 3×3 erosion at lower cost.
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))  # (B, 1, H, W)
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))  # (B, 1, H, W)
    return torch.min(p1, p2)                                                  # (B, 1, H, W)


def soft_dilate(img):
    """Soft morphological dilation via max-pool."""
    return F.max_pool2d(img, kernel_size=(3, 3), stride=1, padding=1)        # (B, 1, H, W)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iters=10):
    """Iterative soft skeletonization for clDice.

    Each iteration peels one layer off the current foreground (delta) and
    adds the residual ridge pixels to the skeleton accumulator.
    img: (B, 1, H, W) sigmoid probabilities in [0, 1].
    """
    skel  = torch.zeros_like(img)   # (B, 1, H, W) — accumulated skeleton so far
    delta = img                     # (B, 1, H, W) — foreground remaining to peel
    for _ in range(iters):
        erod   = soft_erode(delta)          # (B, 1, H, W) — one erosion layer removed
        opened = soft_open(erod)            # (B, 1, H, W) — erode then dilate; removes thin protrusions
        # Ridge pixels: present after erosion but removed by opening → centerline candidates.
        temp  = F.relu(erod - opened)       # (B, 1, H, W)
        # Add new ridge pixels, avoiding double-counting pixels already in skel.
        # F.relu(temp - skel*temp) = temp * (1 - skel), clamped to [0, inf].
        skel  = skel + F.relu(temp - skel * temp)   # (B, 1, H, W)
        delta = erod                        # peel the next layer on the next iteration
        if delta.sum() == 0:               # foreground fully consumed — stop early
            break
    return skel                             # (B, 1, H, W)


def cldice_loss(pred_prob, target, weight=None, eps=1e-6):
    """
    Centerline Dice loss — topology-aware.
    pred_prob: (B, 1, H, W)  sigmoid probabilities
    target:    (B, 1, H, W)  binary float
    weight:    (B, 1, H, W)  Hanning gate
    """
    # Extract centerline of prediction and ground truth independently.
    # Skeletonizing both ensures the loss penalises topology errors
    # (broken vessels, false connections) rather than boundary imprecision.
    skel_pred   = soft_skeletonize(pred_prob)  # (B, 1, H, W)
    skel_target = soft_skeletonize(target)     # (B, 1, H, W)

    if weight is not None:
        # Apply Hanning gate before computing dot products so tile-edge
        # skeleton pixels don't bias the topology score.
        pred_prob   = pred_prob   * weight  # (B, 1, H, W)
        target      = target      * weight  # (B, 1, H, W)
        skel_pred   = skel_pred   * weight  # (B, 1, H, W)
        skel_target = skel_target * weight  # (B, 1, H, W)

    # Topology precision: how much of the predicted skeleton overlaps the GT mask.
    # Low t_prec → predicted centerline runs outside the true vessel.
    t_prec = ((skel_pred   * target).sum()    + eps) / (skel_pred.sum()    + eps)  # scalar
    # Topology sensitivity: how much of the GT skeleton is covered by the prediction.
    # Low t_sens → true vessel centerline is missed (broken/absent vessels).
    t_sens = ((skel_target * pred_prob).sum() + eps) / (skel_target.sum() + eps)   # scalar

    # F1 of the two topology scores, subtracted from 1 to form a loss.
    return 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens + eps)  # scalar


# ── Combined Gated Loss ────────────────────────────────────────────────────────

class VesselLoss(nn.Module):
    """
    L = Tversky(pred, gt, W) + λ_cl·clDice(pred, gt, W)

    Tversky (α=0.3, β=0.7) penalises FN 2.3× more than FP to recover missed thin vessels.
    clDice preserves vessel topology and connectivity.
    BCE dropped: Tversky already subsumes its signal, and hard-neg mining competed with
    the FN-recovery goal by pushing gradients in the opposite direction.

    Hyperparameters:
      lambda_cldice  weight on clDice               default 1.0
      tversky_beta   FN weight in Tversky denominator  default 0.7
                     alpha is derived as (1 - beta) so alpha + beta = 1 always holds
    """
    def __init__(self, lambda_cldice=1.0, tversky_beta=0.7):
        super().__init__()
        self.lam           = lambda_cldice
        self.tversky_beta  = tversky_beta          # FN weight: higher → penalise missed vessels more
        self.tversky_alpha = 1.0 - tversky_beta    # FP weight: derived so alpha + beta = 1

    def forward(self, logits, target, hann_weight):
        """
        logits:      (B, 1, H, W)   raw decoder output
        target:      (B, 1, H, W)   binary float mask
        hann_weight: (B, 1, H, W)   Hanning × sharpness gate
        """
        pred = torch.sigmoid(logits)  # (B, 1, H, W)

        l_tversky = tversky_loss(pred, target, hann_weight,
                                 alpha=self.tversky_alpha,
                                 beta=self.tversky_beta)   # asymmetric FN penalty
        l_cl      = cldice_loss(pred, target, hann_weight) # topology / connectivity

        total = l_tversky + self.lam * l_cl

        return total, {
            'tversky': l_tversky.item(),
            'cldice':  l_cl.item(),
        }
