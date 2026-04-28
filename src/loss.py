import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dice Loss ──────────────────────────────────────────────────────────────────

def dice_loss(pred, target, weight=None, eps=1e-6):
    """
    pred:   (B, 1, H, W)  sigmoid probabilities
    target: (B, 1, H, W)  binary float
    weight: (B, 1, H, W)  Hanning gate  (or None)
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

    # Numerator: weighted overlap between prediction and ground truth.
    inter  = (pred * target).sum(dim=1)          # (B,)
    # Denominator: sum of both masks independently (not intersection).
    denom  = pred.sum(dim=1) + target.sum(dim=1) # (B,)
    # eps prevents division by zero on empty tiles; average over the batch.
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()  # scalar


# ── BCE Loss ───────────────────────────────────────────────────────────────────

def bce_loss(pred, target, weight=None):
    """
    pred:   (B, 1, H, W)  logits (before sigmoid)
    target: (B, 1, H, W)  binary float
    weight: (B, 1, H, W)  Hanning gate
    """
    # reduction='none' keeps per-pixel losses so we can apply the Hanning gate.
    # Uses logits directly (numerically stable log-sum-exp trick internally).
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # (B, 1, H, W)

    if weight is not None:
        loss = loss * weight                          # (B, 1, H, W) — zero out edge pixels
        # Divide by total weight, not pixel count, so the scale stays consistent
        # with the unweighted case despite many near-zero Hanning edge values.
        return loss.sum() / (weight.sum() + 1e-8)    # scalar

    return loss.mean()                               # scalar


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
    L = Dice(pred, gt, W) + BCE(logits, gt, W) + λ·clDice(pred, gt, W)
    All three gated by Hanning weight map W.
    λ = 0.5 (clDice weight)
    """
    def __init__(self, lambda_cldice=0.5):
        super().__init__()
        self.lam = lambda_cldice

    def forward(self, logits, target, hann_weight):
        """
        logits:      (B, 1, H, W)  raw decoder output
        target:      (B, 1, H, W)  binary float mask
        hann_weight: (B, 1, H, W)  Hanning gate
        """
        pred = torch.sigmoid(logits)

        l_dice  = dice_loss(pred,   target, hann_weight)
        l_bce   = bce_loss(logits,  target, hann_weight)
        l_cl    = cldice_loss(pred, target, hann_weight)

        return l_dice + l_bce + self.lam * l_cl, {
            'dice':   l_dice.item(),
            'bce':    l_bce.item(),
            'cldice': l_cl.item(),
        }
