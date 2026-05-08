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

    # w=1 when no Hanning gate → standard Tversky/Dice formula.
    # Weight each term directly; do NOT pre-multiply pred and target by w first —
    # that introduces w² in TP and corrupts FP/FN via (1-target·w) ≠ (1-target)·w.
    w  = weight.contiguous().view(weight.size(0), -1) if weight is not None else 1.0
    tp = (pred * target       * w).sum(dim=1)  # (B,)
    fp = (pred * (1 - target) * w).sum(dim=1)  # (B,)
    fn = ((1 - pred) * target * w).sum(dim=1)  # (B,)

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


def cldice_loss(pred_prob, target, weight=None, iters=5, eps=1e-6):
    """
    Centerline Dice loss (clDice) — topology-aware F1 of skeleton precision × sensitivity.

    iters=5 halves pooling intermediates vs 10 while still producing a usable centerline.

    pred_prob: (B, 1, H, W)  sigmoid probabilities
    target:    (B, 1, H, W)  binary float
    weight:    (B, 1, H, W)  Hanning gate (or None)
    """
    # float32 cast: AMP/autocast gives float16 where erod−opened ≈ 1e-4 rounds to 0,
    # making skel_pred=0 → gradient vanishes. float32 preserves sub-millipoint diffs.
    pred_f32   = pred_prob.float()
    target_f32 = target.float()
    w          = weight.float() if weight is not None else 1.0

    skel_pred   = soft_skeletonize(pred_f32,   iters=iters)  # (B, 1, H, W) float32
    skel_target = soft_skeletonize(target_f32, iters=iters)  # (B, 1, H, W) float32; no backprop

    # Standard clDice formula from Shit et al. 2021.
    t_prec = ((skel_pred   * target_f32 * w).sum() + eps) / ((skel_pred   * w).sum() + eps)
    t_sens = ((skel_target * pred_f32   * w).sum() + eps) / ((skel_target * w).sum() + eps)
    return 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens + eps)  # scalar


# ── Skeleton Density Loss ──────────────────────────────────────────────────────

def skeleton_density_loss(pred_prob, target, weight=None, iters=5, eps=1e-6):
    """
    FN skeleton density: penalises blob-shaped missed vessel branches.

    Operates on the FN region (1−pred)×target (pixels the model misses).
      thin missed vessel  → thin FN  → high density → loss≈0
      large missed branch → blob FN  → low density  → loss≈1

    Per-image density then mean: each image contributes equally regardless of FN area.
    Empty FN (no misses): fn_sum≈0 → density=eps/eps=1 → loss=0.
    """
    pred_f32   = pred_prob.float()
    target_f32 = target.float()
    w          = weight.float() if weight is not None else 1.0

    fn_pred  = (1.0 - pred_f32) * target_f32                  # (B, 1, H, W)
    skel_fn  = soft_skeletonize(fn_pred, iters=iters)          # (B, 1, H, W)
    skel_sum = (skel_fn * w).sum(dim=(1, 2, 3))                # (B,)
    fn_sum   = (fn_pred * w).sum(dim=(1, 2, 3))                # (B,)
    return (1.0 - (skel_sum + eps) / (fn_sum + eps)).mean()    # scalar


# ── Combined Gated Loss ────────────────────────────────────────────────────────

class VesselLoss(nn.Module):
    """
    L = λ_tv·Tversky(pred, gt, W) + λ_cl·clDice(pred, gt, W) + λ_sd·SkelDensity(pred, gt, W)

    All three terms are independently weighted; set any λ to 0 to disable.
    Current defaults: only Tversky active (λ_cl=0, λ_sd=0).

    Tversky: TP / (TP + α·FP + β·FN),  α = 1 − β
      β=0.5  → standard Dice
      β>0.5  → penalises missed vessels more than false alarms

    Hyperparameters:
      lambda_tversky      default 1.0
      lambda_cldice       default 0.0  (disabled — enable for topology training)
      lambda_skel_density default 0.0  (disabled — enable for blob-penalty training)
      tversky_beta        default 0.5
    """
    def __init__(self, lambda_tversky=1.0, lambda_cldice=0.0,
                 lambda_skel_density=0.0, tversky_beta=0.5):
        super().__init__()
        self.lam_tversky     = lambda_tversky
        self.lam_cldice      = lambda_cldice
        self.lam_skel_density = lambda_skel_density
        self.tversky_beta    = tversky_beta
        self.tversky_alpha   = 1.0 - tversky_beta

    def forward(self, logits, target, hann_weight):
        """
        logits:      (B, 1, H, W)   raw decoder output
        target:      (B, 1, H, W)   binary float mask
        hann_weight: (B, 1, H, W)   Hanning × sharpness gate
        """
        pred = torch.sigmoid(logits)

        l_tv = (tversky_loss(pred, target, hann_weight,
                             alpha=self.tversky_alpha, beta=self.tversky_beta)
                if self.lam_tversky > 0 else pred.new_zeros(1))

        l_cl = (cldice_loss(pred, target, hann_weight)
                if self.lam_cldice > 0 else pred.new_zeros(1))

        l_sd = (skeleton_density_loss(pred, target, hann_weight)
                if self.lam_skel_density > 0 else pred.new_zeros(1))

        total = self.lam_tversky * l_tv + self.lam_cldice * l_cl + self.lam_skel_density * l_sd
        return total, {'tversky': l_tv.item(), 'cldice': l_cl.item(), 'skel_density': l_sd.item()}, pred
