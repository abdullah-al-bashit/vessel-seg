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


# ── Hard-negative mining BCE ───────────────────────────────────────────────────

def bce_loss_hard_neg(pred, target, weight=None, hard_neg_factor=2.0):
    """
    BCE with hard-negative mining: increases the loss weight on background pixels
    where the model is confidently predicting "vessel" (these are the mistakes
    that drive over-prediction / FP coverage in low-intensity regions).

    Procedure:
      1. Compute per-pixel sigmoid probabilities (no_grad — used only for weighting).
      2. For each background pixel (target=0), the "wrongness" = sigmoid(logit)
         — high means the model is confidently wrong.
      3. Boost that pixel's BCE weight by (1 + hard_neg_factor * wrongness),
         so a pixel the model is 100% sure is vessel (but isn't) contributes
         (1 + hard_neg_factor)× its normal weight to the loss.
      4. Multiply with the existing Hanning gate so tile-edge pixels still
         get downweighted.

    pred:            (B, 1, H, W)  raw logits (NOT sigmoided)
    target:          (B, 1, H, W)  binary float mask
    weight:          (B, 1, H, W)  Hanning gate (or None)
    hard_neg_factor: scalar — peak boost for the most-wrong background pixel.
                     factor=0 → standard BCE. factor=2 → up to 3× weight.
    """
    # reduction='none' → keeps per-pixel loss so we can apply the weight maps.
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # (B, 1, H, W)

    # Compute hard-negative weight (no gradient — it's just a weighting signal).
    with torch.no_grad():
        prob      = torch.sigmoid(pred)            # (B, 1, H, W) ∈ [0, 1]
        bg        = (target < 0.5).float()         # 1 where background, 0 elsewhere
        wrongness = prob * bg                      # high only where model says "vessel" but truth says "no"
        hn_w      = 1.0 + hard_neg_factor * wrongness  # ∈ [1, 1 + factor]

    if weight is not None:
        # Combined weight: Hanning × hard-negative boost
        w_total = weight * hn_w
        return (loss * w_total).sum() / (w_total.sum() + 1e-8)

    return (loss * hn_w).mean()


# ── Pixel prototype contrastive loss ───────────────────────────────────────────

def prototype_contrastive_loss(feats, target, n_samples=256, temp=0.1):
    """
    Prototype-based contrastive loss on decoder features.

    Idea: vessel-pixel features should cluster around a "vessel prototype" and be
    far from a "background prototype" — and vice versa. Trains the decoder to
    produce features that linearly separate vessel vs background, which directly
    helps in regions where intensity alone is ambiguous (e.g. low-intensity
    background that texturally resembles a vessel).

    Procedure (per batch sample):
      1. L2-normalize all pixel features (so similarity = dot product = cosine).
      2. Sample up to `n_samples` vessel pixels and `n_samples` background pixels.
      3. Compute prototypes by averaging the sampled features:
           v_proto  = mean(vessel feats)        — what a "true vessel" looks like
           b_proto  = mean(background feats)    — what "true background" looks like
      4. For each vessel sample, classify it against {v_proto, b_proto} using
         cross-entropy with target=v_proto. Same for background samples with
         target=b_proto. The temperature `temp` controls how sharp the
         classification is (lower = sharper, harder).
      5. Average the two cross-entropy terms.

    feats:     (B, C, H, W)  decoder features (e.g. F_fused, 32 channels)
    target:    (B, 1, H, W)  binary mask
    n_samples: int — pixels sampled per class per batch element (caps memory)
    temp:      float — InfoNCE temperature
    Returns:   scalar loss
    """
    B, C, Hf, Wf = feats.shape
    feats_flat   = F.normalize(feats.reshape(B, C, -1), dim=1)   # (B, C, Hf*Wf) — unit-norm features

    # Features come from inside the model (head input) and live at SAM2's
    # internal 1024×1024 resolution, while target is at the original tile
    # resolution (e.g. 1300×1024). Downsample target to match feats so pixel
    # indices are valid; nearest-neighbour preserves the binary 0/1 labels.
    if target.shape[-2:] != (Hf, Wf):
        target = F.interpolate(target, size=(Hf, Wf), mode='nearest')
    target_flat = target.reshape(B, -1)                           # (B, Hf*Wf) ∈ {0, 1}

    losses = []
    for b in range(B):
        pos_idx = torch.where(target_flat[b] > 0.5)[0]            # vessel pixel indices
        neg_idx = torch.where(target_flat[b] < 0.5)[0]            # background pixel indices
        if len(pos_idx) < 2 or len(neg_idx) < 2:
            continue                                              # skip degenerate tiles

        # Random subsample to bound memory (full image has up to ~1.3M pixels).
        n_pos   = min(n_samples, len(pos_idx))
        n_neg   = min(n_samples, len(neg_idx))
        pos_idx = pos_idx[torch.randperm(len(pos_idx), device=feats.device)[:n_pos]]
        neg_idx = neg_idx[torch.randperm(len(neg_idx), device=feats.device)[:n_neg]]

        pos = feats_flat[b, :, pos_idx]                           # (C, n_pos)
        neg = feats_flat[b, :, neg_idx]                           # (C, n_neg)

        # Class prototypes (re-normalised after averaging — averaging ≠ unit-norm).
        v_proto = F.normalize(pos.mean(dim=1, keepdim=True), dim=0)   # (C, 1)
        b_proto = F.normalize(neg.mean(dim=1, keepdim=True), dim=0)   # (C, 1)

        # For positives: similarity to v_proto should beat similarity to b_proto.
        # Logits shape (n_pos, 2), target = 0 (= v_proto column).
        sim_pos_v = (pos.T @ v_proto).squeeze(-1) / temp          # (n_pos,)
        sim_pos_b = (pos.T @ b_proto).squeeze(-1) / temp          # (n_pos,)
        loss_pos  = F.cross_entropy(
            torch.stack([sim_pos_v, sim_pos_b], dim=1),
            torch.zeros(n_pos, dtype=torch.long, device=feats.device),
        )

        # For negatives: similarity to b_proto should beat similarity to v_proto.
        # Same 2-class CE but target = 1 (= b_proto column).
        sim_neg_v = (neg.T @ v_proto).squeeze(-1) / temp
        sim_neg_b = (neg.T @ b_proto).squeeze(-1) / temp
        loss_neg  = F.cross_entropy(
            torch.stack([sim_neg_v, sim_neg_b], dim=1),
            torch.ones(n_neg, dtype=torch.long, device=feats.device),
        )

        losses.append((loss_pos + loss_neg) / 2)

    if not losses:
        # All tiles in the batch were degenerate (rare); return zero with grad.
        return feats.sum() * 0.0
    return torch.stack(losses).mean()


# ── Sharpness boundary loss ────────────────────────────────────────────────────

def sharpness_boundary_loss(logits, sharpness):
    """
    Penalise vessel predictions that cross sharp→blurry focus boundaries.

    Biology: real vessels don't straddle the focal plane boundary — if a vessel
    is in focus it stays in focus along its length. A prediction that is high in
    a sharp region AND bleeds into an adjacent blurry region is almost always a
    false positive artefact, not a real vessel continuation.

    Mechanism:
      1. Compute |∇sharpness| — peaks exactly at the sharp/blurry transition.
      2. Treat the boundary as a spatial weight map.
      3. Apply BCE(pred, 0) * boundary_weight → any vessel prediction that sits
         on a focus boundary is penalised, regardless of what GT says.

    logits:     (B, 1, H, W)  raw logits (before sigmoid) — safe for AMP autocast
    sharpness:  (B, 1, H, W)  per-pixel sharpness ∈ [0, 1]
    Returns:    scalar loss
    """
    # ── Sharpness gradient via finite differences ──────────────────────────────
    # dy: vertical gradient — how fast sharpness changes row-to-row.
    #     Computed as: s[row+1] - s[row] for all rows except the last.
    #     Shape (B,1,H-1,W); padded to (B,1,H,W) by repeating the last row.
    #     A large |dy| at row r means rows r and r+1 are on opposite sides of
    #     the focus boundary (one sharp, one blurry).
    dy = sharpness[:, :, 1:, :] - sharpness[:, :, :-1, :]   # (B,1,H-1,W)
    dy = F.pad(dy, (0, 0, 0, 1))                              # pad bottom row → (B,1,H,W)

    # dx: horizontal gradient — how fast sharpness changes column-to-column.
    #     Computed as: s[col+1] - s[col] for all cols except the last.
    #     Shape (B,1,H,W-1); padded to (B,1,H,W) by repeating the last column.
    dx = sharpness[:, :, :, 1:] - sharpness[:, :, :, :-1]   # (B,1,H,W-1)
    dx = F.pad(dx, (0, 1, 0, 0))                              # pad right col  → (B,1,H,W)

    # Gradient magnitude = Euclidean length of (dx, dy) at each pixel.
    # This is the "edge strength" of the sharpness map — 0 in flat regions
    # (uniformly sharp or uniformly blurry) and high at transitions.
    boundary = (dy ** 2 + dx ** 2).sqrt()                     # (B,1,H,W)

    # Normalise per-sample so scale is consistent regardless of tile content.
    boundary = boundary / (boundary.flatten(1).max(dim=1).values[:, None, None, None] + 1e-8)

    # BCE_with_logits(logits, 0) = log(1 + exp(logits)): penalises any
    # positive prediction. AMP-safe (unlike F.binary_cross_entropy which
    # requires float32). Multiplied by boundary so only predictions AT
    # focus transitions are penalised — flat sharp/blurry regions get ~0.
    return (F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
            * boundary).mean()


# ── Combined Gated Loss ────────────────────────────────────────────────────────

class VesselLoss(nn.Module):
    """
    L = Dice(pred, gt, W) + BCE_hard_neg(logits, gt, W) + λ_cl·clDice(pred, gt, W)
        + λ_bd·sharpness_boundary(pred, sharpness)      (when sharpness provided)
        + λ_ct·prototype_contrastive(feats, gt)          (when feats provided)

    Hyperparameters:
      lambda_cldice    weight on clDice                  default 1.0
      lambda_boundary  weight on sharpness boundary loss default 0.5
      lambda_contrast  weight on prototype contrastive   default 0.1
      hard_neg_factor  peak boost for wrong bg pixels    default 2.0
    """
    def __init__(self, lambda_cldice=1.0, lambda_boundary=0.5,
                 lambda_contrast=0.1, hard_neg_factor=2.0):
        super().__init__()
        self.lam             = lambda_cldice
        self.lam_bd          = lambda_boundary
        self.lam_ct          = lambda_contrast
        self.hard_neg_factor = hard_neg_factor

    def forward(self, logits, target, hann_weight, sharpness=None, feats=None):
        """
        logits:      (B, 1, H, W)   raw decoder output
        target:      (B, 1, H, W)   binary float mask
        hann_weight: (B, 1, H, W)   Hanning × sharpness gate
        sharpness:   (B, 1, H, W)   per-pixel sharpness map (optional)
        feats:       (B, C, H, W)   decoder features for contrastive loss (optional)
        """
        pred = torch.sigmoid(logits)

        l_dice = dice_loss(pred,   target, hann_weight)
        l_bce  = bce_loss_hard_neg(logits, target, hann_weight,
                                   hard_neg_factor=self.hard_neg_factor)
        l_cl   = cldice_loss(pred, target, hann_weight)

        total = l_dice + l_bce + self.lam * l_cl

        # Boundary penalty: discourage predictions that cross sharp→blurry edges.
        if sharpness is not None and self.lam_bd > 0:
            # Resize sharpness to match logits resolution if SAM2 changed spatial dims.
            s = sharpness if sharpness.shape == logits.shape else \
                F.interpolate(sharpness, size=logits.shape[-2:], mode='bilinear', align_corners=False)
            l_bd   = sharpness_boundary_loss(logits, s)
            total  = total + self.lam_bd * l_bd
            l_bd_val = l_bd.item()
        else:
            l_bd_val = 0.0

        # Contrastive only contributes when features are passed in.
        if feats is not None and self.lam_ct > 0:
            l_ct   = prototype_contrastive_loss(feats, target)
            total  = total + self.lam_ct * l_ct
            l_ct_val = l_ct.item()
        else:
            l_ct_val = 0.0

        return total, {
            'dice':       l_dice.item(),
            'bce':        l_bce.item(),
            'cldice':     l_cl.item(),
            'boundary':   l_bd_val,
            'contrast':   l_ct_val,
        }
