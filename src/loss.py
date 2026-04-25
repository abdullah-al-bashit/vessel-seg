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
    pred   = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    if weight is not None:
        w  = weight.contiguous().view(weight.size(0), -1)
        pred   = pred * w
        target = target * w

    inter  = (pred * target).sum(dim=1)
    denom  = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


# ── BCE Loss ───────────────────────────────────────────────────────────────────

def bce_loss(pred, target, weight=None):
    """
    pred:   (B, 1, H, W)  logits (before sigmoid)
    target: (B, 1, H, W)  binary float
    weight: (B, 1, H, W)  Hanning gate
    """
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

    if weight is not None:
        loss = loss * weight
        return loss.sum() / (weight.sum() + 1e-8)

    return loss.mean()


# ── clDice Loss ────────────────────────────────────────────────────────────────

def soft_erode(img):
    """Soft morphological erosion via max-pool."""
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    """Soft morphological dilation via max-pool."""
    return F.max_pool2d(img, kernel_size=(3, 3), stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iters=10):
    """Iterative soft skeletonization for clDice."""
    skel  = torch.zeros_like(img)
    delta = img
    for _ in range(iters):
        erod  = soft_erode(delta)
        opened = soft_open(erod)
        temp  = F.relu(erod - opened)
        skel  = skel + F.relu(temp - skel * temp)
        delta = erod
        if delta.sum() == 0:
            break
    return skel


def cldice_loss(pred_prob, target, weight=None, eps=1e-6):
    """
    Centerline Dice loss — topology-aware.
    pred_prob: (B, 1, H, W)  sigmoid probabilities
    target:    (B, 1, H, W)  binary float
    weight:    (B, 1, H, W)  Hanning gate
    """
    skel_pred   = soft_skeletonize(pred_prob)
    skel_target = soft_skeletonize(target)

    if weight is not None:
        pred_prob = pred_prob * weight
        target    = target    * weight
        skel_pred = skel_pred * weight
        skel_target = skel_target * weight

    # Topology sensitivity: how much of skeleton is covered
    t_prec = ((skel_pred  * target).sum()    + eps) / (skel_pred.sum()    + eps)
    t_sens = ((skel_target * pred_prob).sum() + eps) / (skel_target.sum() + eps)

    return 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens + eps)


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
