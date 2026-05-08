import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm


def _build_resnet34_encoder(in_chans=3, out_indices=(0, 1, 2, 3, 4)):
    """Build ResNet34 encoder architecture with random weights. Weight loading is handled by create_model."""
    return timm.create_model('resnet34', pretrained=False,
                             features_only=True, in_chans=in_chans,
                             out_indices=out_indices)


class AttentionGate(nn.Module):
    """Gating mechanism: soft attention on skip-connection features guided by decoder query."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g   = nn.Conv2d(F_g,   F_int, 1, padding=0)
        self.W_x   = nn.Conv2d(F_l,   F_int, 1, padding=0)
        # BN before sigmoid recenters pre-activation distribution so the gate
        # suppresses ~half of spatial locations rather than saturating to 1 everywhere.
        self.psi   = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        # Resize g1 to match x spatial dimensions (handle mismatch from uneven upsampling)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x1  = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        self.last_psi = psi.detach()  # Save for visualization (no grad overhead)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention UNet: ResNet34 encoder + UNet decoder with attention gates on all 5 skip connections.
    3-channel input [gray, grad_mag, sharpness] → logits (B, 1, H, W).
    Fully trainable encoder (no frozen backbone).

    Encoder spatial scales (ResNet34 with timm features_only):
      e0: (B,  64, H/2,  W/2)  — stem, finest scale, captures thin vessel edges
      e1: (B,  64, H/4,  W/4)  — after layer1
      e2: (B, 128, H/8,  W/8)  — after layer2
      e3: (B, 256, H/16, W/16) — after layer3
      e4: (B, 512, H/32, W/32) — after layer4, coarsest scale
    """
    def __init__(self):
        super().__init__()
        self.encoder = _build_resnet34_encoder()

        # ── Skip-connection attention gates (encoder scale → decoder query) ──────
        # e0 is 64ch but d1 (decoder at H/2) is 32ch — project before attention + add
        self.proj_e0 = nn.Conv2d(64, 32, kernel_size=1)          # (B,64,H/2,W/2)  → (B,32,H/2,W/2)
        self.att0 = AttentionGate(F_g=32, F_l=32,  F_int=16)     # g=d1, x=proj_e0 → (B,32,H/2,W/2)
        self.att1 = AttentionGate(F_g=256, F_l=256, F_int=128)   # g=d4, x=e3      → (B,256,H/16,W/16)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)    # g=d3, x=e2      → (B,128,H/8,W/8)
        self.att3 = AttentionGate(F_g=64,  F_l=64,  F_int=32)    # g=d2, x=e1      → (B,64,H/4,W/4)

        # ── Decoder upsampling blocks (ConvTranspose2d doubles spatial resolution) ─
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # H/32 → H/16
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # H/16 → H/8
        self.up2 = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)  # H/8  → H/4
        self.up1 = nn.ConvTranspose2d( 64,  32, kernel_size=2, stride=2)  # H/4  → H/2
        self.up0 = nn.ConvTranspose2d( 32,  16, kernel_size=2, stride=2)  # H/2  → H

        # ── Dilated refinement head ───────────────────────────────────────────────
        # Two-stage: wide receptive field (9×9 effective) to fill vessel gaps,
        # then local 3×3 to sharpen boundaries before the final 1×1 projection.
        self.refine = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, dilation=2, padding=4),  # (B,16,H,W) → (B,32,H,W), 9×9 receptive field
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, dilation=1, padding=1),  # (B,32,H,W) → (B,16,H,W), local refinement
        )

        self.head = nn.Conv2d(16, 1, kernel_size=1)                    # (B,16,H,W) → (B,1,H,W) logits

    def forward(self, x, sharpness=None, grad_mag=None):
        """
        x:         (B, 1, H, W)  grayscale image
        sharpness: (B, 1, H, W)  per-pixel sharpness map (optional)
        grad_mag:  (B, 1, H, W)  gradient magnitude map (optional)
        Returns:   (B, 1, H, W)  raw logits (before sigmoid)
        """
        # 3-channel input: [gray, grad_mag, sharpness] gives the encoder explicit
        # texture cues beyond raw intensity. Falls back to repeated grayscale if maps unavailable.
        if sharpness is not None and grad_mag is not None:
            x_enc = torch.cat([x, grad_mag, sharpness], dim=1)  # (B, 3, H, W)
        else:
            x_enc = x.repeat(1, 3, 1, 1)                        # (B, 3, H, W) — grayscale repeated

        # ── Encoder ──────────────────────────────────────────────────────────────
        e0, e1, e2, e3, e4 = self.encoder(x_enc)
        # e0: (B,  64, H/2,  W/2)   e1: (B,  64, H/4,  W/4)
        # e2: (B, 128, H/8,  W/8)   e3: (B, 256, H/16, W/16)   e4: (B, 512, H/32, W/32)

        # ── Decoder with attention-gated skip connections ─────────────────────────
        d4 = self.up4(e4)                                        # (B, 256, H/16, W/16)
        att1_out = self.att1(d4, e3)                             # (B, 256, H/16, W/16)
        if att1_out.shape[2:] != d4.shape[2:]:
            att1_out = F.interpolate(att1_out, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up3(d4 + att1_out)                            # (B, 128, H/8,  W/8)

        att2_out = self.att2(d3, e2)                             # (B, 128, H/8,  W/8)
        if att2_out.shape[2:] != d3.shape[2:]:
            att2_out = F.interpolate(att2_out, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up2(d3 + att2_out)                            # (B,  64, H/4,  W/4)

        att3_out = self.att3(d2, e1)                             # (B,  64, H/4,  W/4)
        if att3_out.shape[2:] != d2.shape[2:]:
            att3_out = F.interpolate(att3_out, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.up1(d2 + att3_out)                            # (B,  32, H/2,  W/2)

        # e0 skip at H/2: finest encoder scale, contains thin vessel edge detail.
        # Project 64→32ch to match d1, then gate and add before final upsampling.
        e0_proj  = self.proj_e0(e0)                              # (B,  32, H/2,  W/2)
        att0_out = self.att0(d1, e0_proj)                        # (B,  32, H/2,  W/2)
        if att0_out.shape[2:] != d1.shape[2:]:
            att0_out = F.interpolate(att0_out, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d0 = self.up0(d1 + att0_out)                            # (B,  16, H,    W)

        d0     = self.refine(d0)                                 # (B,  16, H,    W) — dilated refinement
        logits = self.head(d0)                                   # (B,   1, H,    W) — raw logits

        # ConvTranspose2d can produce off-by-one spatial sizes; align to input.
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        return logits


def visualize_attention_maps(model, img_rgb_np):
    """
    Extract and visualize attention maps from all 3 gates of AttentionUNet.
    Must be called after a forward pass (reads model.att{1,2,3}.last_psi).

    Args:
        model:       AttentionUNet instance
        img_rgb_np:  (H, W, 3) uint8 RGB — input image for overlay

    Returns:
        dict of {gate_view: wandb.Image} — 9 panels total
    """
    import matplotlib.cm as cm
    try:
        import wandb
    except ImportError:
        return {}

    H, W = img_rgb_np.shape[:2]
    panels = {}

    def to_heatmap(psi_np):
        """(H, W) float [0,1] → (H, W, 3) uint8 jet colormap"""
        colored = (cm.jet(psi_np)[:, :, :3] * 255).astype(np.uint8)
        return colored

    def blend(base, heatmap, alpha=0.5):
        """Blend heatmap onto base image"""
        return (alpha * heatmap.astype(np.float32) + (1 - alpha) * base.astype(np.float32)).astype(np.uint8)

    for gate_name, gate in [("attn1", model.att1), ("attn2", model.att2), ("attn3", model.att3)]:
        if not hasattr(gate, 'last_psi') or gate.last_psi is None:
            continue

        psi = gate.last_psi[0, 0].cpu().numpy()  # (H_att, W_att) float [0,1]

        # Upsample to full image resolution (bilinear)
        psi_t = torch.from_numpy(psi).unsqueeze(0).unsqueeze(0)
        psi_full = F.interpolate(psi_t, size=(H, W), mode='bilinear', align_corners=False)[0, 0].numpy()

        # View 1: raw heatmap (jet colormap)
        heatmap = to_heatmap(psi_full)
        panels[f"{gate_name}_heatmap"] = wandb.Image(
            heatmap,
            caption=f"{gate_name} α (jet: red=attended, blue=suppressed)"
        )

        # View 2: overlay on input image
        overlay = blend(img_rgb_np, heatmap)
        panels[f"{gate_name}_overlay"] = wandb.Image(
            overlay,
            caption=f"{gate_name} α overlay on input"
        )

        # View 3: psi as grayscale (white=attended, black=suppressed)
        psi_gray = (psi_full * 255).astype(np.uint8)
        psi_rgb = np.stack([psi_gray] * 3, axis=-1)
        panels[f"{gate_name}_alpha"] = wandb.Image(
            psi_rgb,
            caption=f"{gate_name} raw α [white=attended, black=suppressed]"
        )

    return panels
