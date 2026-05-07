import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionGate(nn.Module):
    """Gating mechanism: soft attention on skip-connection features guided by decoder query."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, padding=0)
        self.W_x = nn.Conv2d(F_l, F_int, 1, padding=0)
        self.psi = nn.Conv2d(F_int, 1, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        # Resize g1 to match x spatial dimensions (handle mismatch from uneven upsampling)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.W_x(x)
        psi = self.sigmoid(self.psi(self.relu(g1 + x1)))
        self.last_psi = psi.detach()  # Save for visualization (no grad overhead)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention UNet: ResNet34 encoder + UNet decoder with attention gates on skip connections.
    Addresses blurry region suppression via attention gates (learned soft masking).
    3-channel input [gray, grad_mag, sharpness] → logits.
    Fully trainable encoder (no frozen backbone).
    """
    def __init__(self):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for AttentionUNet. Install: pip install timm")

        self.encoder = timm.create_model('resnet34', pretrained=True,
                                         features_only=True, in_chans=3,
                                         out_indices=(0, 1, 2, 3, 4))
        # ResNet34 output channels: [64, 64, 128, 256, 512]
        # Spatial: [H/4, H/8, H/16, H/32, H/32] (note: layer4 also at 1/32)

        self.att1 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att3 = AttentionGate(64, 64, 32)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up0 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # larger kernel for better upsampling

        # Dilated refinement: larger receptive field to fill vessel holes
        self.refine = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, dilation=2, padding=4),  # 5×5 with dilation=2 → 9×9 receptive field
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, dilation=1, padding=1),  # 3×3 local refinement
        )

        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, use_graph=False, sharpness=None, grad_mag=None, sharp_gate=False, use_focus_gate=False):
        """
        x: (B, 1, H, W) grayscale image
        sharpness, grad_mag: optional (B, 1, H, W) feature maps
        Returns: (logits, None, None) — compatible 3-tuple for loss computation
        """
        # Construct 3-channel input: [gray, grad_mag, sharpness] if available, else repeat grayscale
        if sharpness is not None and grad_mag is not None:
            x_enc = torch.cat([x, grad_mag, sharpness], dim=1)  # (B, 3, H, W)
        else:
            x_enc = x.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W) by repeating grayscale

        e0, e1, e2, e3, e4 = self.encoder(x_enc)

        d4 = self.up4(e4)
        att1_out = self.att1(d4, e3)
        # Ensure spatial match before adding
        if att1_out.shape[2:] != d4.shape[2:]:
            att1_out = F.interpolate(att1_out, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up3(d4 + att1_out)

        att2_out = self.att2(d3, e2)
        if att2_out.shape[2:] != d3.shape[2:]:
            att2_out = F.interpolate(att2_out, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up2(d3 + att2_out)

        att3_out = self.att3(d2, e1)
        if att3_out.shape[2:] != d2.shape[2:]:
            att3_out = F.interpolate(att3_out, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.up1(d2 + att3_out)

        d0 = self.up0(d1)  # 1/2 → 1/1 (full resolution)
        d0 = self.refine(d0)  # dilated convolutions to fill vessel holes
        logits = self.head(d0)

        # Ensure output matches input spatial dimensions (ConvTranspose2d can produce off-by-one sizes)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        return logits, None, None


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
