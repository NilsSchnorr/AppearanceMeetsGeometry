"""
U-Net model definition for the AppearanceMeetsGeometry pipeline.

This is a verbatim extraction of the MultiUNet architecture used in the original
training notebooks (02_MachineLearning/*_PytorchUNET.ipynb), with two additions:
  - the channel progression is parameterizable via `width_mult` so that the
    Step-2 architecture sweep (slim / base / wide) can be driven from config
    without code changes. width_mult="base" reproduces the original exactly.
  - normalization inside DoubleConv is selectable via `norm` (Stage 1 recipe
    screen). norm="none" (default) inserts nn.Identity placeholders, which hold
    no parameters, so the state_dict is byte-identical to the original — old
    checkpoints keep loading cleanly. norm="groupnorm" inserts
    GroupNorm(8 groups) after each convolution, before the activation
    (Conv -> GN -> ReLU). GroupNorm is batch-size independent, which suits the
    small-batch regime here better than BatchNorm.

The original (base) encoder progression is [16, 32, 64, 128, 256].
  slim -> [8, 16, 32, 64, 128]
  base -> [16, 32, 64, 128, 256]   (original; default)
  wide -> [32, 64, 128, 256, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Base channel progression of the original notebooks. Dropout rates are tied to
# encoder depth exactly as in the originals.
_BASE_WIDTHS = [16, 32, 64, 128, 256]
_DROPOUTS = [0.1, 0.1, 0.2, 0.2, 0.3]  # inc, down1, down2, down3, down4

_WIDTH_MULTIPLIERS = {
    "slim": 0.5,
    "base": 1.0,
    "wide": 2.0,
}


def resolve_widths(width_mult="base"):
    """Return the encoder channel list for a given width setting."""
    if width_mult not in _WIDTH_MULTIPLIERS:
        raise ValueError(
            f"Unknown width_mult={width_mult!r}; expected one of {list(_WIDTH_MULTIPLIERS)}"
        )
    m = _WIDTH_MULTIPLIERS[width_mult]
    return [int(round(w * m)) for w in _BASE_WIDTHS]


NORM_CHOICES = ("none", "groupnorm")


def _make_norm(norm, channels):
    """Normalization layer factory for DoubleConv.

    "none"      -> nn.Identity(): no parameters, no buffers, so the state_dict
                   stays byte-identical to the original architecture.
    "groupnorm" -> nn.GroupNorm with 8 groups (all width variants use channel
                   counts divisible by 8: slim 8..128, base 16..256, wide
                   32..512); falls back to 1 group if ever indivisible.
    """
    if norm == "none":
        return nn.Identity()
    if norm == "groupnorm":
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown norm={norm!r}; expected one of {NORM_CHOICES}")


class DoubleConv(nn.Module):
    """Double convolution block (verbatim from the original notebooks; Stage 1
    adds optional normalization: Conv -> Norm -> ReLU, with norm="none"
    reproducing the original exactly)."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1, norm="none"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm1 = _make_norm(norm, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm2 = _make_norm(norm, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1, norm="none"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate, norm=norm),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with bilinear interpolation (verbatim from the originals)."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1, norm="none"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_after_up = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        # Norm is scoped to DoubleConv (the screened variable); conv_after_up
        # is deliberately left untouched.
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_after_up(x1)

        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MultiUNet(nn.Module):
    """
    Multi-channel U-Net. Identical to the original notebook model when
    width_mult="base" and norm="none". n_channels selects the variant (3 / 4 / 7).
    """

    def __init__(self, n_channels=7, n_classes=4, width_mult="base", norm="none"):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.width_mult = width_mult
        self.norm = norm

        w = resolve_widths(width_mult)  # [w0, w1, w2, w3, w4]
        d = _DROPOUTS

        # Encoder
        self.inc = DoubleConv(n_channels, w[0], d[0], norm=norm)
        self.down1 = Down(w[0], w[1], d[1], norm=norm)
        self.down2 = Down(w[1], w[2], d[2], norm=norm)
        self.down3 = Down(w[2], w[3], d[3], norm=norm)
        self.down4 = Down(w[3], w[4], d[4], norm=norm)

        # Decoder (dropout rates mirror the original up1..up4 = 0.2,0.2,0.1,0.1)
        self.up1 = Up(w[4], w[3], 0.2, norm=norm)
        self.up2 = Up(w[3], w[2], 0.2, norm=norm)
        self.up3 = Up(w[2], w[1], 0.1, norm=norm)
        self.up4 = Up(w[1], w[0], 0.1, norm=norm)
        self.outc = nn.Conv2d(w[0], n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def initialize_weights(model):
    """Kaiming initialization (verbatim from the originals)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
