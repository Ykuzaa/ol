"""
Convolutional Neural Operator (CNO) - Vanilla 2D
Adapted from camlab-ethz/ConvolutionalNeuralOperator (CNO2d_simplified)
U-Net operator architecture for ocean super-resolution 1/4 -> 1/12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BandlimitedActivation(nn.Module):
    """Upsample -> activation -> downsample."""

    def __init__(self, up_factor=2):
        super().__init__()
        self.up_factor = up_factor
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(h * self.up_factor, w * self.up_factor), mode="bilinear", align_corners=False)
        x = self.act(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x


class CNOBlock(nn.Module):
    """Conv -> bandlimited activation -> Conv."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.act1 = BandlimitedActivation()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.act2 = BandlimitedActivation()

    def forward(self, x):
        residual = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x + residual


class CNODownBlock(nn.Module):
    """Downsample spatially by 2."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad)
        self.act = BandlimitedActivation()

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
        x = self.act(self.conv(x))
        return x


class CNOUpBlock(nn.Module):
    """Upsample spatially by 2 and merge skip."""

    def __init__(self, in_ch, skip_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, padding=pad)
        self.act = BandlimitedActivation()

    def forward(self, x, skip):
        h, w = skip.shape[-2:]
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.conv(x))
        return x


class CNO2d(nn.Module):
    """CNO for deterministic super-resolution."""

    def __init__(self, in_channels=5, out_channels=5, hidden_channels=None, n_res_blocks=2, kernel_size=3):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]

        self.n_levels = len(hidden_channels)
        pad = kernel_size // 2
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size, padding=pad),
            BandlimitedActivation(),
        )

        self.encoders = nn.ModuleList()
        self.encoder_res = nn.ModuleList()
        for i in range(self.n_levels - 1):
            self.encoder_res.append(nn.Sequential(*[CNOBlock(hidden_channels[i], kernel_size) for _ in range(n_res_blocks)]))
            self.encoders.append(CNODownBlock(hidden_channels[i], hidden_channels[i + 1], kernel_size))

        self.bottleneck = nn.Sequential(*[CNOBlock(hidden_channels[-1], kernel_size) for _ in range(n_res_blocks)])

        self.decoders = nn.ModuleList()
        self.decoder_res = nn.ModuleList()
        for i in range(self.n_levels - 1, 0, -1):
            self.decoders.append(CNOUpBlock(hidden_channels[i], hidden_channels[i - 1], hidden_channels[i - 1], kernel_size))
            self.decoder_res.append(nn.Sequential(*[CNOBlock(hidden_channels[i - 1], kernel_size) for _ in range(n_res_blocks)]))

        self.proj = nn.Conv2d(hidden_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.lift(x)
        skips = []
        for res_block, down_block in zip(self.encoder_res, self.encoders):
            x = res_block(x)
            skips.append(x)
            x = down_block(x)

        x = self.bottleneck(x)

        for up_block, res_block, skip in zip(self.decoders, self.decoder_res, reversed(skips)):
            x = up_block(x, skip)
            x = res_block(x)

        return self.proj(x)
