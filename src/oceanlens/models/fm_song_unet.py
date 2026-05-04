"""Song/NCSN++ style U-Net backbones for diffusion and bridge objectives."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fm_unet import SelfAttention, SinusoidalTimeEmbedding


def _group_norm(channels):
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SongResBlock(nn.Module):
    """NCSN++-style residual block with noise/time FiLM modulation."""

    def __init__(self, in_channels, out_channels, emb_dim, dropout=0.0):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return (residual + h) / math.sqrt(2.0)


class SongUNetPosEmbd(nn.Module):
    """Compact PhysicsNeMo-style SongUNetPosEmbd compatible with OceanLens."""

    def __init__(
        self,
        x_channels=5,
        condition_channels=13,
        out_channels=5,
        img_resolution=256,
        model_channels=128,
        channel_mult=None,
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=None,
        time_dim=256,
        gridtype="learnable",
        n_grid_channels=4,
        dropout=0.0,
        attn_heads=8,
    ):
        super().__init__()
        channel_mult = list(channel_mult or [1, 2, 2, 2])
        attn_resolutions = set(attn_resolutions or [16, 32])
        self.img_resolution = int(img_resolution)
        self.gridtype = gridtype
        self.n_grid_channels = int(n_grid_channels)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        emb_dim = int(time_dim * channel_mult_emb)
        self.emb = nn.Sequential(nn.Linear(time_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        input_channels = x_channels + condition_channels + self.n_grid_channels
        channels = [int(model_channels * mult) for mult in channel_mult]
        self.conv_in = nn.Conv2d(input_channels, channels[0], 3, padding=1)

        if self.n_grid_channels > 0:
            if gridtype == "learnable":
                self.grid = nn.Parameter(torch.randn(1, self.n_grid_channels, self.img_resolution, self.img_resolution) * 0.02)
            elif gridtype == "fixed":
                self.register_buffer("grid", self._fixed_grid(self.img_resolution, self.img_resolution, self.n_grid_channels), persistent=False)
            else:
                raise ValueError(f"Unsupported gridtype: {gridtype}")
        else:
            self.grid = None

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        resolution = self.img_resolution
        self.skip_channels = []
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(SongResBlock(current, out_ch, emb_dim, dropout=dropout))
                current = out_ch
                if resolution in attn_resolutions:
                    blocks.append(SelfAttention(current, n_heads=attn_heads))
            self.down_blocks.append(blocks)
            self.skip_channels.append(current)
            if level != len(channels) - 1:
                self.downsamples.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
                resolution = math.ceil(resolution / 2)

        self.mid = nn.ModuleList(
            [
                SongResBlock(current, current, emb_dim, dropout=dropout),
                SelfAttention(current, n_heads=attn_heads),
                SongResBlock(current, current, emb_dim, dropout=dropout),
            ]
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, skip_ch in reversed(list(enumerate(self.skip_channels))):
            blocks = nn.ModuleList()
            blocks.append(SongResBlock(current + skip_ch, skip_ch, emb_dim, dropout=dropout))
            current = skip_ch
            resolution = max(1, self.img_resolution // (2 ** level))
            for _ in range(num_blocks):
                blocks.append(SongResBlock(current, current, emb_dim, dropout=dropout))
                if resolution in attn_resolutions:
                    blocks.append(SelfAttention(current, n_heads=attn_heads))
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(nn.ConvTranspose2d(current, current, 4, stride=2, padding=1))

        self.out = nn.Sequential(_group_norm(current), nn.SiLU(), nn.Conv2d(current, out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _fixed_grid(height, width, channels):
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        parts = [yy, xx, torch.sin(math.pi * xx), torch.cos(math.pi * xx)]
        grid = torch.stack(parts[:channels], dim=0)
        if channels > grid.shape[0]:
            grid = F.pad(grid, (0, 0, 0, 0, 0, channels - grid.shape[0]))
        return grid[None]

    def _grid_for(self, x):
        if self.grid is None:
            return None
        grid = self.grid
        if grid.shape[-2:] != x.shape[-2:]:
            grid = F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return grid.expand(x.shape[0], -1, -1, -1).to(device=x.device, dtype=x.dtype)

    def forward(self, x_t, noise_labels, condition):
        emb = self.emb(self.time_embed(noise_labels))
        parts = [x_t, condition]
        grid = self._grid_for(x_t)
        if grid is not None:
            parts.append(grid)
        x = self.conv_in(torch.cat(parts, dim=1))

        skips = []
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, SongResBlock):
                    x = block(x, emb)
                else:
                    x = block(x)
            skips.append(x)
            if level < len(self.downsamples):
                x = self.downsamples[level](x)

        for block in self.mid:
            if isinstance(block, SongResBlock):
                x = block(x, emb)
            else:
                x = block(x)

        for level, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                if isinstance(block, SongResBlock):
                    x = block(x, emb)
                else:
                    x = block(x)
            if level < len(self.upsamples):
                x = self.upsamples[level](x)

        return self.out(x)
