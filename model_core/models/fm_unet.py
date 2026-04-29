"""
Flow Matching U-Net for stochastic residual generation.
Conditioned on CNO output + LR input + land mask.
Learns velocity field along a linear OT path.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Encode timestep t in [0,1]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class SelfAttention(nn.Module):
    """Self-attention at bottleneck resolution."""

    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        assert channels % n_heads == 0
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch, channels, height, width = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(batch, 3, self.n_heads, self.head_dim, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        return residual + self.proj(out)


class ResBlock(nn.Module):
    """ResBlock with FiLM time injection."""

    def __init__(self, channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.time_proj = nn.Linear(time_dim, channels * 2)

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        scale_shift = self.time_proj(t_emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class DownBlock(nn.Module):
    """Downsample and apply residual blocks."""

    def __init__(self, in_ch, out_ch, time_dim, n_res=2):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock(out_ch, time_dim) for _ in range(n_res)])
        self.conv_down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.conv_down(x)
        for block in self.res_blocks:
            x = block(x, t_emb)
        return x


class UpBlock(nn.Module):
    """Upsample and merge skip."""

    def __init__(self, in_ch, out_ch, time_dim, n_res=2):
        super().__init__()
        self.conv_up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(out_ch, time_dim) for _ in range(n_res)])
        self.channel_reduce = nn.Conv2d(out_ch * 2, out_ch, 1)

    def forward(self, x, skip, t_emb):
        x = self.conv_up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.channel_reduce(x)
        for block in self.res_blocks:
            x = block(x, t_emb)
        return x


class FlowMatchingUNet(nn.Module):
    """U-Net that predicts the velocity field."""

    def __init__(self, in_channels=16, out_channels=5, hidden_channels=None, time_dim=256, n_res=2, attn_heads=8):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.conv_in = nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1)

        self.encoders = nn.ModuleList()
        self.encoder_skips = nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            self.encoder_skips.append(nn.Sequential(*[ResBlock(hidden_channels[i], time_dim) for _ in range(n_res)]))
            self.encoders.append(DownBlock(hidden_channels[i], hidden_channels[i + 1], time_dim, n_res))

        bottleneck_blocks = [ResBlock(hidden_channels[-1], time_dim)]
        if attn_heads > 0:
            bottleneck_blocks.append(SelfAttention(hidden_channels[-1], n_heads=attn_heads))
        bottleneck_blocks.append(ResBlock(hidden_channels[-1], time_dim))
        if attn_heads > 0:
            bottleneck_blocks.append(SelfAttention(hidden_channels[-1], n_heads=attn_heads))
        self.bottleneck = nn.ModuleList(bottleneck_blocks)

        self.decoders = nn.ModuleList()
        for i in range(len(hidden_channels) - 1, 0, -1):
            self.decoders.append(UpBlock(hidden_channels[i], hidden_channels[i - 1], time_dim, n_res))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, hidden_channels[0]),
            nn.GELU(),
            nn.Conv2d(hidden_channels[0], out_channels, 1),
        )

    def forward(self, x_t, t, condition):
        """Predict the velocity field."""
        t_emb = self.time_embed(t)
        x = torch.cat([x_t, condition], dim=1)
        x = self.conv_in(x)

        skips = []
        for skip_block, down_block in zip(self.encoder_skips, self.encoders):
            for block in skip_block:
                x = block(x, t_emb)
            skips.append(x)
            x = down_block(x, t_emb)

        for block in self.bottleneck:
            if isinstance(block, ResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)

        for up_block, skip in zip(self.decoders, reversed(skips)):
            x = up_block(x, skip, t_emb)

        return self.conv_out(x)
