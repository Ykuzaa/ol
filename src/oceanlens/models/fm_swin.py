"""Lightweight Swin U-Net backbone for residual Flow Matching."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fm_unet import SinusoidalTimeEmbedding


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension."""

    def __init__(self, dim, eps=1.0e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU MLP used in modern transformer blocks."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        value, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(value * F.silu(gate))


def _pad_to_window(x, window_size):
    batch, height, width, channels = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return x, height, width


def _window_partition(x, window_size):
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, channels)


def _window_reverse(windows, window_size, batch, height, width, channels):
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, channels)


class WindowAttention(nn.Module):
    """Window self-attention with QK-Norm."""

    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flat = coords.flatten(1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        relative_index = relative.sum(-1)
        self.register_buffer("relative_position_index", relative_index, persistent=False)
        self.relative_position_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(self, x):
        batch_windows, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias[self.relative_position_index.view(-1)]
        bias = bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attn = (attn + bias.unsqueeze(0)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch_windows, tokens, dim)
        return self.proj(out)


class SwinBlock(nn.Module):
    """Shifted-window transformer block with time FiLM and scaled residuals."""

    def __init__(self, dim, num_heads, window_size, shift_size, time_dim):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = RMSNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * 4)
        self.time_proj1 = nn.Linear(time_dim, dim * 2)
        self.time_proj2 = nn.Linear(time_dim, dim * 2)
        self.res_scale = math.sqrt(0.5)

    def _modulate(self, x, proj, t_emb):
        scale, shift = proj(t_emb)[:, None, None, :].chunk(2, dim=-1)
        return x * (1.0 + scale) + shift

    def forward(self, x, t_emb):
        batch, channels, height, width = x.shape
        h = x.permute(0, 2, 3, 1).contiguous()
        residual = h
        h = self._modulate(self.norm1(h), self.time_proj1, t_emb)
        h, orig_h, orig_w = _pad_to_window(h, self.window_size)
        if self.shift_size > 0:
            h = torch.roll(h, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        _, pad_h, pad_w, _ = h.shape
        windows = _window_partition(h, self.window_size)
        windows = self.attn(windows)
        h = _window_reverse(windows, self.window_size, batch, pad_h, pad_w, channels)
        if self.shift_size > 0:
            h = torch.roll(h, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        h = h[:, :orig_h, :orig_w, :]
        h = (residual + h) * self.res_scale

        residual = h
        h = self._modulate(self.norm2(h), self.time_proj2, t_emb)
        h = (residual + self.mlp(h)) * self.res_scale
        return h.permute(0, 3, 1, 2).contiguous()


class PatchMerging(nn.Module):
    """Downsample by 2 and double channel count."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.proj(x)


class PatchExpanding(nn.Module):
    """Upsample by 2."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x, size):
        x = self.proj(x)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


class BottleneckCrossAttention(nn.Module):
    """Single cross-attention block for dynamic conditioning at bottleneck resolution."""

    def __init__(self, dim, cond_channels, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cond_proj = nn.Conv2d(cond_channels, dim, kernel_size=3, padding=1)
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.res_scale = math.sqrt(0.5)

    def forward(self, x, dynamic_condition):
        batch, channels, height, width = x.shape
        cond = F.interpolate(dynamic_condition, size=(height, width), mode="bilinear", align_corners=False)
        cond = self.cond_proj(cond)
        q_tokens = x.flatten(2).transpose(1, 2)
        kv_tokens = cond.flatten(2).transpose(1, 2)
        q_tokens = self.norm_q(q_tokens)
        kv_tokens = self.norm_kv(kv_tokens)
        q = self.q(q_tokens).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(kv_tokens).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(kv_tokens).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, height * width, channels)
        out = self.proj(out).transpose(1, 2).reshape(batch, channels, height, width)
        return (x + out) * self.res_scale


class FlowMatchingSwinUNet(nn.Module):
    """Swin U-Net that predicts the Flow Matching velocity field."""

    def __init__(
        self,
        x_channels=5,
        condition_channels=17,
        dynamic_channels=5,
        out_channels=5,
        patch_size=4,
        embed_dim=96,
        depths=None,
        heads=None,
        window_size=8,
        time_dim=256,
        cross_attn_bottleneck=True,
    ):
        super().__init__()
        depths = depths or [2, 2, 4, 2]
        heads = heads or [3, 6, 12, 6]
        if len(depths) != 4 or len(heads) != 4:
            raise ValueError("FlowMatchingSwinUNet expects depths and heads with 4 entries")
        self.dynamic_channels = dynamic_channels
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        in_channels = x_channels + condition_channels
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.stage1 = self._make_stage(embed_dim, depths[0], heads[0], window_size, time_dim)
        self.down1 = PatchMerging(embed_dim, embed_dim * 2)
        self.stage2 = self._make_stage(embed_dim * 2, depths[1], heads[1], window_size, time_dim)
        self.down2 = PatchMerging(embed_dim * 2, embed_dim * 4)
        self.stage3 = self._make_stage(embed_dim * 4, depths[2], heads[2], window_size, time_dim)

        self.cross_attn = (
            BottleneckCrossAttention(embed_dim * 4, dynamic_channels, heads[2])
            if cross_attn_bottleneck and dynamic_channels > 0
            else None
        )

        self.up2 = PatchExpanding(embed_dim * 4, embed_dim * 2)
        self.skip2 = nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=1)
        self.dec2 = self._make_stage(embed_dim * 2, depths[3], heads[3], window_size, time_dim)
        self.up1 = PatchExpanding(embed_dim * 2, embed_dim)
        self.skip1 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        self.dec1 = self._make_stage(embed_dim, depths[0], heads[0], window_size, time_dim)
        self.out_norm = nn.GroupNorm(8, embed_dim)
        self.out_act = nn.GELU()
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)
        nn.init.zeros_(self.patch_unembed.weight)
        nn.init.zeros_(self.patch_unembed.bias)

    @staticmethod
    def _make_stage(dim, depth, heads, window_size, time_dim):
        blocks = []
        for index in range(depth):
            shift = 0 if index % 2 == 0 else window_size // 2
            blocks.append(SwinBlock(dim, heads, window_size, shift, time_dim))
        return nn.ModuleList(blocks)

    @staticmethod
    def _run_stage(x, stage, t_emb):
        for block in stage:
            x = block(x, t_emb)
        return x

    def forward(self, x_t, t, condition):
        """Predict velocity from current state, time, and conditioning."""
        input_size = x_t.shape[-2:]
        t_emb = self.time_embed(t)
        dynamic_condition = condition[:, : self.dynamic_channels] if self.dynamic_channels > 0 else None
        x = torch.cat([x_t, condition], dim=1)
        x = self.patch_embed(x)

        x1 = self._run_stage(x, self.stage1, t_emb)
        x2 = self._run_stage(self.down1(x1), self.stage2, t_emb)
        x3 = self._run_stage(self.down2(x2), self.stage3, t_emb)
        if self.cross_attn is not None and dynamic_condition is not None:
            x3 = self.cross_attn(x3, dynamic_condition)

        x = self.up2(x3, x2.shape[-2:])
        x = self.skip2(torch.cat([x, x2], dim=1))
        x = self._run_stage(x, self.dec2, t_emb)
        x = self.up1(x, x1.shape[-2:])
        x = self.skip1(torch.cat([x, x1], dim=1))
        x = self._run_stage(x, self.dec1, t_emb)
        x = self.patch_unembed(self.out_act(self.out_norm(x)))
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
