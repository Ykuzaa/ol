"""Pixel-space Diffusion Transformer backbone for residual Flow Matching."""

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
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        value, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(value * F.silu(gate))


def _modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _rotate_pairs(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_axis(x, coords, inv_freq):
    angles = coords[:, None].to(dtype=torch.float32) * inv_freq[None, :]
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(device=x.device, dtype=x.dtype)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(device=x.device, dtype=x.dtype)
    return x * cos[None, None, :, :] + _rotate_pairs(x) * sin[None, None, :, :]


class Rotary2D(nn.Module):
    """2D rotary positional embedding for image tokens."""

    def __init__(self, head_dim):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"RoPE 2D requires head_dim divisible by 4, got {head_dim}")
        axis_dim = head_dim // 4
        inv_freq = 1.0 / (10000 ** (torch.arange(axis_dim, dtype=torch.float32) / axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.axis_width = axis_dim * 2

    def forward(self, q, k, height, width):
        y, x = torch.meshgrid(
            torch.arange(height, device=q.device),
            torch.arange(width, device=q.device),
            indexing="ij",
        )
        y = y.flatten()
        x = x.flatten()

        y_slice = slice(0, self.axis_width)
        x_slice = slice(self.axis_width, self.axis_width * 2)
        q_y = _apply_rope_axis(q[..., y_slice], y, self.inv_freq)
        k_y = _apply_rope_axis(k[..., y_slice], y, self.inv_freq)
        q_x = _apply_rope_axis(q[..., x_slice], x, self.inv_freq)
        k_x = _apply_rope_axis(k[..., x_slice], x, self.inv_freq)
        q = torch.cat([q_y, q_x, q[..., self.axis_width * 2 :]], dim=-1)
        k = torch.cat([k_y, k_x, k[..., self.axis_width * 2 :]], dim=-1)
        return q, k


class DiTBlock(nn.Module):
    """DiT block with QK-Norm, optional 2D RoPE, SwiGLU and AdaLN-Zero."""

    def __init__(self, dim, num_heads, mlp_ratio, cond_dim, rope_2d=True, qk_norm=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = Rotary2D(self.head_dim) if rope_2d else None
        self.qk_norm = qk_norm

        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond, height, width):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(6, dim=-1)

        residual = x
        h = _modulate(self.norm1(x), shift_attn, scale_attn)
        batch, tokens, dim = h.shape
        qkv = h.reshape(batch * tokens, dim)
        qkv = self.qkv(qkv).view(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if self.rope is not None:
            q, k = self.rope(q, k, height, width)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        h = self.proj(h)
        x = residual + gate_attn.unsqueeze(1) * h

        residual = x
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return residual + gate_mlp.unsqueeze(1) * h


class FlowMatchingDiT(nn.Module):
    """Pixel-space DiT that predicts the Flow Matching velocity field."""

    def __init__(
        self,
        x_channels=5,
        condition_channels=14,
        out_channels=5,
        patch_size=4,
        embed_dim=384,
        depth=12,
        heads=8,
        mlp_ratio=4.0,
        time_dim=256,
        qk_norm=True,
        rope_2d=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.cond_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(condition_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.patch_embed = nn.Conv2d(
            x_channels + condition_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    cond_dim=time_dim,
                    rope_2d=rope_2d,
                    qk_norm=qk_norm,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = RMSNorm(embed_dim)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, embed_dim * 2))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.unpatchify = nn.ConvTranspose2d(
            embed_dim,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        nn.init.zeros_(self.unpatchify.weight)
        nn.init.zeros_(self.unpatchify.bias)

    def _pad_input(self, x):
        height, width = x.shape[-2:]
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, height, width

    def forward(self, x_t, t, condition):
        cond = self.time_embed(t) + self.cond_pool(condition)
        x = torch.cat([x_t, condition], dim=1)
        x, orig_h, orig_w = self._pad_input(x)
        x = self.patch_embed(x)
        batch, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)

        for block in self.blocks:
            x = block(x, cond, height, width)

        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.unpatchify(x)
        return x[..., :orig_h, :orig_w]
