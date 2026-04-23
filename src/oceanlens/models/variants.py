"""Variant-specific routing."""

import torch


def _thetao_gradient(field: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return a one-channel gradient magnitude for thetao."""
    theta = field[:, 0:1]
    grad_y = torch.zeros_like(theta)
    grad_x = torch.zeros_like(theta)

    grad_y[:, :, 1:-1, :] = 0.5 * (theta[:, :, 2:, :] - theta[:, :, :-2, :])
    grad_y[:, :, 0:1, :] = theta[:, :, 1:2, :] - theta[:, :, 0:1, :]
    grad_y[:, :, -1:, :] = theta[:, :, -1:, :] - theta[:, :, -2:-1, :]

    grad_x[:, :, :, 1:-1] = 0.5 * (theta[:, :, :, 2:] - theta[:, :, :, :-2])
    grad_x[:, :, :, 0:1] = theta[:, :, :, 1:2] - theta[:, :, :, 0:1]
    grad_x[:, :, :, -1:] = theta[:, :, :, -1:] - theta[:, :, :, -2:-1]

    grad = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12)
    if mask is not None:
        grad = grad * mask
    return grad


def build_mu(cno_output: torch.Tensor, lr: torch.Tensor, cfg) -> torch.Tensor:
    """Build the deterministic field."""
    if cfg.cno_mode == "direct":
        return cno_output
    if cfg.cno_mode == "residual":
        return lr + cno_output
    raise ValueError(f"Unknown cno_mode: {cfg.cno_mode}")


def build_fm_condition(mu, lr: torch.Tensor, mask: torch.Tensor, cfg) -> torch.Tensor:
    """Build FM conditioning channels.

    The model concatenates x_t internally.
    """
    parts = []
    cond_map = {
        "mu": mu,
        "lr": lr,
        "mask": mask,
        "grad_mu": None if mu is None else _thetao_gradient(mu, mask),
        "grad_lr": _thetao_gradient(lr, mask),
    }
    for key in cfg.fm_condition:
        tensor = cond_map[key]
        if tensor is None:
            raise ValueError(f"Condition '{key}' is None for variant {cfg.variant}")
        parts.append(tensor)
    return torch.cat(parts, dim=1)


def compute_fm_target(hr: torch.Tensor, mu, lr: torch.Tensor, ocean_mask: torch.Tensor, cfg) -> torch.Tensor:
    """Build the target field for FM."""
    if getattr(cfg, "fm_target", None) == "residual_lr":
        return (hr - lr) * ocean_mask
    if cfg.variant == "ablation":
        return hr * ocean_mask
    return (hr - mu) * ocean_mask
