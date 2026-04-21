"""Variant-specific routing."""

import torch


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
    cond_map = {"mu": mu, "lr": lr, "mask": mask}
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
