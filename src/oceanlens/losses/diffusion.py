"""Diffusion and bridge losses used by OceanLens v9 variants."""

import torch
import torch.nn.functional as F


def _masked_weighted_mean(loss, mask, cfg):
    valid = mask.to(device=loss.device, dtype=loss.dtype)
    if valid.shape[1] == 1 and loss.shape[1] != 1:
        valid = valid.expand(-1, loss.shape[1], -1, -1)
    weights = getattr(getattr(cfg, "loss", None), "var_weights", None)
    if weights is not None:
        weights = torch.as_tensor(list(weights), device=loss.device, dtype=loss.dtype).view(1, -1, 1, 1)
        loss = loss * weights
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)


def edm_denoising_loss(model, target, condition, mask, cfg, return_details=False):
    """EDM denoising objective: noisy target -> clean target."""
    edm_cfg = getattr(cfg, "edm", getattr(cfg, "diffusion", None))
    sigma_data = float(getattr(edm_cfg, "sigma_data", 0.5))
    sigma_min = float(getattr(edm_cfg, "sigma_min", 0.002))
    sigma_max = float(getattr(edm_cfg, "sigma_max", 80.0))
    p_mean = float(getattr(edm_cfg, "P_mean", -1.2))
    p_std = float(getattr(edm_cfg, "P_std", 1.2))

    batch = target.shape[0]
    rnd = torch.randn(batch, device=target.device, dtype=target.dtype)
    sigma = (rnd * p_std + p_mean).exp().clamp(min=sigma_min, max=sigma_max)
    noise = torch.randn_like(target)
    sigma_view = sigma[:, None, None, None]
    x_noisy = target + sigma_view * noise

    noise_label = sigma.log()
    pred = model(x_noisy, noise_label, condition)
    weight = (sigma_view.square() + sigma_data**2) / ((sigma_view * sigma_data) ** 2)
    loss = _masked_weighted_mean((pred - target).square() * weight, mask, cfg)

    if not return_details:
        return loss
    return {
        "loss": loss,
        "x_t": x_noisy,
        "pred_x1": pred,
        "sigma": sigma,
        "target": target,
    }


def _adaptive_sigma(mu, mask, min_sigma=0.01, max_sigma=1.0):
    """Local adaptive bridge noise from deterministic field activity."""
    smooth = F.avg_pool2d(mu * mask, kernel_size=5, stride=1, padding=2)
    activity = (mu - smooth).abs()
    denom = activity.flatten(2).quantile(0.95, dim=-1).view(mu.shape[0], mu.shape[1], 1, 1).clamp(min=1.0e-6)
    sigma = (activity / denom).clamp(0.0, 1.0)
    return min_sigma + (max_sigma - min_sigma) * sigma


def diffusion_bridge_loss(model, hr, mu, condition, mask, cfg, return_details=False):
    """Stochastic interpolant bridge from adaptive noisy mu to HR."""
    bridge_cfg = getattr(cfg, "bridge", None)
    sigma_min = float(getattr(bridge_cfg, "sigma_min", 0.01))
    sigma_max = float(getattr(bridge_cfg, "sigma_max", 1.0))
    beta_min = float(getattr(bridge_cfg, "beta_min", 0.1))
    beta_max = float(getattr(bridge_cfg, "beta_max", 20.0))

    eps = torch.randn_like(hr)
    sigma_mu = _adaptive_sigma(mu.detach(), mask, min_sigma=sigma_min, max_sigma=sigma_max)
    x0 = (mu.detach() + sigma_mu * eps) * mask
    x1 = hr * mask

    batch = hr.shape[0]
    t = torch.rand(batch, device=hr.device, dtype=hr.dtype)
    t_view = t[:, None, None, None]
    beta_t = beta_min + (beta_max - beta_min) * t_view
    bridge_sigma = torch.sqrt((t_view * (1.0 - t_view)).clamp(min=0.0)) / beta_t.sqrt()
    z = torch.randn_like(hr)
    x_t = ((1.0 - t_view) * x0 + t_view * x1 + bridge_sigma * z) * mask
    v_target = (x1 - x0) * mask
    v_pred = model(x_t, t, condition)
    loss = _masked_weighted_mean((v_pred - v_target).square(), mask, cfg)

    if not return_details:
        return loss
    return {
        "loss": loss,
        "x_t": x_t,
        "v_pred": v_pred,
        "t": t,
        "pred_x1": x_t + (1.0 - t_view) * v_pred,
        "target": x1,
    }
