"""CNO loss."""

import torch


def _masked_l1(pred, target, mask):
    """Masked L1 averaged over valid ocean pixels."""
    valid = mask.to(dtype=pred.dtype)
    denom = valid.sum().clamp(min=1.0)
    return ((pred - target).abs() * valid).sum() / denom


def _valid_gradient_mask(mask):
    """Keep gradients whose two adjacent pixels are both ocean."""
    mask = mask[:, :1].to(dtype=torch.bool)
    mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
    return mask_x, mask_y


def _log_gradient_magnitude(field, mask, eps):
    """Compute log(|grad field| + eps) with land-safe finite differences."""
    dx = field[:, :, :, 1:] - field[:, :, :, :-1]
    dy = field[:, :, 1:, :] - field[:, :, :-1, :]

    # Align x and y differences on cell centers.
    dx_center = 0.5 * (dx[:, :, 1:, :] + dx[:, :, :-1, :])
    dy_center = 0.5 * (dy[:, :, :, 1:] + dy[:, :, :, :-1])
    mag = torch.sqrt(dx_center.square() + dy_center.square() + eps)

    mask_x, mask_y = _valid_gradient_mask(mask)
    valid = mask_x[:, :, 1:, :] & mask_x[:, :, :-1, :] & mask_y[:, :, :, 1:] & mask_y[:, :, :, :-1]
    return torch.log(mag + eps), valid.to(dtype=field.dtype)


def log_gradient_loss(pred_full, hr_target, ocean_mask, cfg):
    """Auxiliary loss on log gradient magnitude for selected variables."""
    grad_cfg = cfg.loss.get("log_gradient", {})
    if not grad_cfg or not grad_cfg.get("enabled", False):
        return pred_full.new_tensor(0.0)

    variables = list(cfg.variables)
    channels = [variables.index(name) for name in grad_cfg.get("variables", ["thetao"]) if name in variables]
    if not channels:
        return pred_full.new_tensor(0.0)

    eps = float(grad_cfg.get("eps", 1.0e-6))
    loss = pred_full.new_tensor(0.0)
    for channel in channels:
        pred_log_grad, valid = _log_gradient_magnitude(pred_full[:, channel : channel + 1], ocean_mask, eps)
        true_log_grad, _ = _log_gradient_magnitude(hr_target[:, channel : channel + 1], ocean_mask, eps)
        loss = loss + _masked_l1(pred_log_grad, true_log_grad, valid)

    return loss / len(channels)


def cno_loss(pred, hr_target, lr_input, ocean_mask, cfg):
    """Compute the variant-specific CNO target with masked L1."""
    if cfg.cno_mode == "direct":
        target = hr_target
    elif cfg.cno_mode == "residual":
        target = (hr_target - lr_input) * ocean_mask
    else:
        raise ValueError(f"Unknown cno_mode: {cfg.cno_mode}")

    n_ocean = ocean_mask.sum().clamp(min=1)
    loss = pred.new_tensor(0.0)
    weights = list(cfg.loss.var_weights)
    channels = pred.shape[1]

    for channel in range(channels):
        diff = (pred[:, channel] - target[:, channel]) * ocean_mask[:, 0]
        loss = loss + weights[channel] * diff.abs().sum() / n_ocean

    data_loss = loss / channels

    grad_cfg = cfg.loss.get("log_gradient", {})
    grad_weight = float(grad_cfg.get("weight", 0.0)) if grad_cfg else 0.0
    if grad_weight <= 0.0:
        return data_loss

    full_pred = pred if cfg.cno_mode == "direct" else lr_input + pred
    full_pred = full_pred * ocean_mask
    grad_loss = log_gradient_loss(full_pred, hr_target * ocean_mask, ocean_mask, cfg)
    return data_loss + grad_weight * grad_loss
