"""Flow Matching loss."""

import torch
from scipy.optimize import linear_sum_assignment


def _expand_mask(mask, target):
    """Broadcast a BCHW ocean mask to the target channels."""
    if mask is None:
        return None
    mask = mask.to(device=target.device, dtype=target.dtype)
    if mask.shape[1] == 1 and target.shape[1] != 1:
        mask = mask.expand(-1, target.shape[1], -1, -1)
    return mask


def _variable_weights(cfg, target):
    weights = getattr(getattr(cfg, "loss", {}), "var_weights", None)
    if weights is None:
        return None
    weights = torch.as_tensor(list(weights), device=target.device, dtype=target.dtype)
    if weights.numel() != target.shape[1]:
        raise ValueError(f"Expected {target.shape[1]} FM variable weights, got {weights.numel()}")
    return weights.view(1, -1, 1, 1)


def _match_minibatch_ot(x0, x1, mask=None):
    """Match Gaussian samples to targets with mini-batch squared-cost OT."""
    batch_size = x0.shape[0]
    if batch_size <= 1:
        return x0

    ocean_mask = _expand_mask(mask, x1)
    with torch.no_grad():
        if ocean_mask is None:
            x0_flat = x0.reshape(batch_size, -1)
            x1_flat = x1.reshape(batch_size, -1)
            cost = torch.cdist(x1_flat.float(), x0_flat.float(), p=2).square()
        else:
            cost_rows = []
            for index in range(batch_size):
                valid = ocean_mask[index : index + 1]
                target = x1[index : index + 1]
                diff = (x0 - target) * valid
                denom = valid.sum().clamp(min=1.0)
                cost_rows.append(diff.float().square().reshape(batch_size, -1).sum(dim=1) / denom.float())
            cost = torch.stack(cost_rows, dim=0)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        order = torch.empty(batch_size, dtype=torch.long, device=x0.device)
        order[torch.as_tensor(row_ind, dtype=torch.long, device=x0.device)] = torch.as_tensor(
            col_ind,
            dtype=torch.long,
            device=x0.device,
        )
    return x0.index_select(0, order)


def _sample_time(batch_size, device, cfg=None):
    fm_cfg = getattr(cfg, "fm", {}) if cfg is not None else {}
    mode = getattr(fm_cfg, "t_sampling", "uniform")
    eps = float(getattr(fm_cfg, "t_eps", 1.0e-5))

    if mode == "uniform":
        return torch.rand(batch_size, device=device) * (1.0 - 2.0 * eps) + eps
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(batch_size, device=device)).clamp(eps, 1.0 - eps)
    raise ValueError(f"Unknown FM t_sampling: {mode}")


def flow_matching_loss(fm_model, x1, condition, mask=None, cfg=None):
    """Compute linear conditional Flow Matching on ocean pixels."""
    batch_size = x1.shape[0]
    t = _sample_time(batch_size, x1.device, cfg=cfg)
    x0 = torch.randn_like(x1)
    coupling = getattr(getattr(cfg, "fm", {}), "coupling", "independent") if cfg is not None else "independent"
    if coupling == "minibatch_ot":
        x0 = _match_minibatch_ot(x0, x1, mask=mask)
    elif coupling != "independent":
        raise ValueError(f"Unknown FM coupling: {coupling}")

    t_expand = t[:, None, None, None]
    x_t = (1 - t_expand) * x0 + t_expand * x1
    v_target = x1 - x0
    v_pred = fm_model(x_t, t, condition)

    loss = (v_pred - v_target).square()
    weights = _variable_weights(cfg, loss) if cfg is not None else None
    if weights is not None:
        loss = loss * weights

    ocean_mask = _expand_mask(mask, loss)
    if ocean_mask is None:
        return loss.mean()

    denom = ocean_mask.sum().clamp(min=1.0)
    return (loss * ocean_mask).sum() / denom
