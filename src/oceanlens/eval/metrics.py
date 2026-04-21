"""Evaluation metrics used during validation and reports."""

import torch


def _masked_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return valid ocean values for a BCHW tensor."""
    if mask.shape[1] == 1 and tensor.shape[1] != 1:
        mask = mask.expand(-1, tensor.shape[1], -1, -1)
    valid = mask.to(dtype=torch.bool)
    return tensor[valid]


def compute_per_variable_metrics(
    pred: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    variables,
):
    """Compute MAE, RMSE and correlation for each variable."""
    results = {}
    for channel, name in enumerate(variables):
        pred_values = _masked_values(pred[:, channel : channel + 1], mask)
        truth_values = _masked_values(truth[:, channel : channel + 1], mask)

        if pred_values.numel() == 0:
            zero = pred.new_tensor(0.0)
            results[name] = {"mae": zero, "rmse": zero, "corr": zero}
            continue

        diff = pred_values - truth_values
        mae = diff.abs().mean()
        rmse = torch.sqrt(diff.square().mean())

        pred_centered = pred_values - pred_values.mean()
        truth_centered = truth_values - truth_values.mean()
        denom = torch.sqrt(pred_centered.square().sum() * truth_centered.square().sum()).clamp(min=1.0e-12)
        corr = (pred_centered * truth_centered).sum() / denom

        results[name] = {"mae": mae, "rmse": rmse, "corr": corr}

    return results


def compute_all_skill_scores(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor):
    """Compute scalar skill scores for numpy-like or torch tensors."""
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)
    if not isinstance(truth, torch.Tensor):
        truth = torch.as_tensor(truth)
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)

    pred_values = _masked_values(pred.unsqueeze(0) if pred.ndim == 3 else pred, mask.unsqueeze(0) if mask.ndim == 2 else mask)
    truth_values = _masked_values(
        truth.unsqueeze(0) if truth.ndim == 3 else truth,
        mask.unsqueeze(0) if mask.ndim == 2 else mask,
    )

    diff = pred_values - truth_values
    mae = diff.abs().mean()
    rmse = torch.sqrt(diff.square().mean())
    bias = diff.mean()

    pred_centered = pred_values - pred_values.mean()
    truth_centered = truth_values - truth_values.mean()
    denom = torch.sqrt(pred_centered.square().sum() * truth_centered.square().sum()).clamp(min=1.0e-12)
    corr = (pred_centered * truth_centered).sum() / denom

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "corr": float(corr),
    }


def vorticity_diagnostic(
    pred_u: torch.Tensor,
    pred_v: torch.Tensor,
    true_u: torch.Tensor,
    true_v: torch.Tensor,
    mask: torch.Tensor,
):
    """Simple finite-difference vorticity diagnostic."""
    if not isinstance(pred_u, torch.Tensor):
        pred_u = torch.as_tensor(pred_u)
        pred_v = torch.as_tensor(pred_v)
        true_u = torch.as_tensor(true_u)
        true_v = torch.as_tensor(true_v)
        mask = torch.as_tensor(mask)

    pred_vort = pred_v[..., 1:, :-1] - pred_v[..., :-1, :-1] - (pred_u[..., :-1, 1:] - pred_u[..., :-1, :-1])
    true_vort = true_v[..., 1:, :-1] - true_v[..., :-1, :-1] - (true_u[..., :-1, 1:] - true_u[..., :-1, :-1])
    vort_mask = mask[..., :-1, :-1]

    pred_values = _masked_values(pred_vort.unsqueeze(0) if pred_vort.ndim == 3 else pred_vort, vort_mask)
    true_values = _masked_values(true_vort.unsqueeze(0) if true_vort.ndim == 3 else true_vort, vort_mask)

    diff = pred_values - true_values
    rmse = torch.sqrt(diff.square().mean())
    nrmse = rmse / true_values.std().clamp(min=1.0e-12)

    pred_centered = pred_values - pred_values.mean()
    true_centered = true_values - true_values.mean()
    denom = torch.sqrt(pred_centered.square().sum() * true_centered.square().sum()).clamp(min=1.0e-12)
    corr = (pred_centered * true_centered).sum() / denom

    return {
        "vort_corr": float(corr),
        "vort_rmse": float(rmse),
        "vort_nrmse": float(nrmse),
    }
