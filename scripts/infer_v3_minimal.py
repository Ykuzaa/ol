"""Minimal V3 inference script.

This script is intentionally small and explicit. It runs one V3 inference and
saves all arrays needed for notebook analysis:

    hr
    lr
    mu_cno
    cno_residual = mu_cno - lr
    fm_residual
    pred_cno_fm = mu_cno + fm_residual
    mask, lat, lon

Modes:
    cno_fm     Normal V3 pipeline: LR -> CNO -> mu_CNO -> FM.
    cno_only   Only computes mu_CNO and saves pred_cno_fm = mu_CNO.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oceanlens.models.cno import CNO2d
from oceanlens.models.fm_unet import FlowMatchingUNet
from oceanlens.models.variants import build_fm_condition, build_mu
from oceanlens.utils import load_config, seed_everything


VARIABLES = ["thetao", "so", "zos", "uo", "vo"]


# -----------------------------------------------------------------------------
# Command-line configuration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define every option needed to run one reproducible V3 inference."""
    parser = argparse.ArgumentParser(description="Minimal V3 CNO/FM inference")
    parser.add_argument("--config_dir", default=str(ROOT / "configs"))
    parser.add_argument("--variant", default="v3")
    parser.add_argument("--mode", choices=["cno_fm", "cno_only"], default="cno_fm")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--day_index", type=int, default=0)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--ensemble_members", type=int, default=1)
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--tile_overlap", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_upsample_mode", choices=["nearest", "bilinear", "bicubic"], default="nearest")
    parser.add_argument("--baseline_upsample_mode", choices=["nearest", "bilinear", "bicubic"], default="nearest")
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--coarsened_dir", default=None)
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--cno_ckpt", default=str(ROOT / "runs" / "v2_loggrad" / "cno" / "checkpoints" / "last.ckpt"))
    parser.add_argument("--fm_ckpt", default=str(ROOT / "runs" / "v3" / "fm" / "checkpoints" / "last.ckpt"))
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Checkpoint and model loading
# -----------------------------------------------------------------------------

def load_state(path: Path) -> Dict[str, torch.Tensor]:
    """Load a Lightning or plain PyTorch checkpoint on CPU."""
    state = torch.load(path, map_location="cpu")
    return state.get("state_dict", state)


def strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Keep only checkpoint weights that belong to one submodule prefix."""
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def count_parameters(module) -> int:
    """Return the number of parameters in a model, or zero if absent."""
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def build_cno(cfg, ckpt: Path, device: torch.device):
    """Instantiate the CNO architecture and load the deterministic checkpoint."""
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing CNO checkpoint: {ckpt}")
    cno = CNO2d(
        in_channels=cfg.cno.in_channels,
        out_channels=cfg.cno.out_channels,
        hidden_channels=list(cfg.cno.hidden_channels),
        n_res_blocks=cfg.cno.n_res_blocks,
        kernel_size=cfg.cno.kernel_size,
    )
    state = strip_prefix(load_state(ckpt), "cno.")
    missing, unexpected = cno.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"CNO checkpoint {ckpt} is incompatible. "
            f"Missing keys ({len(missing)}): {missing[:8]}. "
            f"Unexpected keys ({len(unexpected)}): {unexpected[:8]}."
        )
    return cno.to(device).eval()


def build_fm(cfg, ckpt: Path, device: torch.device):
    """Instantiate the Flow Matching U-Net and load its checkpoint."""
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing FM checkpoint: {ckpt}")
    fm = FlowMatchingUNet(
        in_channels=cfg.fm_in_channels,
        out_channels=cfg.fm.out_channels,
        hidden_channels=list(cfg.fm.hidden_channels),
        time_dim=cfg.fm.time_dim,
        n_res=cfg.fm.n_res,
        attn_heads=cfg.fm.attn_heads,
    )
    state = strip_prefix(load_state(ckpt), "fm.")
    missing, unexpected = fm.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"FM checkpoint {ckpt} is incompatible. "
            f"Missing keys ({len(missing)}): {missing[:8]}. "
            f"Unexpected keys ({len(unexpected)}): {unexpected[:8]}."
        )
    return fm.to(device).eval()


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def open_year(raw_dir: Path, coarsened_dir: Path, year: int):
    """Open the HR yearly NetCDF and its paired coarse LR yearly NetCDF."""
    hr_path = raw_dir / f"glorys12_surface_{year}.nc"
    lr_path = coarsened_dir / f"glorys_coarse15_{year}.nc"
    if not hr_path.exists():
        raise FileNotFoundError(hr_path)
    if not lr_path.exists():
        raise FileNotFoundError(lr_path)
    return xr.open_dataset(hr_path), xr.open_dataset(lr_path)


def find_coord(data, names: List[str]):
    """Find latitude or longitude coordinates under common coordinate names."""
    for name in names:
        if name in data.coords:
            return np.asarray(data.coords[name].values)
        if name in data.dims:
            return np.arange(data.sizes[name])
    return None


def load_snapshot(ds: xr.Dataset, variables: List[str], time_index: int):
    """Load one daily surface snapshot as a channel-first NumPy array."""
    arrays = []
    lat = lon = None
    for variable in variables:
        data = ds[variable].isel(time=time_index)
        if "depth" in data.dims:
            data = data.isel(depth=0)
        if lat is None:
            lat = find_coord(data, ["lat", "latitude", "y"])
            lon = find_coord(data, ["lon", "longitude", "x"])
        arrays.append(data.values)
    return np.stack(arrays, axis=0).astype(np.float32), lat, lon


# -----------------------------------------------------------------------------
# Normalization and grid utilities
# -----------------------------------------------------------------------------

def load_stats(path: Optional[Path]):
    """Load per-variable normalization statistics if a stats file exists."""
    if path is None or not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def normalize(data: np.ndarray, variables: List[str], stats):
    """Convert physical fields to the normalized units used by the networks."""
    data = data.copy()
    if stats is None:
        return data
    for index, variable in enumerate(variables):
        data[index] = (data[index] - stats[variable]["mean"]) / (stats[variable]["std"] + 1.0e-8)
    return data


def denormalize(data: np.ndarray, variables: List[str], stats):
    """Convert normalized full fields back to physical units."""
    data = data.copy()
    if stats is None:
        return data
    for index, variable in enumerate(variables):
        data[index] = data[index] * (stats[variable]["std"] + 1.0e-8) + stats[variable]["mean"]
    return data


def rescale_std(data: np.ndarray, variables: List[str], stats):
    """Convert normalized residuals back to physical units without adding means."""
    data = data.copy()
    if stats is None:
        return data
    for index, variable in enumerate(variables):
        data[index] = data[index] * (stats[variable]["std"] + 1.0e-8)
    return data


def upsample_lr(lr: np.ndarray, out_shape: Tuple[int, int], mode: str) -> np.ndarray:
    """Interpolate the LR input onto the HR grid expected by the models."""
    tensor = torch.from_numpy(lr).unsqueeze(0)
    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(tensor, size=out_shape, mode=mode, **kwargs).squeeze(0).numpy()


# -----------------------------------------------------------------------------
# Tiling utilities
# -----------------------------------------------------------------------------

def tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    """Compute tile start indices, ensuring the final tile reaches the boundary."""
    if tile_size <= 0 or length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(height: int, width: int) -> np.ndarray:
    """Create a smooth 2D window used to blend overlapping tile predictions."""
    wy = np.hanning(height) if height > 2 else np.ones(height)
    wx = np.hanning(width) if width > 2 else np.ones(width)
    window = np.outer(wy, wx).astype(np.float32)
    return np.maximum(window, 1.0e-3)[None, :, :]


# -----------------------------------------------------------------------------
# Core inference
# -----------------------------------------------------------------------------

@torch.no_grad()
def infer_tile(cno, fm, cfg, lr: np.ndarray, mask: np.ndarray, args, device: torch.device):
    """Run CNO first, then optionally run FM on one tile.

    Returned arrays are still in normalized units:
        pred_mean: final prediction, either mu_CNO or mu_CNO + FM residual
        fm_mean: FM residual correction
        mu_np: deterministic CNO field used as FM condition
        cno_residual_np: CNO correction relative to LR
    """
    lr_t = torch.from_numpy(lr).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    # Inference step 1: deterministic CNO.
    # The V3 CNO is residual, so it predicts CNO(LR), then:
    #     mu_CNO = LR_up + CNO(LR_up)
    cno_out = cno(lr_t)
    mu = build_mu(cno_out, lr_t, cfg)
    cno_residual = mu - lr_t

    if args.mode == "cno_only":
        # Stop after the deterministic CNO branch and save zero FM correction.
        mu_np = (mu * mask_t).squeeze(0).cpu().numpy()
        return mu_np, np.zeros_like(mu_np), mu_np, (cno_residual * mask_t).squeeze(0).cpu().numpy()

    # Inference step 2: stochastic Flow Matching residual.
    # For V3, the FM condition is mu_CNO. The FM integrates a random initial
    # state into a predicted residual correction on top of mu_CNO.
    condition = build_fm_condition(mu, lr_t, mask_t, cfg)
    samples = []
    fm_outputs = []
    dt = 1.0 / args.n_steps
    for _ in range(args.ensemble_members):
        # Flow Matching starts from Gaussian noise and integrates velocity to t=1.
        x = torch.randn_like(lr_t)
        for index in range(args.n_steps):
            t = torch.full((x.shape[0],), index * dt, device=device)
            v1 = fm(x, t, condition)
            if args.solver == "heun":
                x_euler = x + dt * v1
                t_next = torch.full((x.shape[0],), min(1.0, (index + 1) * dt), device=device)
                v2 = fm(x_euler, t_next, condition)
                x = x + 0.5 * dt * (v1 + v2)
            else:
                x = x + dt * v1
        # Final composition is an addition, not concatenation:
        #     prediction = mu_CNO + FM_residual
        pred = (mu + x) * mask_t
        samples.append(pred.squeeze(0).cpu().numpy())
        fm_outputs.append((x * mask_t).squeeze(0).cpu().numpy())

    pred_mean = np.stack(samples, axis=0).mean(axis=0)
    fm_mean = np.stack(fm_outputs, axis=0).mean(axis=0)
    mu_np = (mu * mask_t).squeeze(0).cpu().numpy()
    cno_residual_np = (cno_residual * mask_t).squeeze(0).cpu().numpy()
    return pred_mean, fm_mean, mu_np, cno_residual_np


def infer_tiled(cno, fm, cfg, lr: np.ndarray, mask: np.ndarray, args, device: torch.device):
    """Run global inference tile-by-tile and blend overlaps back together."""
    channels, height, width = lr.shape
    pred_sum = np.zeros((channels, height, width), dtype=np.float32)
    fm_sum = np.zeros((channels, height, width), dtype=np.float32)
    mu_sum = np.zeros((channels, height, width), dtype=np.float32)
    cno_residual_sum = np.zeros((channels, height, width), dtype=np.float32)
    weight_sum = np.zeros((1, height, width), dtype=np.float32)

    y_starts = tile_starts(height, args.tile_size, args.tile_overlap)
    x_starts = tile_starts(width, args.tile_size, args.tile_overlap)
    total = len(y_starts) * len(x_starts)
    done = 0
    for top in y_starts:
        bottom = min(top + args.tile_size, height) if args.tile_size > 0 else height
        for left in x_starts:
            right = min(left + args.tile_size, width) if args.tile_size > 0 else width
            done += 1
            print(f"tile {done}/{total}: y={top}:{bottom}, x={left}:{right}", flush=True)
            pred, fm_residual, mu, cno_residual = infer_tile(
                cno,
                fm,
                cfg,
                lr[:, top:bottom, left:right],
                mask[:, top:bottom, left:right],
                args,
                device,
            )
            window = blend_window(bottom - top, right - left)
            # Accumulate weighted outputs so overlapping tiles blend smoothly.
            pred_sum[:, top:bottom, left:right] += pred * window
            fm_sum[:, top:bottom, left:right] += fm_residual * window
            mu_sum[:, top:bottom, left:right] += mu * window
            cno_residual_sum[:, top:bottom, left:right] += cno_residual * window
            weight_sum[:, top:bottom, left:right] += window
            if device.type == "cuda":
                torch.cuda.empty_cache()

    weight = np.maximum(weight_sum, 1.0e-8)
    return pred_sum / weight, fm_sum / weight, mu_sum / weight, cno_residual_sum / weight


# -----------------------------------------------------------------------------
# Metrics and output files
# -----------------------------------------------------------------------------

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a robust Pearson correlation for flattened ocean pixels."""
    a = a.ravel()
    b = b.ravel()
    if a.size < 2 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def write_metrics(path: Path, arrays: Dict[str, np.ndarray], variables: List[str]):
    """Write simple CSV metrics for LR, CNO-only, CNO+FM, and FM residual."""
    valid = arrays["mask"][0] > 0.5
    rows = []
    for channel, variable in enumerate(variables):
        hr = arrays["hr"][channel][valid]
        lr = arrays["lr"][channel][valid]
        mu = arrays["mu_cno"][channel][valid]
        pred = arrays["pred_cno_fm"][channel][valid]
        fm = arrays["fm_residual"][channel][valid]
        target_fm = hr - mu
        for name, field in [("lr", lr), ("mu_cno", mu), ("pred_cno_fm", pred)]:
            diff = field - hr
            rows.append(
                {
                    "variable": variable,
                    "field": name,
                    "mae": float(np.mean(np.abs(diff))),
                    "rmse": float(np.sqrt(np.mean(diff ** 2))),
                    "bias": float(np.mean(diff)),
                    "corr_with_hr": safe_corr(field, hr),
                }
            )
        rows.append(
            {
                "variable": variable,
                "field": "fm_residual_vs_target",
                "mae": float(np.mean(np.abs(fm - target_fm))),
                "rmse": float(np.sqrt(np.mean((fm - target_fm) ** 2))),
                "bias": float(np.mean(fm - target_fm)),
                "corr_with_hr": safe_corr(fm, target_fm),
            }
        )

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, arrays: Dict[str, np.ndarray], lat, lon):
    """Save model outputs plus coordinates in one compressed NPZ file."""
    payload = dict(arrays)
    if lat is not None:
        payload["lat"] = lat
    if lon is not None:
        payload["lon"] = lon
    np.savez_compressed(path, **payload)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    """Run the complete inference workflow and save arrays, metrics, metadata."""
    # 1. Parse CLI options, set deterministic seed, and create output folder.
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load configuration, paths, variables, year, and normalization stats.
    cfg = load_config(args.variant, args.config_dir)
    variables = list(cfg.variables)
    year = args.year if args.year is not None else int(list(cfg.data.test_years)[0])
    raw_dir = Path(args.raw_dir or cfg.paths.raw_dir)
    coarsened_dir = Path(args.coarsened_dir or cfg.paths.coarsened_dir)
    stats_path = Path(args.norm_stats or cfg.paths.norm_stats)
    stats = load_stats(stats_path)

    # 3. Load only the model branches needed by the selected mode.
    cno = None
    cno = build_cno(cfg, Path(args.cno_ckpt), device)
    fm = None
    if args.mode == "cno_fm":
        fm = build_fm(cfg, Path(args.fm_ckpt), device)

    # 4. Open paired HR/LR yearly datasets and select the requested day.
    ds_hr, ds_lr = open_year(raw_dir, coarsened_dir, year)
    if args.day_index >= ds_hr.sizes["time"]:
        raise IndexError(f"day_index={args.day_index} outside available range 0..{ds_hr.sizes['time'] - 1}")

    time_value = ds_hr["time"].isel(time=args.day_index).values
    date = str(np.datetime_as_string(time_value, unit="D"))
    print(f"Running minimal inference for {date}", flush=True)

    # 5. Load HR and LR daily fields, then build the ocean mask from HR NaNs.
    hr_raw, lat, lon = load_snapshot(ds_hr, variables, args.day_index)
    lr_raw, _, _ = load_snapshot(ds_lr, variables, args.day_index)
    mask = np.all(~np.isnan(hr_raw), axis=0, keepdims=True).astype(np.float32)

    # 6. Fill land NaNs with zeros, normalize fields, and upsample LR to HR grid.
    hr_filled = np.nan_to_num(hr_raw, nan=0.0)
    lr_filled = np.nan_to_num(lr_raw, nan=0.0)
    hr_norm = normalize(hr_filled, variables, stats)
    lr_norm = normalize(lr_filled, variables, stats)
    lr_model_norm = upsample_lr(lr_norm, hr_norm.shape[-2:], args.model_upsample_mode)
    lr_baseline_norm = upsample_lr(lr_norm, hr_norm.shape[-2:], args.baseline_upsample_mode)

    # 7. Run tiled inference and record GPU memory/time.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    pred_norm, fm_norm, mu_norm, cno_residual_norm = infer_tiled(cno, fm, cfg, lr_model_norm, mask, args, device)
    elapsed = time.perf_counter() - start
    peak_gpu_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None

    # 8. Convert outputs back to physical units and build diagnostic residuals.
    arrays = {
        "hr": denormalize(hr_norm, variables, stats) * mask,
        "lr": denormalize(lr_baseline_norm, variables, stats) * mask,
        "mu_cno": denormalize(mu_norm, variables, stats) * mask,
        "cno_residual": rescale_std(cno_residual_norm, variables, stats) * mask,
        "fm_residual": rescale_std(fm_norm, variables, stats) * mask,
        "pred_cno_fm": denormalize(pred_norm, variables, stats) * mask,
        "mask": mask,
    }
    arrays["target_total_residual"] = arrays["hr"] - arrays["lr"]
    arrays["target_fm_residual"] = arrays["hr"] - arrays["mu_cno"]
    arrays["error_before_fm"] = arrays["hr"] - arrays["mu_cno"]
    arrays["error_after_fm"] = arrays["hr"] - arrays["pred_cno_fm"]

    # 9. Save the NPZ for notebooks and a compact CSV for quick terminal checks.
    npz_path = out_dir / f"day_{date}.npz"
    save_npz(npz_path, arrays, lat, lon)
    write_metrics(out_dir / "metrics.csv", arrays, variables)

    # 10. Save metadata so the run can be reproduced later.
    metadata = {
        "date": date,
        "year": year,
        "day_index": args.day_index,
        "variant": args.variant,
        "mode": args.mode,
        "variables": variables,
        "n_steps": args.n_steps,
        "solver": args.solver,
        "ensemble_members": args.ensemble_members,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "model_upsample_mode": args.model_upsample_mode,
        "baseline_upsample_mode": args.baseline_upsample_mode,
        "raw_dir": str(raw_dir),
        "coarsened_dir": str(coarsened_dir),
        "norm_stats": str(stats_path) if stats is not None else None,
        "cno_ckpt": str(args.cno_ckpt) if cno is not None else None,
        "fm_ckpt": str(args.fm_ckpt) if fm is not None else None,
        "cno_parameters": count_parameters(cno),
        "fm_parameters": count_parameters(fm),
        "inference_seconds": elapsed,
        "peak_gpu_allocated_mb": peak_gpu_mb,
        "npz": str(npz_path),
    }
    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    # 11. Close datasets and print the most important output paths.
    ds_hr.close()
    ds_lr.close()
    print(f"Saved {npz_path}", flush=True)
    print(f"Metrics: {out_dir / 'metrics.csv'}", flush=True)
    print(f"Elapsed seconds: {elapsed:.3f}", flush=True)
    print(f"Peak GPU allocated MB: {peak_gpu_mb}", flush=True)


if __name__ == "__main__":
    main()
