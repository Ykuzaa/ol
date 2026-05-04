"""Minimal ablation FM inference script.

The ablation model has no CNO. Its pipeline is:

    LR -> upsample to HR grid -> condition FM on LR_up
    x_0 ~ N(0, I)
    FM ODE integration -> pred_ablation

Unlike V3, the ablation FM output is a full field, not a residual added to
mu_CNO. The script saves arrays for notebook analysis:

    hr
    lr
    pred_ablation
    error_ablation = hr - pred_ablation
    target_total_residual = hr - lr
    correction_ablation = pred_ablation - lr
    mask, lat, lon
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

from oceanlens.models.fm_unet import FlowMatchingUNet
from oceanlens.models.variants import build_fm_condition
from oceanlens.utils import load_config, seed_everything


VARIABLES = ["thetao", "so", "zos", "uo", "vo"]


# -----------------------------------------------------------------------------
# Command-line configuration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define every option needed to run one reproducible ablation inference."""
    parser = argparse.ArgumentParser(description="Minimal ablation FM inference")
    parser.add_argument("--config_dir", default=str(ROOT / "configs"))
    parser.add_argument("--variant", default="ablation")
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
    parser.add_argument("--input_npz", default=None)
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--fm_ckpt", default=str(ROOT / "runs" / "ablation" / "fm" / "checkpoints" / "last.ckpt"))
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
    """Return the number of parameters in a model."""
    return int(sum(parameter.numel() for parameter in module.parameters()))


def build_fm(cfg, ckpt: Path, device: torch.device):
    """Instantiate the ablation Flow Matching U-Net and load its checkpoint."""
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


def load_snapshot_from_npz(path: Path):
    """Load one saved test case from NPZ with physical-unit hr/lr/mask arrays."""
    data = np.load(path)
    lat = data["lat"] if "lat" in data.files else None
    lon = data["lon"] if "lon" in data.files else None
    return data["hr"].astype(np.float32), data["lr"].astype(np.float32), data["mask"].astype(np.float32), lat, lon


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
    """Convert physical fields to the normalized units used by the network."""
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


def upsample_lr(lr: np.ndarray, out_shape: Tuple[int, int], mode: str) -> np.ndarray:
    """Interpolate the LR input onto the HR grid expected by the FM."""
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
    """Create a smooth 2D Gaussian window used to blend overlapping tiles."""
    if height <= 2 or width <= 2:
        return np.ones((1, height, width), dtype=np.float32)

    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    sigma = 0.5
    wy = np.exp(-0.5 * (y / sigma) ** 2)
    wx = np.exp(-0.5 * (x / sigma) ** 2)
    window = np.outer(wy, wx).astype(np.float32)
    return np.maximum(window, 1.0e-3)[None, :, :]


# -----------------------------------------------------------------------------
# Core ablation inference
# -----------------------------------------------------------------------------

@torch.no_grad()
def infer_tile(fm, cfg, lr: np.ndarray, mask: np.ndarray, args, device: torch.device):
    """Run ablation FM on one tile and return a normalized full-field output."""
    lr_t = torch.from_numpy(lr).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    # Ablation condition is LR_up only. There is no CNO and no mu_CNO.
    condition = build_fm_condition(None, lr_t, mask_t, cfg)
    samples = []
    dt = 1.0 / args.n_steps
    for _ in range(args.ensemble_members):
        # The ablation FM starts from Gaussian noise and integrates to a full HR field.
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
        if getattr(cfg, "fm_target", None) == "residual_lr":
            pred = (lr_t + x) * mask_t
        else:
            pred = x * mask_t
        samples.append(pred.squeeze(0).cpu().numpy())
    return np.stack(samples, axis=0).mean(axis=0)


def infer_tiled(fm, cfg, lr: np.ndarray, mask: np.ndarray, args, device: torch.device):
    """Run global ablation inference tile-by-tile and blend overlaps."""
    channels, height, width = lr.shape
    pred_sum = np.zeros((channels, height, width), dtype=np.float32)
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
            pred_tile = infer_tile(
                fm,
                cfg,
                lr[:, top:bottom, left:right],
                mask[:, top:bottom, left:right],
                args,
                device,
            )
            window = blend_window(bottom - top, right - left)
            pred_sum[:, top:bottom, left:right] += pred_tile * window
            weight_sum[:, top:bottom, left:right] += window
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return pred_sum / np.maximum(weight_sum, 1.0e-8)


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
    """Write simple CSV metrics for LR and ablation FM."""
    valid = arrays["mask"][0] > 0.5
    rows = []
    for channel, variable in enumerate(variables):
        hr = arrays["hr"][channel][valid]
        lr = arrays["lr"][channel][valid]
        pred = arrays["pred_ablation"][channel][valid]
        for name, field in [("lr", lr), ("pred_ablation", pred)]:
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
    """Run ablation FM inference and save arrays, metrics, metadata."""
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.variant, args.config_dir)
    variables = list(cfg.variables)
    year = args.year if args.year is not None else int(list(cfg.data.test_years)[0])
    raw_dir = Path(args.raw_dir or cfg.paths.raw_dir)
    coarsened_dir = Path(args.coarsened_dir or cfg.paths.coarsened_dir)
    stats_path = Path(args.norm_stats or cfg.paths.norm_stats)
    stats = load_stats(stats_path)

    fm = build_fm(cfg, Path(args.fm_ckpt), device)

    ds_hr = ds_lr = None
    if args.input_npz is not None:
        npz_path = Path(args.input_npz)
        hr_raw, lr_raw, mask, lat, lon = load_snapshot_from_npz(npz_path)
        date = npz_path.stem.replace("day_", "")
        print(f"Running ablation inference from NPZ for {date}", flush=True)
    else:
        ds_hr, ds_lr = open_year(raw_dir, coarsened_dir, year)
        if args.day_index >= ds_hr.sizes["time"]:
            raise IndexError(f"day_index={args.day_index} outside available range 0..{ds_hr.sizes['time'] - 1}")

        time_value = ds_hr["time"].isel(time=args.day_index).values
        date = str(np.datetime_as_string(time_value, unit="D"))
        print(f"Running ablation inference for {date}", flush=True)

        hr_raw, lat, lon = load_snapshot(ds_hr, variables, args.day_index)
        lr_raw, _, _ = load_snapshot(ds_lr, variables, args.day_index)
        mask = np.all(~np.isnan(hr_raw), axis=0, keepdims=True).astype(np.float32)

    hr_filled = np.nan_to_num(hr_raw, nan=0.0)
    lr_filled = np.nan_to_num(lr_raw, nan=0.0)
    hr_norm = normalize(hr_filled, variables, stats)
    lr_norm = normalize(lr_filled, variables, stats)
    lr_model_norm = upsample_lr(lr_norm, hr_norm.shape[-2:], args.model_upsample_mode)
    lr_baseline_norm = upsample_lr(lr_norm, hr_norm.shape[-2:], args.baseline_upsample_mode)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    pred_norm = infer_tiled(fm, cfg, lr_model_norm, mask, args, device)
    elapsed = time.perf_counter() - start
    peak_gpu_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None

    arrays = {
        "hr": denormalize(hr_norm, variables, stats) * mask,
        "lr": denormalize(lr_baseline_norm, variables, stats) * mask,
        "pred_ablation": denormalize(pred_norm, variables, stats) * mask,
        "mask": mask,
    }
    arrays["target_total_residual"] = arrays["hr"] - arrays["lr"]
    arrays["correction_ablation"] = arrays["pred_ablation"] - arrays["lr"]
    arrays["error_ablation"] = arrays["hr"] - arrays["pred_ablation"]

    npz_path = out_dir / f"day_{date}.npz"
    save_npz(npz_path, arrays, lat, lon)
    write_metrics(out_dir / "metrics.csv", arrays, variables)

    metadata = {
        "date": date,
        "year": year,
        "day_index": args.day_index,
        "variant": args.variant,
        "mode": "ablation_fm",
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
        "fm_ckpt": str(args.fm_ckpt),
        "fm_parameters": count_parameters(fm),
        "inference_seconds": elapsed,
        "peak_gpu_allocated_mb": peak_gpu_mb,
        "npz": str(npz_path),
    }
    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    if ds_hr is not None:
        ds_hr.close()
    if ds_lr is not None:
        ds_lr.close()
    print(f"Saved {npz_path}", flush=True)
    print(f"Metrics: {out_dir / 'metrics.csv'}", flush=True)
    print(f"Elapsed seconds: {elapsed:.3f}", flush=True)
    print(f"Peak GPU allocated MB: {peak_gpu_mb}", flush=True)


if __name__ == "__main__":
    main()
