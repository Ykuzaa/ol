"""Run one FM-only v5 inference on one monthly ECMWF file and save diagnostics."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5 monthly single-day inference")
    parser.add_argument("--config_dir", default=str(ROOT / "configs"))
    parser.add_argument("--variant", default="v5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_nc", required=True)
    parser.add_argument("--time_index", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--ensemble_members", type=int, default=1)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--fm_ckpt", required=True)
    parser.add_argument("--downsample_factor", type=int, default=3)
    return parser.parse_args()


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    return state.get("state_dict", state)


def strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def build_fm(cfg, ckpt: Path, device: torch.device):
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


def find_coord(data, names: List[str]):
    for name in names:
        if name in data.coords:
            return np.asarray(data.coords[name].values)
        if name in data.dims:
            return np.arange(data.sizes[name])
    return None


def load_snapshot(ds: xr.Dataset, variables: List[str], time_index: int):
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


def load_stats(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def normalize(data: np.ndarray, variables: List[str], stats):
    data = data.copy()
    if stats is None:
        return data
    for index, variable in enumerate(variables):
        data[index] = (data[index] - stats[variable]["mean"]) / (stats[variable]["std"] + 1.0e-8)
    return data


def denormalize(data: np.ndarray, variables: List[str], stats):
    data = data.copy()
    if stats is None:
        return data
    for index, variable in enumerate(variables):
        data[index] = data[index] * (stats[variable]["std"] + 1.0e-8) + stats[variable]["mean"]
    return data


def rescale_std(data: np.ndarray, variables: List[str], stats):
    data = data.copy()
    if stats is None:
        return data
    for index, variable in enumerate(variables):
        data[index] = data[index] * (stats[variable]["std"] + 1.0e-8)
    return data


def downsample_and_upsample(hr: np.ndarray, factor: int) -> np.ndarray:
    tensor = torch.from_numpy(hr).unsqueeze(0)
    lr = F.avg_pool2d(tensor, kernel_size=factor, stride=factor, ceil_mode=True)
    return F.interpolate(lr, size=hr.shape[-2:], mode="nearest").squeeze(0).numpy()


def tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if tile_size <= 0 or length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(height: int, width: int) -> np.ndarray:
    wy = np.hanning(height) if height > 2 else np.ones(height)
    wx = np.hanning(width) if width > 2 else np.ones(width)
    window = np.outer(wy, wx).astype(np.float32)
    return np.maximum(window, 1.0e-3)[None, :, :]


@torch.no_grad()
def infer_tile(fm, cfg, lr: np.ndarray, mask: np.ndarray, args, device: torch.device):
    lr_t = torch.from_numpy(lr).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    condition = build_fm_condition(None, lr_t, mask_t, cfg)
    samples = []
    residuals = []
    dt = 1.0 / args.n_steps
    for _ in range(args.ensemble_members):
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
        pred = (lr_t + x) * mask_t
        samples.append(pred.squeeze(0).cpu().numpy())
        residuals.append((x * mask_t).squeeze(0).cpu().numpy())
    pred_mean = np.stack(samples, axis=0).mean(axis=0)
    residual_mean = np.stack(residuals, axis=0).mean(axis=0)
    return pred_mean, residual_mean


def infer_tiled(fm, cfg, lr: np.ndarray, mask: np.ndarray, args, device: torch.device):
    channels, height, width = lr.shape
    pred_sum = np.zeros((channels, height, width), dtype=np.float32)
    fm_sum = np.zeros((channels, height, width), dtype=np.float32)
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
            pred, fm_residual = infer_tile(
                fm,
                cfg,
                lr[:, top:bottom, left:right],
                mask[:, top:bottom, left:right],
                args,
                device,
            )
            window = blend_window(bottom - top, right - left)
            pred_sum[:, top:bottom, left:right] += pred * window
            fm_sum[:, top:bottom, left:right] += fm_residual * window
            weight_sum[:, top:bottom, left:right] += window
            if device.type == "cuda":
                torch.cuda.empty_cache()

    weight = np.maximum(weight_sum, 1.0e-8)
    return pred_sum / weight, fm_sum / weight


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.size < 2 or np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def write_metrics(path: Path, arrays: Dict[str, np.ndarray], variables: List[str]):
    valid = arrays["mask"][0] > 0.5
    rows = []
    for channel, variable in enumerate(variables):
        hr = arrays["hr"][channel][valid]
        lr = arrays["lr"][channel][valid]
        pred = arrays["pred_ablation"][channel][valid]
        fm = arrays["correction_ablation"][channel][valid]
        target = (hr - lr)
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
        rows.append(
            {
                "variable": variable,
                "field": "fm_residual_vs_target",
                "mae": float(np.mean(np.abs(fm - target))),
                "rmse": float(np.sqrt(np.mean((fm - target) ** 2))),
                "bias": float(np.mean(fm - target)),
                "corr_with_hr": safe_corr(fm, target),
            }
        )
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, arrays: Dict[str, np.ndarray], lat, lon):
    payload = dict(arrays)
    if lat is not None:
        payload["lat"] = lat
    if lon is not None:
        payload["lon"] = lon
    np.savez_compressed(path, **payload)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.variant, args.config_dir)
    variables = list(cfg.variables)
    stats_path = Path(args.norm_stats or cfg.paths.norm_stats)
    stats = load_stats(stats_path)

    fm = build_fm(cfg, Path(args.fm_ckpt), device)

    ds_hr = xr.open_dataset(args.input_nc)
    if args.time_index >= ds_hr.sizes["time"]:
        raise IndexError(f"time_index={args.time_index} outside available range 0..{ds_hr.sizes['time'] - 1}")

    time_value = ds_hr["time"].isel(time=args.time_index).values
    date = str(np.datetime_as_string(time_value, unit="D"))
    print(f"Running v5 monthly inference for {date}", flush=True)

    hr_raw, lat, lon = load_snapshot(ds_hr, variables, args.time_index)
    mask = np.all(~np.isnan(hr_raw), axis=0, keepdims=True).astype(np.float32)

    hr_filled = np.nan_to_num(hr_raw, nan=0.0)
    lr_filled = downsample_and_upsample(hr_filled, args.downsample_factor)
    lr_filled *= mask

    hr_norm = normalize(hr_filled, variables, stats)
    lr_norm = normalize(lr_filled, variables, stats)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    pred_norm, fm_norm = infer_tiled(fm, cfg, lr_norm, mask, args, device)
    elapsed = time.perf_counter() - start
    peak_gpu_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None

    arrays = {
        "hr": denormalize(hr_norm, variables, stats) * mask,
        "lr": denormalize(lr_norm, variables, stats) * mask,
        "pred_ablation": denormalize(pred_norm, variables, stats) * mask,
        "correction_ablation": rescale_std(fm_norm, variables, stats) * mask,
        "mask": mask,
    }
    arrays["target_total_residual"] = arrays["hr"] - arrays["lr"]
    arrays["error_ablation"] = arrays["hr"] - arrays["pred_ablation"]

    npz_path = out_dir / f"day_{date}.npz"
    save_npz(npz_path, arrays, lat, lon)
    write_metrics(out_dir / "metrics.csv", arrays, variables)

    metadata = {
        "date": date,
        "time_index": args.time_index,
        "variant": args.variant,
        "variables": variables,
        "input_nc": str(args.input_nc),
        "n_steps": args.n_steps,
        "solver": args.solver,
        "ensemble_members": args.ensemble_members,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "norm_stats": str(stats_path) if stats is not None else None,
        "fm_ckpt": str(args.fm_ckpt),
        "inference_seconds": elapsed,
        "peak_gpu_allocated_mb": peak_gpu_mb,
        "npz": str(npz_path),
    }
    with open(out_dir / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    ds_hr.close()
    print(f"Saved {npz_path}", flush=True)
    print(f"Metrics: {out_dir / 'metrics.csv'}", flush=True)
    print(f"Elapsed seconds: {elapsed:.3f}", flush=True)
    print(f"Peak GPU allocated MB: {peak_gpu_mb}", flush=True)


if __name__ == "__main__":
    main()
