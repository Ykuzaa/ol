import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from model_core.models.cno import CNO2d
from model_core.models.fm_unet import FlowMatchingUNet
from model_core.models.variants import build_fm_condition, build_mu
from model_core.utils import load_config, seed_everything


VARIABLES = ["thetao", "so", "zos", "uo", "vo"]
DOMAINS = {
    "ibi": {"lat": (26.0, 56.0), "lon": (-20.0, 13.0)},
    "med": {"lat": (30.0, 46.5), "lon": (-6.0, 37.0)},
    "global": None,
}


def _coord_name(ds: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"Could not find coordinate among {candidates}")


def _variable_2d(ds: xr.Dataset, variable: str) -> xr.DataArray:
    if variable not in ds:
        raise ValueError(f"Input NetCDF is missing required variable: {variable}")
    da = ds[variable]
    if "time" in da.dims:
        da = da.isel(time=0)
    if "depth" in da.dims:
        da = da.isel(depth=0)
    return da.squeeze(drop=True)


def _crop_domain(ds: xr.Dataset, domain: str) -> xr.Dataset:
    domain = domain.lower()
    if domain not in DOMAINS:
        raise ValueError(f"DOMAIN must be one of {sorted(DOMAINS)}")
    if DOMAINS[domain] is None:
        return ds

    lat_name = _coord_name(ds, ["latitude", "lat", "y"])
    lon_name = _coord_name(ds, ["longitude", "lon", "x"])
    lat_min, lat_max = DOMAINS[domain]["lat"]
    lon_min, lon_max = DOMAINS[domain]["lon"]
    lat = ds[lat_name]
    lon = ds[lon_name]
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon180 >= lon_min) & (lon180 <= lon_max)
    cropped = ds.where(lat_mask & lon_mask, drop=True)
    if cropped.sizes.get(lat_name, 0) == 0 or cropped.sizes.get(lon_name, 0) == 0:
        raise ValueError(f"Domain {domain} does not overlap the input grid.")
    return cropped


def _load_lr(ds: xr.Dataset, domain: str):
    ds = _crop_domain(ds, domain)
    arrays = [_variable_2d(ds, variable).values.astype(np.float32) for variable in VARIABLES]
    lr = np.stack(arrays, axis=0)
    mask = np.all(np.isfinite(lr), axis=0, keepdims=True).astype(np.float32)
    lr = np.nan_to_num(lr, nan=0.0)

    sample = _variable_2d(ds, VARIABLES[0])
    lat_name = _coord_name(sample.to_dataset(name=VARIABLES[0]), ["latitude", "lat", "y"])
    lon_name = _coord_name(sample.to_dataset(name=VARIABLES[0]), ["longitude", "lon", "x"])
    lat = np.asarray(sample[lat_name].values)
    lon = np.asarray(sample[lon_name].values)
    return lr, mask, lat, lon


def _load_stats(path: Path):
    with open(path) as handle:
        return json.load(handle)


def _normalize(data: np.ndarray, stats: dict) -> np.ndarray:
    out = data.copy()
    for idx, variable in enumerate(VARIABLES):
        out[idx] = (out[idx] - stats[variable]["mean"]) / (stats[variable]["std"] + 1.0e-8)
    return out


def _denormalize(data: np.ndarray, stats: dict) -> np.ndarray:
    out = data.copy()
    for idx, variable in enumerate(VARIABLES):
        out[idx] = out[idx] * (stats[variable]["std"] + 1.0e-8) + stats[variable]["mean"]
    return out


def _upsample(data: np.ndarray, shape: tuple[int, int], mode: str = "nearest") -> np.ndarray:
    tensor = torch.from_numpy(data).unsqueeze(0)
    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(tensor, size=shape, mode=mode, **kwargs).squeeze(0).numpy()


def _high_res_coords(coord: np.ndarray, size: int) -> np.ndarray:
    if coord.size == size:
        return coord.astype(np.float32)
    return np.linspace(float(coord[0]), float(coord[-1]), size, dtype=np.float32)


def _load_state(path: Path):
    state = torch.load(path, map_location="cpu")
    return state.get("state_dict", state)


def _strip_prefix(state: dict, prefix: str):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def _build_cno(cfg, ckpt: Path, device: torch.device):
    cno = CNO2d(
        in_channels=cfg.cno.in_channels,
        out_channels=cfg.cno.out_channels,
        hidden_channels=list(cfg.cno.hidden_channels),
        n_res_blocks=cfg.cno.n_res_blocks,
        kernel_size=cfg.cno.kernel_size,
    )
    missing, unexpected = cno.load_state_dict(_strip_prefix(_load_state(ckpt), "cno."), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Incompatible CNO checkpoint. Missing={missing[:5]}, unexpected={unexpected[:5]}")
    return cno.to(device).eval()


def _build_fm(cfg, ckpt: Path, device: torch.device):
    fm = FlowMatchingUNet(
        in_channels=cfg.fm_in_channels,
        out_channels=cfg.fm.out_channels,
        hidden_channels=list(cfg.fm.hidden_channels),
        time_dim=cfg.fm.time_dim,
        n_res=cfg.fm.n_res,
        attn_heads=cfg.fm.attn_heads,
    )
    missing, unexpected = fm.load_state_dict(_strip_prefix(_load_state(ckpt), "fm."), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Incompatible FM checkpoint. Missing={missing[:5]}, unexpected={unexpected[:5]}")
    return fm.to(device).eval()


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(height: int, width: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    wy = np.exp(-0.5 * (y / 0.5) ** 2)
    wx = np.exp(-0.5 * (x / 0.5) ** 2)
    return np.maximum(np.outer(wy, wx).astype(np.float32), 1.0e-3)[None, :, :]


@torch.no_grad()
def _infer_tile(cno, fm, cfg, lr: np.ndarray, mask: np.ndarray, args, device: torch.device):
    lr_t = torch.from_numpy(lr).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    mu = build_mu(cno(lr_t), lr_t, cfg)
    condition = build_fm_condition(mu, lr_t, mask_t, cfg)

    x = args.noise_sigma * torch.randn_like(lr_t)
    dt = 1.0 / args.n_steps
    for index in range(args.n_steps):
        t = torch.full((x.shape[0],), index * dt, device=device)
        x = x + dt * fm(x, t, condition)

    pred = (mu + x) * mask_t
    return pred.squeeze(0).cpu().numpy()


def _infer_tiled(cno, fm, cfg, lr: np.ndarray, mask: np.ndarray, args, device: torch.device):
    channels, height, width = lr.shape
    pred_sum = np.zeros((channels, height, width), dtype=np.float32)
    weight_sum = np.zeros((1, height, width), dtype=np.float32)
    y_starts = _tile_starts(height, args.tile_size, args.tile_overlap)
    x_starts = _tile_starts(width, args.tile_size, args.tile_overlap)

    total = len(y_starts) * len(x_starts)
    done = 0
    for top in y_starts:
        bottom = min(top + args.tile_size, height)
        for left in x_starts:
            right = min(left + args.tile_size, width)
            done += 1
            print(f"tile {done}/{total}: y={top}:{bottom}, x={left}:{right}", flush=True)
            pred = _infer_tile(cno, fm, cfg, lr[:, top:bottom, left:right], mask[:, top:bottom, left:right], args, device)
            window = _blend_window(bottom - top, right - left)
            pred_sum[:, top:bottom, left:right] += pred * window
            weight_sum[:, top:bottom, left:right] += window
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return pred_sum / np.maximum(weight_sum, 1.0e-8)


def run_oceanlens_super_resolution(
    lr_dataset: xr.Dataset,
    model_dir: str,
    domain: str = "ibi",
    n_steps: int = 20,
    upsample_factor: int = 15,
    tile_size: int = 512,
    tile_overlap: int = 64,
    seed: int = 42,
) -> tuple[xr.Dataset, dict]:
    seed_everything(seed)
    model_path = Path(model_dir)
    cfg = load_config("v4_s1_logit_t", str(model_path / "configs"))
    stats = _load_stats(model_path / "norm_stats.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr, mask_lr, lat_lr, lon_lr = _load_lr(lr_dataset, domain)
    high_shape = (int(lr.shape[-2] * upsample_factor), int(lr.shape[-1] * upsample_factor))
    lr_norm = _normalize(lr, stats)
    lr_model = _upsample(lr_norm, high_shape, "nearest")
    mask = _upsample(mask_lr, high_shape, "nearest")

    cno = _build_cno(cfg, model_path / "checkpoints/cno_v2_loggrad.ckpt", device)
    fm = _build_fm(cfg, model_path / "checkpoints/fm_v4_s1_logit_t.ckpt", device)
    args = SimpleNamespace(n_steps=n_steps, noise_sigma=1.0, tile_size=tile_size, tile_overlap=tile_overlap)
    pred_norm = _infer_tiled(cno, fm, cfg, lr_model, mask, args, device)
    pred = _denormalize(pred_norm, stats) * mask

    lat_hr = _high_res_coords(lat_lr, high_shape[0])
    lon_hr = _high_res_coords(lon_lr, high_shape[1])
    time = _extract_time(lr_dataset)
    coords = {"time": [time], "latitude": lat_hr, "longitude": lon_hr}
    data_vars = {
        variable: (("time", "latitude", "longitude"), pred[index][None].astype(np.float32))
        for index, variable in enumerate(VARIABLES)
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs.update(
        {
            "Conventions": "CF-1.8",
            "title": "OceanLens on-demand super-resolution",
            "institution": "Mercator Ocean International",
            "source": "OceanLens CNO v2_loggrad + FM v4_s1_logit_t",
            "domain": domain,
        }
    )
    _add_variable_attrs(ds)
    metadata = {
        "domain": domain,
        "variables": VARIABLES,
        "solver": "euler",
        "n_steps": n_steps,
        "ensemble_members": 1,
        "noise_sigma": 1.0,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "upsample_factor": upsample_factor,
        "device": str(device),
        "input_lr_shape": list(lr.shape),
        "output_hr_shape": list(pred.shape),
    }
    return ds, metadata


def _extract_time(ds: xr.Dataset) -> np.datetime64:
    if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
        return ds["time"].values[0].astype("datetime64[ns]")
    return np.datetime64("today", "ns")


def _add_variable_attrs(ds: xr.Dataset):
    attrs = {
        "thetao": ("sea_water_potential_temperature", "Temperature", "degrees_C"),
        "so": ("sea_water_salinity", "Salinity", "1e-3"),
        "zos": ("sea_surface_height_above_geoid", "Sea surface height", "m"),
        "uo": ("eastward_sea_water_velocity", "Eastward velocity", "m s-1"),
        "vo": ("northward_sea_water_velocity", "Northward velocity", "m s-1"),
    }
    for variable, (standard_name, long_name, units) in attrs.items():
        ds[variable].attrs.update({"standard_name": standard_name, "long_name": long_name, "units": units})
    ds["latitude"].attrs.update({"standard_name": "latitude", "units": "degrees_north", "axis": "Y"})
    ds["longitude"].attrs.update({"standard_name": "longitude", "units": "degrees_east", "axis": "X"})
