"""PyTorch Lightning data module for GLORYS patches."""

import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset


VARIABLES = ["thetao", "so", "zos", "uo", "vo"]


class GlorysDataset(Dataset):
    """Return {lr, hr, mask} patch triplets."""

    def __init__(self, raw_dir, coarsened_dir, years, variables=None, patch_size=None, stats=None, patch_sampling=None):
        self.variables = variables or VARIABLES
        self.patch_size = patch_size
        self.stats = stats
        if isinstance(patch_sampling, str):
            self.patch_sampling = {"mode": patch_sampling, "enabled": patch_sampling == "weighted"}
        else:
            self.patch_sampling = patch_sampling or {}
        self.index = []
        self.hr_files = {}
        self.lr_files = {}
        self._datasets = {}
        self._static_cache = {}

        for year in years:
            hr_path = os.path.join(raw_dir, f"glorys12_surface_{year}.nc")
            lr_path = os.path.join(coarsened_dir, f"glorys_coarse15_{year}.nc")
            if not os.path.exists(hr_path) or not os.path.exists(lr_path):
                continue

            with xr.open_dataset(hr_path) as ds_hr:
                n_times = ds_hr.sizes["time"]

            self.hr_files[year] = hr_path
            self.lr_files[year] = lr_path
            for time_index in range(n_times):
                self.index.append((year, time_index))

        if not self.index:
            raise FileNotFoundError(
                "No paired GLORYS HR/LR files found. "
                f"raw_dir={raw_dir}, coarsened_dir={coarsened_dir}, years={list(years)}"
            )

    def _static_fields(self, filepath, height, width, ocean_mask):
        key = (filepath, height, width)
        cached = self._static_cache.get(key)
        if cached is None:
            cached = {
                "dist_coast": self._distance_to_coast(ocean_mask),
                "region_id": self._region_id(filepath, height, width, ocean_mask),
                "geo_coords": self._geo_coordinates(filepath, height, width, ocean_mask),
            }
            self._static_cache[key] = cached
        return cached

    def __len__(self):
        return len(self.index)

    def close(self):
        """Close cached xarray datasets."""
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()

    def __del__(self):
        self.close()

    def _dataset(self, filepath):
        ds = self._datasets.get(filepath)
        if ds is None:
            ds = xr.open_dataset(filepath)
            self._datasets[filepath] = ds
        return ds

    def _load_snapshot(self, filepath, time_idx):
        ds = self._dataset(filepath)
        arrays = []
        for var in self.variables:
            data = ds[var].isel(time=time_idx)
            if "depth" in data.dims:
                data = data.isel(depth=0)
            arrays.append(data.values)
        return np.stack(arrays, axis=0).astype(np.float32)

    def _load_optional_snapshot(self, filepath, variable, time_idx, shape):
        ds = self._dataset(filepath)
        if variable not in ds:
            return np.zeros((1, *shape), dtype=np.float32)
        data = ds[variable].isel(time=time_idx)
        if "depth" in data.dims:
            data = data.isel(depth=0)
        values = np.nan_to_num(data.values.astype(np.float32), nan=0.0)
        return values[None, ...]

    def _coord_values(self, filepath, height, width):
        ds = self._dataset(filepath)
        lat = None
        lon = None
        for name in ("latitude", "lat", "y"):
            if name in ds.coords:
                lat = np.asarray(ds.coords[name].values)
                break
        for name in ("longitude", "lon", "x"):
            if name in ds.coords:
                lon = np.asarray(ds.coords[name].values)
                break
        if lat is None or lat.ndim != 1 or lat.size != height:
            lat = np.linspace(-90.0, 90.0, height, dtype=np.float32)
        if lon is None or lon.ndim != 1 or lon.size != width:
            lon = np.linspace(-180.0, 180.0, width, endpoint=False, dtype=np.float32)
        lon = ((lon + 180.0) % 360.0) - 180.0
        return lat, lon

    def _normalize(self, data):
        if self.stats is None:
            return data
        for channel, var in enumerate(self.variables):
            data[channel] = (data[channel] - self.stats[var]["mean"]) / (self.stats[var]["std"] + 1e-8)
        return data

    def _region_ocean_fraction(self, top, left, ph, pw, lat, lon, ocean_mask, region):
        lat_min, lat_max, lon_min, lon_max = region
        patch_mask = ocean_mask[0, top:top + ph, left:left + pw]
        patch_lat = lat[top:top + ph]
        patch_lon = lon[left:left + pw]
        lat_sel = (patch_lat >= lat_min) & (patch_lat <= lat_max)
        lon_sel = (patch_lon >= lon_min) & (patch_lon <= lon_max)
        if not lat_sel.any() or not lon_sel.any():
            return 0.0
        region_mask = patch_mask[np.ix_(lat_sel, lon_sel)]
        if region_mask.size == 0:
            return 0.0
        return float(region_mask.mean())

    def _patch_category_fractions(self, top, left, ph, pw, lat, lon, ocean_mask, siconc=None):
        patch_mask = ocean_mask[0, top : top + ph, left : left + pw] > 0.5
        patch_lat = lat[top : top + ph]
        patch_lon = lon[left : left + pw]
        lat_grid, lon_grid = np.meshgrid(patch_lat, patch_lon, indexing="ij")
        ocean = patch_mask
        denom = max(float(ocean.sum()), 1.0)

        coastal = self._coastal_band(ocean_mask[:, top : top + ph, left : left + pw])[0] > 0.5
        med = (lat_grid >= 30.0) & (lat_grid <= 46.5) & (lon_grid >= -6.0) & (lon_grid <= 37.0)
        black = (lat_grid >= 40.0) & (lat_grid <= 48.5) & (lon_grid >= 27.0) & (lon_grid <= 42.5)
        arctic = lat_grid >= 60.0
        antarctic = lat_grid <= -55.0
        ice = np.zeros_like(ocean, dtype=bool)
        if siconc is not None:
            ice = siconc[0, top : top + ph, left : left + pw] > 0.15

        return {
            "coastal": float((coastal & ocean).sum()) / denom,
            "mediterranean": float((med & ocean).sum()) / denom,
            "black_sea": float((black & ocean).sum()) / denom,
            "arctic_ice": float((arctic & ice & ocean).sum()) / denom,
            "arctic_open": float((arctic & ~ice & ocean).sum()) / denom,
            "antarctic": float((antarctic & ocean).sum()) / denom,
            "open_ocean": float(ocean.sum()) / denom,
        }

    def _patch_weight(self, top, left, ph, pw, height, width, lat, lon, ocean_mask, siconc=None):
        cfg = self.patch_sampling
        weights_cfg = getattr(cfg, "weights", cfg.get("weights", None))
        if weights_cfg is not None:
            fractions = self._patch_category_fractions(top, left, ph, pw, lat, lon, ocean_mask, siconc=siconc)
            weight = 0.0
            for name, fraction in fractions.items():
                category_weight = float(getattr(weights_cfg, name, weights_cfg.get(name, 1.0)))
                weight += category_weight * fraction
            return max(weight, 1.0e-6)

        default_weight = float(getattr(cfg, "default_weight", cfg.get("default_weight", 1.0)))
        med_black_weight = float(getattr(cfg, "med_black_weight", cfg.get("med_black_weight", 5.0)))
        arctic_weight = float(getattr(cfg, "arctic_weight", cfg.get("arctic_weight", 3.0)))
        min_region_ocean_fraction = float(
            getattr(cfg, "min_region_ocean_fraction", cfg.get("min_region_ocean_fraction", 0.3))
        )

        weight = default_weight
        med_fraction = self._region_ocean_fraction(top, left, ph, pw, lat, lon, ocean_mask, (30.0, 46.5, -6.0, 37.0))
        black_fraction = self._region_ocean_fraction(top, left, ph, pw, lat, lon, ocean_mask, (40.0, 48.5, 27.0, 42.5))
        arctic_fraction = self._region_ocean_fraction(top, left, ph, pw, lat, lon, ocean_mask, (60.0, 90.0, -180.0, 180.0))
        if max(med_fraction, black_fraction) >= min_region_ocean_fraction:
            weight *= med_black_weight
        if arctic_fraction >= min_region_ocean_fraction:
            weight *= arctic_weight
        return max(weight, 1.0e-6)

    def _sample_patch_origin(self, filepath, height, width, ph, pw, ocean_mask, siconc=None):
        cfg = self.patch_sampling
        mode = getattr(cfg, "mode", None)
        enabled = bool(getattr(cfg, "enabled", cfg.get("enabled", False))) or cfg == "weighted" or mode == "weighted"
        if not enabled:
            top = torch.randint(0, height - ph + 1, (1,)).item()
            left = torch.randint(0, width - pw + 1, (1,)).item()
            return top, left

        num_candidates = int(getattr(cfg, "num_candidates", cfg.get("num_candidates", 32)))
        lat, lon = self._coord_values(filepath, height, width)
        candidates = []
        weights = []
        for _ in range(max(1, num_candidates)):
            top = torch.randint(0, height - ph + 1, (1,)).item()
            left = torch.randint(0, width - pw + 1, (1,)).item()
            candidates.append((top, left))
            weights.append(self._patch_weight(top, left, ph, pw, height, width, lat, lon, ocean_mask, siconc=siconc))

        weights_t = torch.as_tensor(weights, dtype=torch.float32)
        index = torch.multinomial(weights_t, num_samples=1).item()
        return candidates[index]

    def _coastal_band(self, ocean_mask):
        ocean = ocean_mask[0] > 0.5
        dist_ocean = distance_transform_edt(ocean)
        dist_land = distance_transform_edt(~ocean)
        coast_distance = np.minimum(dist_ocean, dist_land)
        return (coast_distance <= 3.0).astype(np.float32)[None, ...]

    def _distance_to_coast(self, ocean_mask):
        ocean = ocean_mask[0] > 0.5
        dist_ocean = distance_transform_edt(ocean).astype(np.float32)
        scale = np.percentile(dist_ocean[ocean], 99.0) if ocean.any() else 1.0
        scale = max(float(scale), 1.0)
        dist = np.clip(dist_ocean / scale, 0.0, 1.0)
        return (dist[None, ...] * ocean_mask).astype(np.float32)

    def _region_id(self, filepath, height, width, ocean_mask):
        lat, lon = self._coord_values(filepath, height, width)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        med = (lat_grid >= 30.0) & (lat_grid <= 46.5) & (lon_grid >= -6.0) & (lon_grid <= 37.0)
        black = (lat_grid >= 40.0) & (lat_grid <= 48.5) & (lon_grid >= 27.0) & (lon_grid <= 42.5)
        arctic = lat_grid >= 60.0
        antarctic = lat_grid <= -55.0
        equatorial = (lat_grid >= -10.0) & (lat_grid <= 10.0)
        coastal = self._coastal_band(ocean_mask)[0] > 0.5
        ocean = ocean_mask[0] > 0.5
        special = med | black | arctic | antarctic | equatorial | coastal
        open_ocean = ocean & ~special
        regions = np.stack(
            [
                open_ocean,
                med & ocean,
                black & ocean,
                arctic & ocean,
                antarctic & ocean,
                equatorial & ocean,
                coastal & ocean,
            ],
            axis=0,
        )
        return regions.astype(np.float32)

    def _geo_coordinates(self, filepath, height, width, ocean_mask):
        lat, lon = self._coord_values(filepath, height, width)
        lat_norm = np.clip(lat.astype(np.float32) / 90.0, -1.0, 1.0)
        lon_rad = np.deg2rad(lon.astype(np.float32))
        lat_grid = np.broadcast_to(lat_norm[:, None], (height, width))
        lon_sin = np.broadcast_to(np.sin(lon_rad)[None, :], (height, width))
        lon_cos = np.broadcast_to(np.cos(lon_rad)[None, :], (height, width))
        coords = np.stack([lat_grid, lon_sin, lon_cos], axis=0).astype(np.float32)
        return coords * ocean_mask

    def __getitem__(self, idx):
        year, time_idx = self.index[idx]
        hr = self._load_snapshot(self.hr_files[year], time_idx)
        lr = self._load_snapshot(self.lr_files[year], time_idx)

        ocean_mask = np.all(~np.isnan(hr), axis=0, keepdims=True).astype(np.float32)
        siconc_hr = self._load_optional_snapshot(self.hr_files[year], "siconc", time_idx, hr.shape[-2:])
        hr = np.nan_to_num(hr, nan=0.0)
        lr = np.nan_to_num(lr, nan=0.0)
        hr = self._normalize(hr)
        lr = self._normalize(lr)
        hr = hr * ocean_mask
        siconc_hr = np.clip(np.nan_to_num(siconc_hr, nan=0.0), 0.0, 1.0) * ocean_mask
        static = self._static_fields(self.hr_files[year], hr.shape[-2], hr.shape[-1], ocean_mask)
        dist_coast = static["dist_coast"]
        region_id = static["region_id"]
        geo_coords = static["geo_coords"]

        hr = torch.from_numpy(hr)
        lr_native = torch.from_numpy(lr)
        ocean_mask = torch.from_numpy(ocean_mask)
        siconc = torch.from_numpy(siconc_hr)
        dist_coast = torch.from_numpy(dist_coast)
        region_id = torch.from_numpy(region_id)
        geo_coords = torch.from_numpy(geo_coords)

        lr_upsampled = F.interpolate(lr_native.unsqueeze(0), size=hr.shape[-2:], mode="nearest").squeeze(0)
        lr_upsampled = lr_upsampled * ocean_mask

        if self.patch_size is not None:
            ph, pw = self.patch_size
            _, height, width = hr.shape
            if height >= ph and width >= pw:
                top, left = self._sample_patch_origin(
                    self.hr_files[year],
                    height,
                    width,
                    ph,
                    pw,
                    ocean_mask.numpy(),
                    siconc=siconc.numpy(),
                )
                hr = hr[:, top:top + ph, left:left + pw]
                lr_upsampled = lr_upsampled[:, top:top + ph, left:left + pw]
                ocean_mask = ocean_mask[:, top:top + ph, left:left + pw]
                siconc = siconc[:, top:top + ph, left:left + pw]
                dist_coast = dist_coast[:, top:top + ph, left:left + pw]
                region_id = region_id[:, top:top + ph, left:left + pw]
                geo_coords = geo_coords[:, top:top + ph, left:left + pw]
                lr_height, lr_width = lr_native.shape[-2:]
                lr_ph = max(1, int(round(ph * lr_height / height)))
                lr_pw = max(1, int(round(pw * lr_width / width)))
                lr_top = int(round(top * lr_height / height))
                lr_left = int(round(left * lr_width / width))
                lr_bottom = int(round((top + ph) * lr_height / height))
                lr_right = int(round((left + pw) * lr_width / width))
                lr_bottom = max(lr_bottom, lr_top + 1)
                lr_right = max(lr_right, lr_left + 1)
                lr_bottom = min(lr_bottom, lr_height)
                lr_right = min(lr_right, lr_width)
                lr_native = lr_native[:, lr_top:lr_bottom, lr_left:lr_right]
                if lr_native.shape[-2:] != (lr_ph, lr_pw):
                    lr_native = F.interpolate(
                        lr_native.unsqueeze(0),
                        size=(lr_ph, lr_pw),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

        return {
            "lr": lr_upsampled,
            "lr_native": lr_native,
            "hr": hr,
            "mask": ocean_mask,
            "siconc": siconc,
            "dist_coast": dist_coast,
            "region_id": region_id,
            "lat": geo_coords[0:1],
            "lon_sin": geo_coords[1:2],
            "lon_cos": geo_coords[2:3],
        }


class OceanDataModule(pl.LightningDataModule):
    """Build train and validation dataloaders."""

    def __init__(self, cfg, phase="cno"):
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        self.stats = None

        stats_path = cfg.paths.norm_stats
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Missing normalization stats: {stats_path}")
        with open(stats_path) as handle:
            self.stats = json.load(handle)

    def setup(self, stage=None):
        patch = (self.cfg.data.patch_size, self.cfg.data.patch_size)
        common = dict(
            raw_dir=self.cfg.paths.raw_dir,
            coarsened_dir=self.cfg.paths.coarsened_dir,
            variables=list(self.cfg.variables),
            patch_size=patch,
            stats=self.stats,
        )

        if stage in ("fit", None):
            train_common = dict(common)
            train_common["patch_sampling"] = getattr(self.cfg.data, "patch_sampling", {})
            self.train_dataset = GlorysDataset(years=list(self.cfg.data.train_years), **train_common)
            self.val_dataset = GlorysDataset(years=list(self.cfg.data.val_years), **common)

    def train_dataloader(self):
        tcfg = self.cfg.training[self.phase]
        return DataLoader(
            self.train_dataset,
            batch_size=tcfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        tcfg = self.cfg.training[self.phase]
        return DataLoader(
            self.val_dataset,
            batch_size=tcfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
        )
