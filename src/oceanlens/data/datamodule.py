"""PyTorch Lightning data module for GLORYS patches."""

import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset


VARIABLES = ["thetao", "so", "zos", "uo", "vo"]


class GlorysDataset(Dataset):
    """Return {lr, hr, mask} patch triplets."""

    def __init__(self, raw_dir, coarsened_dir, years, variables=None, patch_size=None, stats=None):
        self.variables = variables or VARIABLES
        self.patch_size = patch_size
        self.stats = stats
        self.index = []
        self.hr_files = {}
        self.lr_files = {}
        self._datasets = {}

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

    def _normalize(self, data):
        if self.stats is None:
            return data
        for channel, var in enumerate(self.variables):
            data[channel] = (data[channel] - self.stats[var]["mean"]) / (self.stats[var]["std"] + 1e-8)
        return data

    def __getitem__(self, idx):
        year, time_idx = self.index[idx]
        hr = self._load_snapshot(self.hr_files[year], time_idx)
        lr = self._load_snapshot(self.lr_files[year], time_idx)

        ocean_mask = np.all(~np.isnan(hr), axis=0, keepdims=True).astype(np.float32)
        hr = np.nan_to_num(hr, nan=0.0)
        lr = np.nan_to_num(lr, nan=0.0)
        hr = self._normalize(hr)
        lr = self._normalize(lr)
        hr = hr * ocean_mask

        hr = torch.from_numpy(hr)
        lr = torch.from_numpy(lr)
        ocean_mask = torch.from_numpy(ocean_mask)

        lr_upsampled = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode="nearest").squeeze(0)
        lr_upsampled = lr_upsampled * ocean_mask

        if self.patch_size is not None:
            ph, pw = self.patch_size
            _, height, width = hr.shape
            if height >= ph and width >= pw:
                top = torch.randint(0, height - ph + 1, (1,)).item()
                left = torch.randint(0, width - pw + 1, (1,)).item()
                hr = hr[:, top:top + ph, left:left + pw]
                lr_upsampled = lr_upsampled[:, top:top + ph, left:left + pw]
                ocean_mask = ocean_mask[:, top:top + ph, left:left + pw]

        return {"lr": lr_upsampled, "hr": hr, "mask": ocean_mask}


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
            self.train_dataset = GlorysDataset(years=list(self.cfg.data.train_years), **common)
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
