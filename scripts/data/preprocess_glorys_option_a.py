#!/usr/bin/env python
"""Build OceanLens_git annual HR/LR NetCDF files from monthly GLORYS files."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


VARIABLES = ["thetao", "so", "zos", "uo", "vo"]


def _year_files(input_dir: Path, year: int) -> list[Path]:
    files = sorted((input_dir / str(year)).glob("*.nc"))
    if not files:
        files = sorted(input_dir.glob(f"**/*{year}*.nc"))
    if not files:
        raise FileNotFoundError(f"No monthly GLORYS files found for {year} under {input_dir}")
    return files


def _select_surface(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    selected = {}
    for variable in variables:
        data = ds[variable]
        if "depth" in data.dims:
            data = data.isel(depth=0, drop=True)
        selected[variable] = data.astype("float32")
    return xr.Dataset(selected, coords={name: ds.coords[name] for name in ds.coords if name != "depth"})


def _open_year(files: list[Path], variables: list[str]) -> xr.Dataset:
    datasets = []
    for path in files:
        ds = xr.open_dataset(path)
        datasets.append(_select_surface(ds, variables))
    try:
        year_ds = xr.concat(datasets, dim="time").sortby("time")
        # xarray concat can preserve duplicate boundary times depending on CMEMS interval semantics.
        year_ds = year_ds.isel(time=~year_ds.get_index("time").duplicated())
        return year_ds
    finally:
        for ds in datasets:
            ds.close()


def _encoding(ds: xr.Dataset, complevel: int) -> dict[str, dict[str, object]]:
    encoding: dict[str, dict[str, object]] = {}
    for name, arr in ds.data_vars.items():
        chunksizes = []
        for dim in arr.dims:
            size = arr.sizes[dim]
            chunksizes.append(1 if dim == "time" else min(size, 512))
        encoding[name] = {
            "zlib": True,
            "complevel": complevel,
            "dtype": "float32",
            "chunksizes": tuple(chunksizes),
        }
    return encoding


def _write_dataset(ds: xr.Dataset, path: Path, complevel: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    ds.to_netcdf(tmp, engine="netcdf4", encoding=_encoding(ds, complevel))
    tmp.replace(path)


def _build_lr(hr: xr.Dataset, factor: int) -> xr.Dataset:
    # boundary="pad" keeps the final partial latitude band, giving 2041 -> 681.
    lr = hr.coarsen(latitude=factor, longitude=factor, boundary="pad").mean(skipna=True)
    return lr.astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/ec/res4/scratch/fra0606/work/OceanLens/data/raw_monthly")
    parser.add_argument("--raw-dir", default="/ec/res4/scratch/fra0606/work/OceanLens/data/raw")
    parser.add_argument(
        "--coarsened-dir",
        default="/ec/res4/scratch/fra0606/work/OceanLens/data/coarsened/coarsened15",
    )
    parser.add_argument("--start-year", type=int, default=1994)
    parser.add_argument("--end-year", type=int, default=2004)
    parser.add_argument("--coarse-factor", type=int, default=3)
    parser.add_argument("--complevel", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    raw_dir = Path(args.raw_dir)
    coarsened_dir = Path(args.coarsened_dir)

    for year in range(args.start_year, args.end_year + 1):
        hr_path = raw_dir / f"glorys12_surface_{year}.nc"
        lr_path = coarsened_dir / f"glorys_coarse15_{year}.nc"
        if hr_path.exists() and lr_path.exists() and not args.overwrite:
            print(f"[skip] {year}: outputs already exist", flush=True)
            continue

        files = _year_files(input_dir, year)
        print(f"[open] {year}: {len(files)} files", flush=True)
        hr = _open_year(files, VARIABLES)
        try:
            print(f"[shape] HR {year}: {dict(hr.sizes)}", flush=True)
            if args.overwrite or not hr_path.exists():
                print(f"[write] {hr_path}", flush=True)
                _write_dataset(hr, hr_path, args.complevel)

            if args.overwrite or not lr_path.exists():
                lr = _build_lr(hr, args.coarse_factor)
                try:
                    print(f"[shape] LR {year}: {dict(lr.sizes)}", flush=True)
                    print(f"[write] {lr_path}", flush=True)
                    _write_dataset(lr, lr_path, args.complevel)
                finally:
                    lr.close()
        finally:
            hr.close()

    print("[done] annual OceanLens files are ready", flush=True)


if __name__ == "__main__":
    main()
