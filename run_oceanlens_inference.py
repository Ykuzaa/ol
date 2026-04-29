import json
import os
import sys
from pathlib import Path

import s3fs
import xarray as xr

from generate_thumbnails import generate_thumbnails
from model import synchronize_model_locally
from oceanlens_inference import run_oceanlens_super_resolution
from s3_upload import get_s3_endpoint_url_with_protocol, save_large_file_to_s3


LOCAL_MODEL_DIR = "/tmp/oceanlens"


def _split_s3_folder(folder: str) -> tuple[str, str]:
    parts = folder.replace("s3://", "", 1).strip("/").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _s3_path_from_url(url: str) -> str:
    endpoint = get_s3_endpoint_url_with_protocol().rstrip("/") + "/"
    if url.startswith("s3://"):
        return url.replace("s3://", "", 1)
    if url.startswith(endpoint):
        return url[len(endpoint) :].lstrip("/")
    parts = url.replace("https://", "", 1).split("/", 1)
    return parts[1] if len(parts) > 1 else parts[0]


def _open_input_dataset(url: str) -> xr.Dataset:
    s3 = s3fs.S3FileSystem(
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"],
        client_kwargs={"endpoint_url": get_s3_endpoint_url_with_protocol()},
    )
    s3_path = _s3_path_from_url(url)
    print(f"Opening LR input: {s3_path}")
    with s3.open(s3_path, "rb") as handle:
        return xr.open_dataset(handle, engine="h5netcdf").load()


def _date_label(ds: xr.Dataset) -> str:
    if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
        return str(ds["time"].values[0].astype("datetime64[D]"))
    return "no_date"


def _plot_variables() -> list[str]:
    raw = os.environ.get("PLOT_VARIABLES", "thetao,so,zos,speed")
    variables = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return variables or ["thetao", "so", "zos", "speed"]


def main():
    print("=" * 60)
    print("OceanLens On-Demand Super-Resolution")
    print("=" * 60)

    lr_file_url = os.environ.get("LR_FILE_URL")
    if not lr_file_url:
        print("Error: LR_FILE_URL is required")
        sys.exit(1)

    s3_output_folder = os.environ.get("S3_OUTPUT_FOLDER")
    if not s3_output_folder:
        bucket_name = os.environ.get("AWS_BUCKET_NAME")
        if not bucket_name:
            print("Error: S3_OUTPUT_FOLDER or AWS_BUCKET_NAME is required")
            sys.exit(1)
        s3_output_folder = f"{bucket_name}/OceanLens"

    domain = os.environ.get("DOMAIN", "ibi").lower()
    n_steps = int(os.environ.get("N_STEPS", "20"))
    upsample_factor = int(os.environ.get("UPSCALE_FACTOR", "15"))
    plot_variables = _plot_variables()
    bucket_name, base_prefix = _split_s3_folder(s3_output_folder)

    print(f"Domain: {domain}")
    print(f"Plot variables: {plot_variables}")
    print(f"Output folder: s3://{bucket_name}/{base_prefix}")

    print("\n--- Downloading OceanLens assets ---")
    synchronize_model_locally(LOCAL_MODEL_DIR)

    print("\n--- Loading LR input ---")
    lr_dataset = _open_input_dataset(lr_file_url)
    date_label = _date_label(lr_dataset)

    print("\n--- Running inference ---")
    forecast_dataset, metadata = run_oceanlens_super_resolution(
        lr_dataset=lr_dataset,
        model_dir=LOCAL_MODEL_DIR,
        domain=domain,
        n_steps=n_steps,
        upsample_factor=upsample_factor,
        tile_size=int(os.environ.get("TILE_SIZE", "512")),
        tile_overlap=int(os.environ.get("TILE_OVERLAP", "64")),
    )
    metadata.update({"date": date_label, "lr_file_url": lr_file_url, "plot_variables": plot_variables})

    run_prefix = f"{base_prefix}/{date_label}/{domain}".strip("/")
    file_name = f"OceanLens_{date_label}_{domain}.nc"
    local_nc = f"/tmp/{file_name}"
    object_key = f"{run_prefix}/{file_name}".lstrip("/")

    print("\n--- Saving NetCDF ---")
    encoding = {variable: {"zlib": True, "complevel": 4} for variable in forecast_dataset.data_vars}
    forecast_dataset.to_netcdf(local_nc, encoding=encoding, engine="h5netcdf")
    save_large_file_to_s3(bucket_name=bucket_name, local_file_path=local_nc, object_key=object_key)
    os.remove(local_nc)

    print("\n--- Generating thumbnails ---")
    thumbnail_urls = generate_thumbnails(bucket_name, run_prefix, forecast_dataset, plot_variables)

    metadata["netcdf"] = f"s3://{bucket_name}/{object_key}"
    metadata["thumbnails"] = thumbnail_urls
    metadata_key = f"{run_prefix}/metadata.json".lstrip("/")
    metadata_path = Path("/tmp/oceanlens_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    save_large_file_to_s3(bucket_name=bucket_name, local_file_path=str(metadata_path), object_key=metadata_key)
    metadata_path.unlink()

    print("\n" + "=" * 60)
    print(f"OceanLens inference completed: s3://{bucket_name}/{object_key}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        print(f"\nError: {exc}")
        traceback.print_exc()
        sys.exit(1)
