import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from s3_upload import save_bytes_to_s3


PLOT_CONFIG = {
    "thetao": {"cmap": "turbo", "label": "Temperature (degC)"},
    "so": {"cmap": "viridis", "label": "Salinity"},
    "zos": {"cmap": "coolwarm", "label": "Sea surface height (m)"},
    "speed": {"cmap": "magma", "label": "Surface speed (m s-1)"},
}


def _field(ds, variable: str):
    if variable == "speed":
        return np.sqrt(ds["uo"].isel(time=0).values ** 2 + ds["vo"].isel(time=0).values ** 2)
    return ds[variable].isel(time=0).values


def _render(ds, variable: str, title: str) -> bytes:
    config = PLOT_CONFIG[variable]
    data = _field(ds, variable)
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    finite = np.isfinite(data)
    vmin, vmax = np.nanpercentile(data[finite], [2, 98]) if np.any(finite) else (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = ax.imshow(
        data,
        origin="lower",
        extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        cmap=config["cmap"],
        vmin=float(vmin),
        vmax=float(vmax),
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(image, ax=ax, shrink=0.82, label=config["label"])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def generate_thumbnails(bucket_name: str, prefix: str, forecast_dataset, variables: list[str]) -> dict[str, str]:
    urls = {}
    for variable in variables:
        if variable not in PLOT_CONFIG:
            raise ValueError(f"Unsupported plot variable: {variable}. Supported: {sorted(PLOT_CONFIG)}")
        object_key = f"{prefix}/{variable}.png".lstrip("/")
        title = f"OceanLens {variable}"
        save_bytes_to_s3(bucket_name=bucket_name, object_bytes=_render(forecast_dataset, variable, title), object_key=object_key)
        urls[variable] = f"s3://{bucket_name}/{object_key}"
    return urls
