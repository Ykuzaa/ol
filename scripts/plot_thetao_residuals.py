"""Plot thetao residual diagnostics from an OceanLens inference NPZ."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def masked_channel(data, key, mask, channel=0):
    arr = data[key][channel].astype(np.float32)
    return np.where(mask, arr, np.nan)


def main():
    args = parse_args()
    data = np.load(args.npz)
    mask = data["mask"][0] > 0.5

    if "cno_residual" in data:
        cno_title = "CNO residual"
        cno_field = masked_channel(data, "cno_residual", mask)
    else:
        cno_title = "CNO residual (not used)"
        cno_field = np.zeros_like(masked_channel(data, "target_total_residual", mask))

    if "fm_residual" in data:
        fm_title = "FM residual"
        fm_field = masked_channel(data, "fm_residual", mask)
    elif "correction_ablation" in data:
        fm_title = "FM residual / LR correction"
        fm_field = masked_channel(data, "correction_ablation", mask)
    else:
        raise KeyError("Expected either 'fm_residual' or 'correction_ablation' in NPZ.")

    fields = [
        ("HR - LR", masked_channel(data, "target_total_residual", mask)),
        (cno_title, cno_field),
        (fm_title, fm_field),
    ]

    values = np.concatenate([field[np.isfinite(field)].ravel() for _, field in fields])
    vmax = np.nanpercentile(np.abs(values), args.percentile)
    vmax = max(float(vmax), 1.0e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for ax, (title, field) in zip(axes, fields):
        image = ax.imshow(field, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        finite = field[np.isfinite(field)]
        ax.text(
            0.02,
            0.02,
            f"mean={np.mean(finite):.3g}\nstd={np.std(finite):.3g}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    fig.colorbar(image, ax=axes, shrink=0.82, label="thetao residual")
    if args.title:
        fig.suptitle(args.title, fontsize=14)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
