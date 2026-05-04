"""Plot residual diagnostics for all OceanLens variables from an inference NPZ."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIABLES = ["thetao", "so", "zos", "uo", "vo"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def masked_channel(data, key, mask, channel):
    arr = data[key][channel].astype(np.float32)
    return np.where(mask, arr, np.nan)


def main():
    args = parse_args()
    data = np.load(args.npz)
    mask = data["mask"][0] > 0.5

    cols = [
        ("HR - LR", "target_total_residual"),
        ("CNO residual", "cno_residual"),
        ("FM residual", "fm_residual"),
    ]

    fig, axes = plt.subplots(len(VARIABLES), len(cols), figsize=(18, 18), constrained_layout=True)
    if args.title:
        fig.suptitle(args.title, fontsize=16)

    last_image = None
    for row, variable in enumerate(VARIABLES):
        fields = [masked_channel(data, key, mask, row) for _, key in cols]
        values = np.concatenate([field[np.isfinite(field)].ravel() for field in fields])
        vmax = np.nanpercentile(np.abs(values), args.percentile)
        vmax = max(float(vmax), 1.0e-8)
        for col, ((name, _), field) in enumerate(zip(cols, fields)):
            ax = axes[row, col]
            last_image = ax.imshow(field, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{variable} | {name}")
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
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    fig.colorbar(last_image, ax=axes, shrink=0.72, label="physical residual")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
