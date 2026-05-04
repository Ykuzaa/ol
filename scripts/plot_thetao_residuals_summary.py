"""Build one summary figure comparing thetao residual maps across variants."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", nargs="+", required=True, help="label=npz_path pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--title", default="Thetao residual comparison across variants")
    return parser.parse_args()


def masked_channel(data, key, mask, channel=0):
    arr = data[key][channel].astype(np.float32)
    return np.where(mask, arr, np.nan)


def parse_item(text):
    if "=" not in text:
        raise ValueError(f"Expected label=npz_path, got: {text}")
    label, path = text.split("=", 1)
    return label, Path(path)


def fields_for_npz(path):
    data = np.load(path)
    mask = data["mask"][0] > 0.5
    hr_lr = masked_channel(data, "target_total_residual", mask)
    if "cno_residual" in data:
        cno = masked_channel(data, "cno_residual", mask)
        cno_title = "CNO residual"
    else:
        cno = np.zeros_like(hr_lr)
        cno_title = "CNO residual (not used)"

    if "fm_residual" in data:
        fm = masked_channel(data, "fm_residual", mask)
        fm_title = "FM residual"
    elif "correction_ablation" in data:
        fm = masked_channel(data, "correction_ablation", mask)
        fm_title = "FM residual / LR correction"
    else:
        raise KeyError(f"Expected residual field in {path}")

    return mask, [("HR - LR", hr_lr), (cno_title, cno), (fm_title, fm)]


def main():
    args = parse_args()
    items = [parse_item(item) for item in args.items]
    rows = []
    values = []
    for label, path in items:
        _, fields = fields_for_npz(path)
        rows.append((label, fields))
        for _, field in fields:
            finite = field[np.isfinite(field)]
            if finite.size:
                values.append(finite.ravel())

    vmax = np.nanpercentile(np.abs(np.concatenate(values)), args.percentile)
    vmax = max(float(vmax), 1.0e-8)

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 4.5 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (label, fields) in enumerate(rows):
        for col_idx, (field_title, field) in enumerate(fields):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(field, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            title = f"{label}\n{field_title}"
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            finite = field[np.isfinite(field)]
            if finite.size:
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

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="thetao residual")
    fig.suptitle(args.title, fontsize=16)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
