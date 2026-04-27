from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.paths import RANDOM_SPAN_STATS_DIR, RANDOM_SPAN_STATS_FIGURES_DIR


SUMMARY_CSV = RANDOM_SPAN_STATS_DIR / "metrics_summary_mean_std.csv"
METHODS = [
    "linear_interp",
    "savgol_w5_p2",
    "kalman_cv_dt1.0_q1e-3_r1e-2",
]
METHOD_LABELS = {
    "linear_interp": "Linear",
    "savgol_w5_p2": "SG",
    "kalman_cv_dt1.0_q1e-3_r1e-2": "Kalman",
}
COLORS = {
    "linear_interp": "#4C78A8",
    "savgol_w5_p2": "#72B7B2",
    "kalman_cv_dt1.0_q1e-3_r1e-2": "#595959",
}
HATCHES = {
    "linear_interp": "",
    "savgol_w5_p2": "//",
    "kalman_cv_dt1.0_q1e-3_r1e-2": "xx",
}


def load_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"method", "metric", "mean", "std"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Summary CSV missing columns: {sorted(missing)}")
    return rows


def get_stat(rows, method: str, metric: str, key: str):
    matches = [row for row in rows if row["method"] == method and row["metric"] == metric]
    if not matches:
        raise ValueError(f"No summary row found for method={method!r}, metric={metric!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple summary rows found for method={method!r}, metric={metric!r}")
    return float(matches[0][key])


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_metric_bar(rows, output_dir: Path, metric: str, stem: str, title: str):
    x = np.arange(len(METHODS))
    means = [get_stat(rows, method, metric, "mean") for method in METHODS]
    stds = [get_stat(rows, method, metric, "std") for method in METHODS]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[COLORS[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, method in zip(bars, METHODS):
        bar.set_hatch(HATCHES[method])

    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def plot_full_vs_masked(rows, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=False)
    for ax, metric, title in [
        (axes[0], "ADE", "Full-Trajectory View"),
        (axes[1], "masked_ADE", "Missing-Segment View"),
    ]:
        x = np.arange(len(METHODS))
        means = [get_stat(rows, method, metric, "mean") for method in METHODS]
        stds = [get_stat(rows, method, metric, "std") for method in METHODS]
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=[COLORS[m] for m in METHODS],
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, method in zip(bars, METHODS):
            bar.set_hatch(HATCHES[method])
        ax.set_title(title)
        ax.set_xlabel("Method")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
        ax.grid(axis="y", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Random-Span Full vs Masked Comparison", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, output_dir, "full_vs_masked_comparison")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Stage 3 Phase 1 random-span statistics figures."
    )
    parser.add_argument("--summary_csv", type=str, default=str(SUMMARY_CSV))
    parser.add_argument("--output_dir", type=str, default=str(RANDOM_SPAN_STATS_FIGURES_DIR))
    args = parser.parse_args()

    rows = load_rows(Path(args.summary_csv))
    output_dir = Path(args.output_dir)
    generated = [
        plot_metric_bar(
            rows,
            output_dir,
            metric="ADE",
            stem="ADE_mean_std_bar",
            title="Random-Span ADE Mean +- Std",
        ),
        plot_metric_bar(
            rows,
            output_dir,
            metric="RMSE",
            stem="RMSE_mean_std_bar",
            title="Random-Span RMSE Mean +- Std",
        ),
        plot_metric_bar(
            rows,
            output_dir,
            metric="masked_ADE",
            stem="masked_ADE_mean_std_bar",
            title="Random-Span masked_ADE Mean +- Std",
        ),
        plot_full_vs_masked(rows, output_dir),
    ]

    print("=" * 60)
    print("Stage 3 random-span figures generated")
    print(f"summary_csv = {args.summary_csv}")
    for path in generated:
        print(f"figure      = {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
