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


SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "stage3"
    / "phase1"
    / "canonical_room3"
    / "eval"
    / "summary_metrics.csv"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage3"
    / "phase1"
    / "canonical_room3"
    / "figures"
)

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
SPAN_EXPERIMENTS = {
    "span10_fixed_seed42": 0.1,
    "span20_fixed_seed42": 0.2,
    "span30_fixed_seed42": 0.3,
}
REQUIRED_COLUMNS = {
    "experiment_id",
    "method_tag",
    "ADE",
    "masked_ADE",
}


def load_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"summary_metrics.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"summary_metrics.csv has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"summary_metrics.csv is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"summary_metrics.csv contains no rows: {path}")
    return rows


def get_metric(rows, experiment_id: str, method_tag: str, metric: str):
    matches = [
        row
        for row in rows
        if row["experiment_id"] == experiment_id and row["method_tag"] == method_tag
    ]
    if not matches:
        raise ValueError(
            "No row found for "
            f"experiment_id={experiment_id!r}, method_tag={method_tag!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple rows found for "
            f"experiment_id={experiment_id!r}, method_tag={method_tag!r}"
        )
    try:
        return float(matches[0][metric])
    except KeyError as exc:
        raise ValueError(f"Metric column not found: {metric}") from exc
    except ValueError as exc:
        raise ValueError(
            f"Metric {metric} is not numeric for "
            f"experiment_id={experiment_id}, method_tag={method_tag}"
        ) from exc


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def plot_exp0_baseline_bar(rows, output_dir: Path):
    experiment_id = "span20_fixed_seed42"
    ade = [get_metric(rows, experiment_id, method, "ADE") for method in METHODS]
    masked_ade = [
        get_metric(rows, experiment_id, method, "masked_ADE") for method in METHODS
    ]
    labels = [METHOD_LABELS[method] for method in METHODS]

    x = np.arange(len(METHODS))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(x - width / 2, ade, width, label="ADE", color="#4C78A8")
    ax.bar(x + width / 2, masked_ade, width, label="masked_ADE", color="#72B7B2")
    ax.set_title("Experiment 0 Baseline Error Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    paths = save_figure(fig, output_dir, "exp0_baseline_ade_maskedade_bar")
    plt.close(fig)
    return paths


def plot_exp1_span_sweep(rows, output_dir: Path):
    ratios = [SPAN_EXPERIMENTS[exp] for exp in SPAN_EXPERIMENTS]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = {
        "linear_interp": "#4C78A8",
        "savgol_w5_p2": "#72B7B2",
        "kalman_cv_dt1.0_q1e-3_r1e-2": "#595959",
    }
    markers = {
        "linear_interp": "o",
        "savgol_w5_p2": "s",
        "kalman_cv_dt1.0_q1e-3_r1e-2": "^",
    }

    for method in METHODS:
        values = [
            get_metric(rows, experiment_id, method, "masked_ADE")
            for experiment_id in SPAN_EXPERIMENTS
        ]
        ax.plot(
            ratios,
            values,
            marker=markers[method],
            linewidth=1.8,
            label=METHOD_LABELS[method],
            color=colors[method],
        )

    ax.set_title("Experiment 1 Missing-Span Sweep")
    ax.set_xlabel("Missing Span Ratio")
    ax.set_ylabel("masked_ADE")
    ax.set_xticks(ratios)
    ax.legend(frameon=False)
    ax.grid(axis="both", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    paths = save_figure(fig, output_dir, "exp1_span_sweep_maskedade_line")
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot Stage 3 Phase 1 canonical room3 reporting figures."
    )
    parser.add_argument("--summary_csv", type=str, default=str(SUMMARY_CSV))
    parser.add_argument("--output_dir", type=str, default=str(FIGURE_DIR))
    args = parser.parse_args()

    rows = load_rows(Path(args.summary_csv))
    output_dir = Path(args.output_dir)
    generated = [
        *plot_exp0_baseline_bar(rows, output_dir),
        *plot_exp1_span_sweep(rows, output_dir),
    ]

    print("=" * 60)
    print("Stage 3 Phase 1 figures generated")
    print(f"summary_csv = {args.summary_csv}")
    for path in generated:
        print(f"figure      = {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
