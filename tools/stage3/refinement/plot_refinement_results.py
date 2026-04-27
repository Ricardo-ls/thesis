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

from tools.stage3.controlled.evaluate_coarse_reconstruction import plot_segments
from tools.stage3.refinement.refiners import REFINER_LABELS, REFINER_NAMES
from tools.stage3.refinement.run_refinement_interface import (
    REFINEMENT_FIGURE_DIR,
    ensure_refinement_dirs,
    refined_path,
)
from utils.stage3.controlled_benchmark import (
    DEGRADATION_LABELS,
    DEGRADATION_NAMES,
    METHOD_LABELS,
    METHODS,
    DEFAULT_SAMPLE_INDEX,
    DEFAULT_SEED,
    DEFAULT_SPAN_MODE,
    DEFAULT_SPAN_RATIO,
    clean_path,
    experiment_tag,
    mask_path,
    reconstruction_path,
)


def metrics_csv_path():
    return PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "eval" / "refinement_metrics.csv"


def parse_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = dict(row)
            for key in [
                "ADE",
                "FDE",
                "RMSE",
                "masked_ADE",
                "masked_RMSE",
                "improvement_ADE",
                "improvement_masked_ADE",
                "off_map_ratio",
            ]:
                parsed[key] = float(parsed[key])
            parsed["wall_crossing_count"] = int(row["wall_crossing_count"])
            rows.append(parsed)
    return rows


def row_lookup(rows: list[dict], degradation: str, coarse_method: str, refiner: str):
    for row in rows:
        if (
            row["degradation"] == degradation
            and row["coarse_method"] == coarse_method
            and row["refiner"] == refiner
        ):
            return row
    raise KeyError(f"Missing row for {degradation}/{coarse_method}/{refiner}")


def plot_metric_comparison(rows: list[dict], metric: str, ylabel: str, output_path: Path):
    labels = [f"{DEGRADATION_LABELS[d]}\n{METHOD_LABELS[m]}" for d in DEGRADATION_NAMES for m in METHODS]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14.8, 4.8))
    palette = {
        "coarse": "#4C78A8",
        "identity_refiner": "#9E9E9E",
        "light_savgol_refiner": "#E45756",
        "ddpm_prior_interface_v0": "#54A24B",
        "ddpm_prior_masked_replace_v1": "#B279A2",
    }

    coarse_values = [
        row_lookup(rows, degradation, coarse_method, "identity_refiner")[metric]
        for degradation in DEGRADATION_NAMES
        for coarse_method in METHODS
    ]
    identity_values = [
        row_lookup(rows, degradation, coarse_method, "identity_refiner")[metric]
        for degradation in DEGRADATION_NAMES
        for coarse_method in METHODS
    ]
    sg_values = [
        row_lookup(rows, degradation, coarse_method, "light_savgol_refiner")[metric]
        for degradation in DEGRADATION_NAMES
        for coarse_method in METHODS
    ]
    ddpm_values = [
        row_lookup(rows, degradation, coarse_method, "ddpm_prior_interface_v0")[metric]
        for degradation in DEGRADATION_NAMES
        for coarse_method in METHODS
    ]
    ddpm_masked_values = [
        row_lookup(rows, degradation, coarse_method, "ddpm_prior_masked_replace_v1")[metric]
        for degradation in DEGRADATION_NAMES
        for coarse_method in METHODS
    ]

    ax.bar(x - 2.0 * width, coarse_values, width=width, color=palette["coarse"], label="Coarse")
    ax.bar(x - 1.0 * width, identity_values, width=width, color=palette["identity_refiner"], label="Identity")
    ax.bar(x, sg_values, width=width, color=palette["light_savgol_refiner"], label="Light SG")
    ax.bar(x + 1.0 * width, ddpm_values, width=width, color=palette["ddpm_prior_interface_v0"], label="DDPM prior v0")
    ax.bar(
        x + 2.0 * width,
        ddpm_masked_values,
        width=width,
        color=palette["ddpm_prior_masked_replace_v1"],
        label="DDPM masked replace v1",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_improvement(rows: list[dict], output_path: Path):
    labels = [f"{DEGRADATION_LABELS[d]}\n{METHOD_LABELS[m]}" for d in DEGRADATION_NAMES for m in METHODS]
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(14.0, 4.8))
    light_imp = [
        100.0 * row_lookup(rows, degradation, coarse_method, "light_savgol_refiner")["improvement_ADE"]
        for degradation in DEGRADATION_NAMES
        for coarse_method in METHODS
    ]
    ddpm_v0_imp = [
        100.0 * row_lookup(rows, degradation, coarse_method, "ddpm_prior_interface_v0")["improvement_ADE"]
        for degradation in DEGRADATION_NAMES
        for coarse_method in METHODS
    ]
    ddpm_masked_imp = [
        100.0 * row_lookup(rows, degradation, coarse_method, "ddpm_prior_masked_replace_v1")["improvement_ADE"]
        for degradation in DEGRADATION_NAMES
        for coarse_method in METHODS
    ]
    ax.bar(x - width, light_imp, width=width, color="#E45756", label="Light SG")
    ax.bar(x, ddpm_v0_imp, width=width, color="#54A24B", label="DDPM prior v0")
    ax.bar(x + width, ddpm_masked_imp, width=width, color="#B279A2", label="DDPM masked replace v1")
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Improvement_ADE (%)")
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ddpm_vs_naive(rows: list[dict], output_path: Path):
    labels = [f"{DEGRADATION_LABELS[d]}\n{METHOD_LABELS[m]}" for d in DEGRADATION_NAMES for m in METHODS]
    x = np.arange(len(labels))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 4.8), sharex=True)

    values = {}
    for refiner in REFINER_NAMES:
        values[(refiner, "improvement_ADE")] = [
            100.0 * row_lookup(rows, degradation, coarse_method, refiner)["improvement_ADE"]
            for degradation in DEGRADATION_NAMES
            for coarse_method in METHODS
        ]
        values[(refiner, "improvement_masked_ADE")] = [
            100.0 * row_lookup(rows, degradation, coarse_method, refiner)["improvement_masked_ADE"]
            for degradation in DEGRADATION_NAMES
            for coarse_method in METHODS
        ]

    palette = {
        "identity_refiner": "#9E9E9E",
        "light_savgol_refiner": "#E45756",
        "ddpm_prior_interface_v0": "#54A24B",
        "ddpm_prior_masked_replace_v1": "#B279A2",
    }
    offsets = {
        "identity_refiner": -1.5 * width,
        "light_savgol_refiner": -0.5 * width,
        "ddpm_prior_interface_v0": 0.5 * width,
        "ddpm_prior_masked_replace_v1": 1.5 * width,
    }

    panels = [
        (axes[0], "improvement_ADE", "Improvement_ADE (%)"),
        (axes[1], "improvement_masked_ADE", "Improvement_masked_ADE (%)"),
    ]
    for ax, metric, ylabel in panels:
        for refiner in REFINER_NAMES:
            ax.bar(
                x + offsets[refiner],
                values[(refiner, metric)],
                width=width,
                color=palette[refiner],
                label=REFINER_LABELS[refiner],
            )
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title("Full-trajectory view")
    axes[1].set_title("Missing-segment view")
    axes[1].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(-0.1, 1.22))
    fig.suptitle("DDPM prior v0 vs naive refinement")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_full_vs_masked_improvement(rows: list[dict], output_path: Path):
    labels = [f"{DEGRADATION_LABELS[d]}\n{METHOD_LABELS[m]}" for d in DEGRADATION_NAMES for m in METHODS]
    x = np.arange(len(labels))
    width = 0.55
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 4.8), sharex=True)

    refiners_to_show = [
        "light_savgol_refiner",
        "ddpm_prior_interface_v0",
        "ddpm_prior_masked_replace_v1",
    ]
    palette = {
        "light_savgol_refiner": "#E45756",
        "ddpm_prior_interface_v0": "#54A24B",
        "ddpm_prior_masked_replace_v1": "#B279A2",
    }
    offsets = {
        "light_savgol_refiner": -0.25,
        "ddpm_prior_interface_v0": 0.0,
        "ddpm_prior_masked_replace_v1": 0.25,
    }

    panel_specs = [
        (axes[0], "improvement_ADE", "Improvement_ADE (%)", "Full-trajectory view"),
        (axes[1], "improvement_masked_ADE", "Improvement_masked_ADE (%)", "Missing-segment view"),
    ]

    for ax, metric_name, ylabel, title in panel_specs:
        for refiner in refiners_to_show:
            values = [
                100.0 * row_lookup(rows, degradation, coarse_method, refiner)[metric_name]
                for degradation in DEGRADATION_NAMES
                for coarse_method in METHODS
            ]
            ax.bar(
                x + offsets[refiner],
                values,
                width=0.24,
                color=palette[refiner],
                label=REFINER_LABELS[refiner],
            )
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(-0.05, 1.22))

    fig.suptitle("Refinement improvement: full vs masked views")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_example(sample_idx: int, tag: str, output_path: Path):
    clean = np.load(clean_path(), allow_pickle=False).astype(np.float32)
    obs_mask = np.load(mask_path(tag), allow_pickle=False).astype(np.uint8)
    degradation = "missing_only"
    coarse_method = "linear_interp"
    refiner = "light_savgol_refiner"
    coarse = np.load(reconstruction_path(degradation, coarse_method, tag), allow_pickle=False).astype(np.float32)
    refined = np.load(refined_path(degradation, coarse_method, refiner), allow_pickle=False).astype(np.float32)

    mask = obs_mask[sample_idx].astype(bool)
    gt = clean[sample_idx]
    coarse_sample = coarse[sample_idx]
    refined_sample = refined[sample_idx]
    missing_idx = np.flatnonzero(~mask)
    gap_slice = slice(max(missing_idx[0] - 1, 0), min(missing_idx[-1] + 2, gt.shape[0]))
    gt_gap = gt[gap_slice]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharex=True, sharey=True)
    fig.suptitle("Trajectory example: coarse vs refined")
    titles = ["Ground truth", "Coarse (Linear)", "Refined (Light SG)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(linewidth=0.6, alpha=0.25)

    axes[0].plot(gt[:, 0], gt[:, 1], color="#222222", linewidth=2.2)

    for ax, sample, title_color in [(axes[1], coarse_sample, "#d62728"), (axes[2], refined_sample, "#2ca02c")]:
        plot_segments(ax, gt, mask, color="#1f77b4", linewidth=2.2)
        ax.plot(gt_gap[:, 0], gt_gap[:, 1], color="#b0b0b0", linestyle="--", linewidth=1.3)
        pred_gap = sample[gap_slice]
        ax.plot(pred_gap[:, 0], pred_gap[:, 1], color=title_color, linewidth=2.0)

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Stage 3 refinement results.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--span_ratio", type=float, default=DEFAULT_SPAN_RATIO)
    parser.add_argument("--span_mode", type=str, default=DEFAULT_SPAN_MODE, choices=["fixed", "random"])
    parser.add_argument("--sample_idx", type=int, default=DEFAULT_SAMPLE_INDEX)
    args = parser.parse_args()

    ensure_refinement_dirs()
    rows = parse_rows(metrics_csv_path())
    tag = experiment_tag(args.span_ratio, args.span_mode, args.seed)

    ade_path = REFINEMENT_FIGURE_DIR / "coarse_vs_refined_ADE.png"
    masked_ade_path = REFINEMENT_FIGURE_DIR / "coarse_vs_refined_masked_ADE.png"
    improvement_path = REFINEMENT_FIGURE_DIR / "improvement_bar_chart.png"
    full_vs_masked_path = REFINEMENT_FIGURE_DIR / "full_vs_masked_refinement_improvement.png"
    ddpm_vs_naive_path = REFINEMENT_FIGURE_DIR / "ddpm_vs_naive_refinement_improvement.png"
    example_path = REFINEMENT_FIGURE_DIR / "trajectory_example_coarse_refined.png"

    plot_metric_comparison(rows, "ADE", "ADE", ade_path)
    plot_metric_comparison(rows, "masked_ADE", "masked_ADE", masked_ade_path)
    plot_improvement(rows, improvement_path)
    plot_full_vs_masked_improvement(rows, full_vs_masked_path)
    plot_ddpm_vs_naive(rows, ddpm_vs_naive_path)
    plot_trajectory_example(args.sample_idx, tag, example_path)

    print("=" * 60)
    print("Refinement figures generated")
    print(f"ade_path         = {ade_path}")
    print(f"masked_ade_path  = {masked_ade_path}")
    print(f"improvement_path = {improvement_path}")
    print(f"full_vs_masked   = {full_vs_masked_path}")
    print(f"ddpm_vs_naive    = {ddpm_vs_naive_path}")
    print(f"example_path     = {example_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
