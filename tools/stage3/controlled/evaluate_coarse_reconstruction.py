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

from tools.stage3.eval.eval_geometry_metrics import is_off_map, segment_crosses_wall
from utils.stage3.controlled_benchmark import (
    CONTROLLED_FIGURE_DIR,
    DEFAULT_DRIFT_AMP,
    DEFAULT_NOISE_STD,
    DEFAULT_SAMPLE_INDEX,
    DEFAULT_SEED,
    DEFAULT_SPAN_MODE,
    DEFAULT_SPAN_RATIO,
    DEGRADATION_LABELS,
    DEGRADATION_NAMES,
    METHODS,
    METHOD_LABELS,
    ade_bar_path,
    clean_path,
    degraded_path,
    ensure_controlled_dirs,
    experiment_tag,
    mask_path,
    masked_ade_bar_path,
    metrics_csv_path,
    metrics_json_path,
    reconstruction_path,
    rmse_bar_path,
    trajectory_figure_path,
)
from utils.stage3.io import save_json


def load_array(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path, allow_pickle=False)


def room3_map_meta():
    return {
        "occupancy": np.zeros((128, 128), dtype=np.uint8),
        "x_min": 0.0,
        "x_max": 3.0,
        "y_min": 0.0,
        "y_max": 3.0,
    }


def compute_reconstruction_metrics(clean: np.ndarray, pred: np.ndarray, obs_mask: np.ndarray):
    if clean.shape != pred.shape:
        raise ValueError(f"Shape mismatch: clean {clean.shape} vs pred {pred.shape}")
    if obs_mask.shape != clean.shape[:2]:
        raise ValueError(f"obs_mask shape {obs_mask.shape} does not match clean {clean.shape[:2]}")

    diff = pred - clean
    point_error = np.linalg.norm(diff, axis=-1)
    missing_mask = obs_mask == 0
    if np.any(missing_mask):
        masked_point_error = point_error[missing_mask]
        masked_diff = diff[missing_mask]
        masked_ade = float(masked_point_error.mean())
        masked_rmse = float(np.sqrt(np.mean(masked_diff ** 2)))
    else:
        masked_ade = 0.0
        masked_rmse = 0.0

    return {
        "ADE": float(point_error.mean()),
        "FDE": float(point_error[:, -1].mean()),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "masked_ADE": masked_ade,
        "masked_RMSE": masked_rmse,
    }


def compute_geometry_metrics(pred: np.ndarray, map_meta: dict):
    off_map_count = 0
    wall_crossing_count = 0
    total_points = pred.shape[0] * pred.shape[1]
    total_segments = pred.shape[0] * max(pred.shape[1] - 1, 0)

    for traj in pred:
        for point in traj:
            if is_off_map(point, map_meta):
                off_map_count += 1
        for t in range(traj.shape[0] - 1):
            if segment_crosses_wall(traj[t], traj[t + 1], map_meta):
                wall_crossing_count += 1

    return {
        "off_map_ratio": float(off_map_count / total_points) if total_points > 0 else 0.0,
        "wall_crossing_count": int(wall_crossing_count),
    }


def write_metrics_csv(rows: list[dict], output_path: Path):
    fieldnames = [
        "degradation",
        "method",
        "ADE",
        "FDE",
        "RMSE",
        "masked_ADE",
        "masked_RMSE",
        "off_map_ratio",
        "wall_crossing_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar_metric(rows: list[dict], metric: str, ylabel: str, output_path: Path):
    x = np.arange(len(DEGRADATION_NAMES))
    width = 0.24
    colors = {
        "linear_interp": "#4C78A8",
        "savgol_w5_p2": "#72B7B2",
        "kalman_cv_dt1.0_q1e-3_r1e-2": "#595959",
    }

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for offset, method in zip([-width, 0.0, width], METHODS):
        values = [
            next(row[metric] for row in rows if row["degradation"] == degradation and row["method"] == method)
            for degradation in DEGRADATION_NAMES
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=colors[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DEGRADATION_LABELS[name] for name in DEGRADATION_NAMES], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Degradation")
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_segments(ax, traj: np.ndarray, mask: np.ndarray, color: str, linestyle: str = "-", linewidth: float = 2.0):
    start = None
    for idx, visible in enumerate(mask):
        if visible and start is None:
            start = idx
        is_last = idx == len(mask) - 1
        if start is not None and ((not visible) or is_last):
            end = idx if visible and is_last else idx - 1
            segment = traj[start : end + 1]
            if len(segment) >= 2:
                ax.plot(segment[:, 0], segment[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)
            elif len(segment) == 1:
                ax.scatter(segment[0, 0], segment[0, 1], color=color, s=16)
            start = None


def plot_trajectory_example(
    degradation_name: str,
    sample_idx: int,
    clean: np.ndarray,
    degraded: np.ndarray,
    obs_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
):
    mask = obs_mask[sample_idx].astype(bool)
    gt = clean[sample_idx]
    degraded_sample = degraded[sample_idx]
    observed = np.where(mask[:, None], degraded_sample, np.nan)
    missing_idx = np.flatnonzero(~mask)
    gap_slice = slice(max(missing_idx[0] - 1, 0), min(missing_idx[-1] + 2, gt.shape[0]))
    gt_gap = gt[gap_slice]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True, sharey=True)
    fig.suptitle(f"Trajectory example: {DEGRADATION_LABELS[degradation_name]} (sample {sample_idx})")
    titles = ["Ground truth", "Degraded input", "Linear", "SG", "Kalman"]

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(linewidth=0.6, alpha=0.25)

    axes[0].plot(gt[:, 0], gt[:, 1], color="#222222", linewidth=2.2)

    plot_segments(axes[1], observed, mask, color="#1f77b4", linewidth=2.2)
    axes[1].plot(gt_gap[:, 0], gt_gap[:, 1], color="#b0b0b0", linestyle="--", linewidth=1.5)

    method_to_axis = {
        "linear_interp": axes[2],
        "savgol_w5_p2": axes[3],
        "kalman_cv_dt1.0_q1e-3_r1e-2": axes[4],
    }
    for method, ax in method_to_axis.items():
        pred = predictions[method][sample_idx]
        pred_gap = pred[gap_slice]
        plot_segments(ax, observed, mask, color="#1f77b4", linewidth=2.2)
        ax.plot(gt_gap[:, 0], gt_gap[:, 1], color="#b0b0b0", linestyle="--", linewidth=1.3)
        ax.plot(pred_gap[:, 0], pred_gap[:, 1], color="#d62728", linewidth=2.0)

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = trajectory_figure_path(degradation_name, sample_idx)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_grouped_table(rows: list[dict]):
    print("=" * 60)
    print("Metrics summary grouped by degradation")
    for degradation in DEGRADATION_NAMES:
        print(f"[{degradation}]")
        for row in rows:
            if row["degradation"] != degradation:
                continue
            print(
                f"  {row['method']:30s} "
                f"ADE={row['ADE']:.6f} "
                f"FDE={row['FDE']:.6f} "
                f"RMSE={row['RMSE']:.6f} "
                f"masked_ADE={row['masked_ADE']:.6f} "
                f"masked_RMSE={row['masked_RMSE']:.6f} "
                f"off_map={row['off_map_ratio']:.6f} "
                f"wall={row['wall_crossing_count']}"
            )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate controlled Stage 3 coarse reconstruction and generate figures."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--span_ratio", type=float, default=DEFAULT_SPAN_RATIO)
    parser.add_argument("--span_mode", type=str, default=DEFAULT_SPAN_MODE, choices=["fixed", "random"])
    parser.add_argument("--sample_idx", type=int, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--noise_std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--drift_amp", type=float, default=DEFAULT_DRIFT_AMP)
    args = parser.parse_args()

    ensure_controlled_dirs()
    tag = experiment_tag(args.span_ratio, args.span_mode, args.seed)

    clean = load_array(clean_path()).astype(np.float32)
    obs_mask = load_array(mask_path(tag)).astype(np.uint8)
    if clean.ndim != 3 or clean.shape[-1] != 2:
        raise ValueError(f"Expected clean trajectory shape [N, T, 2], got {clean.shape}")
    if obs_mask.shape != clean.shape[:2]:
        raise ValueError(f"obs_mask shape {obs_mask.shape} does not match clean shape {clean.shape[:2]}")
    if args.sample_idx < 0 or args.sample_idx >= clean.shape[0]:
        raise IndexError(f"sample_idx {args.sample_idx} is out of range for {clean.shape[0]} samples")

    map_meta = room3_map_meta()
    rows = []
    grouped = {}
    trajectory_paths = []

    for degradation_name in DEGRADATION_NAMES:
        degraded = load_array(degraded_path(degradation_name, tag)).astype(np.float32)
        if degraded.shape != clean.shape:
            raise ValueError(
                f"Degraded shape mismatch for {degradation_name}: {degraded.shape} vs clean {clean.shape}"
            )

        predictions = {}
        grouped_rows = []
        for method in METHODS:
            pred_path = reconstruction_path(degradation_name, method, tag)
            pred = load_array(pred_path).astype(np.float32)
            predictions[method] = pred
            recon_metrics = compute_reconstruction_metrics(clean, pred, obs_mask)
            geom_metrics = compute_geometry_metrics(pred, map_meta)
            row = {
                "degradation": degradation_name,
                "method": method,
                **recon_metrics,
                **geom_metrics,
            }
            rows.append(row)
            grouped_rows.append(row)
        grouped[degradation_name] = grouped_rows
        trajectory_paths.append(
            str(plot_trajectory_example(degradation_name, args.sample_idx, clean, degraded, obs_mask, predictions))
        )

    write_metrics_csv(rows, metrics_csv_path())
    save_json(
        metrics_json_path(),
        {
            "config": {
                "seed": args.seed,
                "span_ratio": args.span_ratio,
                "span_mode": args.span_mode,
                "sample_idx": args.sample_idx,
                "noise_std": args.noise_std,
                "drift_amp": args.drift_amp,
                "experiment_tag": tag,
            },
            "rows": rows,
            "by_degradation": grouped,
        },
    )

    plot_bar_metric(rows, "ADE", "ADE", ade_bar_path())
    plot_bar_metric(rows, "RMSE", "RMSE", rmse_bar_path())
    plot_bar_metric(rows, "masked_ADE", "masked_ADE", masked_ade_bar_path())

    print_grouped_table(rows)
    print("Generated figure files:")
    for path in trajectory_paths:
        print(f"  {path}")
    print(f"  {ade_bar_path()}")
    print(f"  {rmse_bar_path()}")
    print(f"  {masked_ade_bar_path()}")
    print(f"metrics_csv = {metrics_csv_path()}")
    print(f"metrics_json = {metrics_json_path()}")


if __name__ == "__main__":
    main()
