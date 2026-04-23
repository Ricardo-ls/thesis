from __future__ import annotations

import argparse
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


DEFAULT_EXPERIMENT_ID = "span20_fixed_seed42"
DEFAULT_SAMPLE_INDEX = 0
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage3"
    / "phase1"
    / "canonical_room3"
    / "figures"
)
DATA_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3"
X_LIMITS = (0.0, 3.0)
Y_LIMITS = (0.0, 3.0)


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def require_key(data: dict, path: Path, key: str):
    if key not in data:
        raise KeyError(f"'{key}' not found in {path}. Available keys: {sorted(data.keys())}")
    return np.asarray(data[key])


def resolve_prediction_key(data: dict, path: Path):
    if "traj_hat" in data:
        return "traj_hat"

    candidates = []
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-1] == 2:
            candidates.append(key)

    if not candidates:
        raise KeyError(
            f"No prediction key found in {path}. Expected 'traj_hat' or another [N, T, 2] array. "
            f"Available keys: {sorted(data.keys())}"
        )
    if len(candidates) > 1:
        raise KeyError(
            f"Multiple candidate prediction keys found in {path}: {candidates}. "
            "Please make the prediction key explicit."
        )
    return candidates[0]


def validate_sample_index(sample_index: int, num_samples: int):
    if sample_index < 0 or sample_index >= num_samples:
        raise IndexError(
            f"sample_index {sample_index} is out of range for dataset with {num_samples} samples"
        )


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
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
            elif len(segment) == 1:
                ax.scatter(segment[0, 0], segment[0, 1], color=color, s=18)
            start = None


def configure_axis(ax, title: str):
    ax.set_title(title)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(linewidth=0.6, alpha=0.25)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a representative Stage 3 Phase 1 canonical_room3 reconstruction example."
    )
    parser.add_argument("--experiment_id", type=str, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--sample_index", type=int, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    clean_path = DATA_ROOT / "data" / "clean_windows_room3.npz"
    degraded_path = DATA_ROOT / "data" / "experiments" / args.experiment_id / "missing_span_windows.npz"
    linear_path = DATA_ROOT / "baselines" / args.experiment_id / "linear_interp" / "results.npz"
    sg_path = DATA_ROOT / "baselines" / args.experiment_id / "savgol_w5_p2" / "results.npz"
    kalman_path = (
        DATA_ROOT
        / "baselines"
        / args.experiment_id
        / "kalman_cv_dt1.0_q1e-3_r1e-2"
        / "results.npz"
    )

    clean_data = load_npz(clean_path)
    degraded_data = load_npz(degraded_path)
    linear_data = load_npz(linear_path)
    sg_data = load_npz(sg_path)
    kalman_data = load_npz(kalman_path)

    clean = require_key(clean_data, clean_path, "traj_abs").astype(np.float32)
    traj_abs = require_key(degraded_data, degraded_path, "traj_abs").astype(np.float32)
    traj_obs = require_key(degraded_data, degraded_path, "traj_obs").astype(np.float32)
    obs_mask = require_key(degraded_data, degraded_path, "obs_mask").astype(np.uint8)
    span_start = require_key(degraded_data, degraded_path, "span_start").astype(np.int64)
    span_end = require_key(degraded_data, degraded_path, "span_end").astype(np.int64)

    linear_key = resolve_prediction_key(linear_data, linear_path)
    sg_key = resolve_prediction_key(sg_data, sg_path)
    kalman_key = resolve_prediction_key(kalman_data, kalman_path)

    linear = np.asarray(linear_data[linear_key], dtype=np.float32)
    sg = np.asarray(sg_data[sg_key], dtype=np.float32)
    kalman = np.asarray(kalman_data[kalman_key], dtype=np.float32)

    num_samples = traj_abs.shape[0]
    validate_sample_index(args.sample_index, num_samples)

    idx = args.sample_index
    gt_traj = clean[idx]
    degraded_gt = traj_abs[idx]
    observed = traj_obs[idx]
    mask = obs_mask[idx].astype(bool)
    gap_start = int(span_start[idx])
    gap_end = int(span_end[idx])

    gap_line_slice = slice(max(gap_start - 1, 0), min(gap_end + 2, gt_traj.shape[0]))
    gap_gt = degraded_gt[gap_line_slice]
    gap_pred_linear = linear[idx][gap_line_slice]
    gap_pred_sg = sg[idx][gap_line_slice]
    gap_pred_kalman = kalman[idx][gap_line_slice]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharex=True, sharey=True)
    fig.suptitle(f"Representative reconstruction example under {args.experiment_id}")

    configure_axis(axes[0], "Ground truth")
    axes[0].plot(gt_traj[:, 0], gt_traj[:, 1], color="#222222", linewidth=2.2)

    configure_axis(axes[1], "Degraded input")
    plot_segments(axes[1], observed, mask, color="#1f77b4", linewidth=2.2)
    axes[1].plot(
        gap_gt[:, 0],
        gap_gt[:, 1],
        color="#b0b0b0",
        linestyle="--",
        linewidth=1.6,
    )

    for ax, title, pred_gap in [
        (axes[2], "Linear", gap_pred_linear),
        (axes[3], "SG", gap_pred_sg),
        (axes[4], "Kalman", gap_pred_kalman),
    ]:
        configure_axis(ax, title)
        plot_segments(ax, observed, mask, color="#1f77b4", linewidth=2.2)
        ax.plot(
            gap_gt[:, 0],
            gap_gt[:, 1],
            color="#b0b0b0",
            linestyle="--",
            linewidth=1.3,
        )
        ax.plot(pred_gap[:, 0], pred_gap[:, 1], color="#d62728", linewidth=2.0)

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "exp0_representative_reconstruction.png"
    pdf_path = output_dir / "exp0_representative_reconstruction.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("=" * 60)
    print("Stage 3 representative reconstruction figure generated")
    print(f"experiment_id = {args.experiment_id}")
    print(f"sample_index  = {args.sample_index}")
    print(f"clean_path    = {clean_path}")
    print(f"degraded_path = {degraded_path}")
    print(f"linear_key    = {linear_key}")
    print(f"sg_key        = {sg_key}")
    print(f"kalman_key    = {kalman_key}")
    print(f"png_path      = {png_path}")
    print(f"pdf_path      = {pdf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
