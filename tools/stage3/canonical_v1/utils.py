from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.stage3.baselines.run_kalman import kalman_reconstruct, validate_sample as validate_kalman_sample
from tools.stage3.baselines.run_linear_interp import interpolate_sample
from tools.stage3.baselines.run_savgol import validate_and_interp
from tools.stage3.refinement.ddpm_refiner import DDPMPriorInterfaceConfig, ddpm_prior_inpainting_v3
from tools.stage3.refinement.ddpm_refiner import load_prior_checkpoint
from tools.stage3.controlled.evaluate_coarse_reconstruction import room3_map_meta
from utils.prior.ablation_paths import get_recommended_prior_paths
from utils.prior.run_metadata import resolve_current_run_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "stage3_canonical_v1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "canonical_v1"
CODE_ROOT = PROJECT_ROOT / "tools" / "stage3" / "canonical_v1"

README_PATH = OUTPUT_ROOT / "README.md"
CHANGELOG_PATH = OUTPUT_ROOT / "CHANGELOG.md"
CONFIG_PATH = OUTPUT_ROOT / "config" / "stage3_canonical_v1_config.json"

RAW_SEED_LEVEL_PATH = OUTPUT_ROOT / "raw" / "per_case_results_seed_level.csv"
RAW_TRAJ_LEVEL_PATH = OUTPUT_ROOT / "raw" / "per_case_results_trajectory_level.csv"
SELECTED_CASES_PATH = OUTPUT_ROOT / "raw" / "selected_ddpm_cases.json"

FULL_MATRIX_SEED_CSV = OUTPUT_ROOT / "tables" / "full_matrix_seed_level.csv"
FULL_MATRIX_SEED_MD = OUTPUT_ROOT / "tables" / "full_matrix_seed_level.md"
FULL_MATRIX_TRAJ_CSV = OUTPUT_ROOT / "tables" / "full_matrix_trajectory_level.csv"
FULL_MATRIX_TRAJ_MD = OUTPUT_ROOT / "tables" / "full_matrix_trajectory_level.md"
TABLE2_REPLACEMENT_MD = OUTPUT_ROOT / "tables" / "table2_complete_replacement.md"
MISSING_CELL_AUDIT_CSV = OUTPUT_ROOT / "tables" / "missing_cell_audit.csv"

FIGURE3_PATH = OUTPUT_ROOT / "figures" / "figure3_replacement_raw_spread.png"
FIGURE5_PATH = OUTPUT_ROOT / "figures" / "figure5_replacement_alpha_variance.png"
FIG_MEDIAN_PATH = OUTPUT_ROOT / "figures" / "ddpm_case_median_five_column.png"
FIG_BEST_PATH = OUTPUT_ROOT / "figures" / "ddpm_case_best_improvement_five_column.png"
FIG_WORST_PATH = OUTPUT_ROOT / "figures" / "ddpm_case_worst_degradation_five_column.png"

FIGURE3_AUDIT_PATH = OUTPUT_ROOT / "audit" / "figure3_spread_definition.md"
FIGURE5_AUDIT_PATH = OUTPUT_ROOT / "audit" / "figure5_spread_definition.md"
CASE_TRACEABILITY_PATH = OUTPUT_ROOT / "audit" / "ddpm_case_traceability.md"
GEOMETRY_STATEMENT_PATH = OUTPUT_ROOT / "audit" / "geometry_usage_statement.md"
FDE_AUDIT_PATH = OUTPUT_ROOT / "audit" / "fde_zero_audit.md"

CLEAN_ROOM3_PATH = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "data" / "clean_windows_room3.npz"
PHASE1_EVAL_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "eval"
RANDOM_SPAN_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "random_span_statistics"
ALPHA_SWEEP_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "alpha_sweep"
INPAINTING_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "inpainting_experiment"

CONDITIONS = [
    "span10_fixed_seed42",
    "span20_fixed_seed42",
    "span30_fixed_seed42",
    "span20_random_seed42",
    "span20_random_seed43",
    "span20_random_seed44",
]
METHODS = [
    "linear_interp",
    "savgol_w5_p2",
    "kalman_cv_dt1.0_q1e-3_r1e-2",
    "ddpm_v3_inpainting",
    "ddpm_v3_inpainting_anchored",
]
METRICS = [
    "ADE",
    "FDE",
    "RMSE",
    "masked_ADE",
    "masked_RMSE",
    "endpoint_error",
    "path_length_error",
    "acceleration_error",
    "off_map_ratio",
    "wall_crossing_count",
]
DDPM_AGGREGATIONS = ["seed_mean", "seed_median", "seed_best", "seed_worst"]
METHOD_LABELS = {
    "linear_interp": "Linear",
    "savgol_w5_p2": "Savitzky-Golay",
    "kalman_cv_dt1.0_q1e-3_r1e-2": "Kalman",
    "ddpm_v3_inpainting": "DDPM v3 inpainting",
    "ddpm_v3_inpainting_anchored": "DDPM v3 anchored",
}
STATUS_VALUES = ["ok", "not_implemented", "not_run", "missing_raw_data", "metric_unavailable"]


@dataclass
class CanonicalConfig:
    max_trajectories: int = 1024
    num_ddpm_seeds: int = 5
    ddpm_device: str = "cpu"
    ddpm_objective: str = "optimization_best"
    ddpm_train_seed: int = 42
    ddpm_train_epochs: int = 100
    ddpm_timesteps: int = 100
    ddpm_hidden_dim: int = 128
    savgol_window_length: int = 5
    savgol_polyorder: int = 2
    kalman_dt: float = 1.0
    kalman_process_var: float = 1e-3
    kalman_measure_var: float = 1e-2
    alpha_grid: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, message: str = ""):
        print(message)
        self._fh.write(message + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def ensure_output_dirs():
    for path in [
        OUTPUT_ROOT,
        OUTPUT_ROOT / "config",
        OUTPUT_ROOT / "raw",
        OUTPUT_ROOT / "tables",
        OUTPUT_ROOT / "figures",
        OUTPUT_ROOT / "audit",
        OUTPUT_ROOT / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_clean_room3(max_trajectories: int) -> np.ndarray:
    data = np.load(CLEAN_ROOM3_PATH, allow_pickle=False)
    traj_abs = np.asarray(data["traj_abs"], dtype=np.float32)
    return traj_abs[:max_trajectories]


def parse_condition(condition: str) -> tuple[float, str, int]:
    span_part, mode_part, seed_part = condition.split("_")
    span_ratio = int(span_part.replace("span", "")) / 100.0
    span_mode = mode_part
    seed = int(seed_part.replace("seed", ""))
    return span_ratio, span_mode, seed


def resolve_span_length(seq_len: int, span_ratio: float) -> int:
    span_len = int(round(seq_len * span_ratio))
    span_len = max(1, span_len)
    span_len = min(span_len, seq_len - 2)
    return span_len


def build_obs_mask(num_samples: int, seq_len: int, span_ratio: float, span_mode: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    span_len = resolve_span_length(seq_len, span_ratio)
    span_start = np.zeros(num_samples, dtype=np.int64)
    span_end = np.zeros(num_samples, dtype=np.int64)
    max_start = seq_len - span_len - 1
    if span_mode == "fixed":
        start = (seq_len - span_len) // 2
        span_start.fill(start)
        span_end.fill(start + span_len - 1)
    else:
        rng = np.random.default_rng(seed)
        for idx in range(num_samples):
            start = int(rng.integers(low=1, high=max_start + 1))
            span_start[idx] = start
            span_end[idx] = start + span_len - 1
    obs_mask = np.ones((num_samples, seq_len), dtype=np.uint8)
    for idx in range(num_samples):
        obs_mask[idx, span_start[idx] : span_end[idx] + 1] = 0
    return obs_mask, span_start, span_end


def build_degraded(clean: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    degraded = clean.copy()
    degraded[obs_mask == 0] = np.nan
    return degraded


def run_linear_interp(traj_obs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(traj_obs, dtype=np.float32)
    for idx in range(traj_obs.shape[0]):
        out[idx] = interpolate_sample(traj_obs[idx], obs_mask[idx], index=idx)
    return out


def run_savgol(traj_obs: np.ndarray, obs_mask: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    from scipy.signal import savgol_filter

    out = np.zeros_like(traj_obs, dtype=np.float32)
    for idx in range(traj_obs.shape[0]):
        filled = validate_and_interp(traj_obs[idx], obs_mask[idx], index=idx)
        for dim in range(filled.shape[1]):
            out[idx, :, dim] = savgol_filter(
                filled[:, dim],
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            ).astype(np.float32)
    return out


def run_kalman(traj_obs: np.ndarray, obs_mask: np.ndarray, dt: float, process_var: float, measure_var: float) -> np.ndarray:
    out = np.zeros_like(traj_obs, dtype=np.float32)
    for idx in range(traj_obs.shape[0]):
        validate_kalman_sample(obs_mask[idx], index=idx)
        out[idx] = kalman_reconstruct(
            traj_obs=traj_obs[idx],
            mask=obs_mask[idx],
            dt=dt,
            process_var=process_var,
            measure_var=measure_var,
        )
    return out


def run_ddpm(traj_obs: np.ndarray, obs_mask: np.ndarray, condition_seed: int, config: CanonicalConfig) -> np.ndarray:
    ddpm_config = DDPMPriorInterfaceConfig(
        objective=config.ddpm_objective,
        train_seed=config.ddpm_train_seed,
        train_epochs=config.ddpm_train_epochs,
        timesteps=config.ddpm_timesteps,
        hidden_dim=config.ddpm_hidden_dim,
        device=config.ddpm_device,
    )
    return ddpm_prior_inpainting_v3(
        traj_obs,
        obs_mask,
        num_samples_per_traj=config.num_ddpm_seeds,
        seed_base=condition_seed,
        config=ddpm_config,
    ).astype(np.float32)


def anchor_missing_spans(pred_ns: np.ndarray, observed_abs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    pred_ns = pred_ns.astype(np.float32, copy=True)
    n, s, t, _ = pred_ns.shape
    for traj_idx in range(n):
        missing_idx = np.where(obs_mask[traj_idx] == 0)[0]
        if missing_idx.size == 0:
            pred_ns[traj_idx] = observed_abs[traj_idx][None, :, :]
            continue
        left = int(missing_idx[0] - 1)
        right = int(missing_idx[-1] + 1)
        left_obs = observed_abs[traj_idx, left]
        right_obs = observed_abs[traj_idx, right]
        observed_bool = obs_mask[traj_idx] == 1
        for seed_idx in range(s):
            segment = pred_ns[traj_idx, seed_idx, left : right + 1].copy()
            seg_len = right - left
            left_delta = left_obs - segment[0]
            right_delta = right_obs - segment[-1]
            for local_idx in range(seg_len + 1):
                lam = local_idx / seg_len if seg_len > 0 else 0.0
                segment[local_idx] = segment[local_idx] + (1.0 - lam) * left_delta + lam * right_delta
            segment[0] = left_obs
            segment[-1] = right_obs
            pred_ns[traj_idx, seed_idx, left : right + 1] = segment
            pred_ns[traj_idx, seed_idx, observed_bool] = observed_abs[traj_idx, observed_bool]
    return pred_ns


def compute_off_map_ratio(pred: np.ndarray) -> float:
    meta = room3_map_meta()
    x = pred[:, 0]
    y = pred[:, 1]
    off = (x < meta["x_min"]) | (x > meta["x_max"]) | (y < meta["y_min"]) | (y > meta["y_max"])
    return float(np.mean(off))


def compute_metrics(clean: np.ndarray, pred: np.ndarray, obs_mask: np.ndarray, span_start: int, span_end: int) -> dict[str, float]:
    diff = pred - clean
    point_error = np.linalg.norm(diff, axis=-1)
    missing = obs_mask == 0
    masked_error = point_error[missing]
    masked_diff = diff[missing]
    ade = float(np.mean(point_error))
    fde = float(point_error[-1])
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    masked_ade = float(np.mean(masked_error)) if masked_error.size else 0.0
    masked_rmse = float(np.sqrt(np.mean(masked_diff ** 2))) if masked_diff.size else 0.0
    endpoint_error = float(np.linalg.norm(pred[span_end] - clean[span_end]))

    left = max(span_start - 1, 0)
    right = min(span_end + 1, clean.shape[0] - 1)
    clean_seg = clean[left : right + 1]
    pred_seg = pred[left : right + 1]
    clean_len = np.linalg.norm(np.diff(clean_seg, axis=0), axis=-1).sum()
    pred_len = np.linalg.norm(np.diff(pred_seg, axis=0), axis=-1).sum()
    path_length_error = float(abs(pred_len - clean_len))

    acc_clean = np.diff(np.diff(clean, axis=0), axis=0)
    acc_pred = np.diff(np.diff(pred, axis=0), axis=0)
    acceleration_error = float(np.sqrt(np.mean((acc_pred - acc_clean) ** 2))) if acc_clean.size else 0.0
    return {
        "ADE": ade,
        "FDE": fde,
        "RMSE": rmse,
        "masked_ADE": masked_ade,
        "masked_RMSE": masked_rmse,
        "endpoint_error": endpoint_error,
        "path_length_error": path_length_error,
        "acceleration_error": acceleration_error,
        "off_map_ratio": compute_off_map_ratio(pred),
        "wall_crossing_count": 0.0,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        nan = float("nan")
        return {"N": 0, "mean": nan, "std": nan, "median": nan, "min": nan, "max": nan, "p05": nan, "p25": nan, "p75": nan, "p95": nan}
    return {
        "N": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def get_prior_metadata(config: CanonicalConfig) -> dict[str, Any]:
    prior_paths = get_recommended_prior_paths(config.ddpm_objective)
    run_meta = resolve_current_run_metadata(
        variant=prior_paths["variant"],
        train_seed=config.ddpm_train_seed,
        train_epochs=config.ddpm_train_epochs,
        train_root=PROJECT_ROOT / "outputs" / "prior" / "train",
    )
    return {
        "objective": config.ddpm_objective,
        "variant": prior_paths["variant"],
        "checkpoint_path": run_meta["ckpt_path"],
        "run_metadata": run_meta,
    }


def load_alpha_sweep_source() -> tuple[Path, list[dict[str, Any]]]:
    path = ALPHA_SWEEP_ROOT / "alpha_sweep_metrics.csv"
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
    return path, rows


def plot_segments(ax, traj: np.ndarray, obs_mask: np.ndarray | None, color: str, linewidth: float = 2.0, alpha: float = 1.0):
    if obs_mask is None:
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        return
    drawn = False
    for idx in range(traj.shape[0] - 1):
        if obs_mask[idx] == 1 and obs_mask[idx + 1] == 1:
            ax.plot(traj[idx : idx + 2, 0], traj[idx : idx + 2, 1], color=color, linewidth=linewidth, alpha=alpha)
            drawn = True
    if not drawn:
        obs = np.flatnonzero(obs_mask == 1)
        if obs.size:
            ax.scatter(traj[obs, 0], traj[obs, 1], color=color, s=10, alpha=alpha)


def highlight_gap(ax, traj: np.ndarray, span_start: int, span_end: int, color: str):
    ax.plot(traj[span_start : span_end + 1, 0], traj[span_start : span_end + 1, 1], "o-", color=color, linewidth=2.0, markersize=4)


def make_five_column_figure(
    figure_path: Path,
    title: str,
    clean: np.ndarray,
    degraded: np.ndarray,
    coarse: np.ndarray,
    ddpm_candidate: np.ndarray,
    final_refined: np.ndarray,
    obs_mask: np.ndarray,
    span_start: int,
    span_end: int,
):
    finite = np.concatenate([clean, coarse, ddpm_candidate, final_refined], axis=0)
    x_min, y_min = finite.min(axis=0)
    x_max, y_max = finite.max(axis=0)
    pad_x = max(0.08 * (x_max - x_min), 0.1)
    pad_y = max(0.08 * (y_max - y_min), 0.1)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    cols = [
        ("clean target", clean, None, "#1f77b4"),
        ("degraded input", degraded, obs_mask, "#7f7f7f"),
        ("coarse reconstruction", coarse, None, "#d62728"),
        ("DDPM candidate", ddpm_candidate, None, "#ff7f0e"),
        ("final refined output", final_refined, None, "#2ca02c"),
    ]
    for ax, (label, traj, mask, color) in zip(axes, cols):
        plot_segments(ax, traj, mask, color)
        if label == "degraded input":
            highlight_gap(ax, clean, span_start, span_end, "#111111")
        else:
            highlight_gap(ax, traj if label != "clean target" else clean, span_start, span_end, color)
        ax.set_title(label, fontsize=9)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
