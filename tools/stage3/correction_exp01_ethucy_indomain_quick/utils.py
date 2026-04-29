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
from utils.prior.ablation_paths import get_recommended_prior_paths
from utils.prior.run_metadata import resolve_current_run_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "exp01_ethucy_indomain_quick"
OUTPUT_DIR_NAME = "correction_exp01_ethucy_indomain_quick"
CODE_ROOT = PROJECT_ROOT / "tools" / "stage3" / "correction_exp01_ethucy_indomain_quick"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stage3" / OUTPUT_DIR_NAME

CONFIG_DIR = OUTPUT_ROOT / "config"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
RAW_DIR = OUTPUT_ROOT / "raw"
LOGS_DIR = OUTPUT_ROOT / "logs"

INPUT_ABS_PATH = PROJECT_ROOT / "datasets" / "processed" / "data_eth_ucy_20.npy"
INPUT_REL_PATH = PROJECT_ROOT / "datasets" / "processed" / "data_eth_ucy_20_rel.npy"
INPUT_META_PATH = PROJECT_ROOT / "datasets" / "processed" / "data_eth_ucy_20_meta.csv"
INPUT_SUMMARY_PATH = PROJECT_ROOT / "datasets" / "processed" / "data_eth_ucy_20_summary.csv"

CONFIG_JSON_PATH = CONFIG_DIR / "exp01_config.json"
PER_CASE_CSV_PATH = RAW_DIR / "per_case_results.csv"
SELECTED_CASES_JSON_PATH = RAW_DIR / "selected_cases.json"
FULL_STATS_CSV_PATH = TABLES_DIR / "full_stats_matrix.csv"
FULL_STATS_MD_PATH = TABLES_DIR / "full_stats_matrix.md"
SUMMARY_MD_PATH = TABLES_DIR / "summary_key_metrics.md"
MEDIAN_FIG_PATH = FIGURES_DIR / "median_case_five_column.png"
BEST_FIG_PATH = FIGURES_DIR / "best_ddpm_improvement_five_column.png"
WORST_FIG_PATH = FIGURES_DIR / "worst_ddpm_degradation_five_column.png"
RUN_LOG_PATH = LOGS_DIR / "run_log.txt"
OUTPUT_README_PATH = OUTPUT_ROOT / "README.md"
OUTPUT_CHANGELOG_PATH = OUTPUT_ROOT / "CHANGELOG.md"

MISSING_CONDITION = "missing_only"
COARSE_REFERENCE_METHOD = "linear_interp"
DEFAULT_SPAN_RATIO = 0.2
DEFAULT_SPAN_MODE = "fixed"
DEFAULT_SEED = 42
DEFAULT_NUM_DDPM_SEEDS = 5
DEFAULT_MAX_TRAJECTORIES = 256

METHOD_SPECS = [
    {
        "method": "linear_interp",
        "kind": "deterministic_baseline",
        "implemented": True,
    },
    {
        "method": "savgol_w5_p2",
        "kind": "deterministic_baseline",
        "implemented": True,
    },
    {
        "method": "kalman_cv_dt1.0_q1e-3_r1e-2",
        "kind": "deterministic_baseline",
        "implemented": True,
    },
    {
        "method": "ddpm_v3_inpainting",
        "kind": "stochastic_ddpm_candidate",
        "implemented": True,
    },
    {
        "method": "ddpm_v3_inpainting_anchored",
        "kind": "stochastic_ddpm_final",
        "implemented": True,
    },
]

METHOD_LABELS = {
    "linear_interp": "Linear",
    "savgol_w5_p2": "Savitzky-Golay",
    "kalman_cv_dt1.0_q1e-3_r1e-2": "Kalman",
    "ddpm_v3_inpainting": "DDPM v3 inpainting candidate",
    "ddpm_v3_inpainting_anchored": "DDPM v3 inpainting anchored final",
}
METHOD_ORDER = [item["method"] for item in METHOD_SPECS]

METRIC_ORDER = [
    "masked_ADE",
    "masked_RMSE",
    "endpoint_error",
    "path_length_error",
    "acceleration_error",
]


@dataclass
class ExperimentConfig:
    experiment_name: str = EXPERIMENT_NAME
    missing_condition: str = MISSING_CONDITION
    span_ratio: float = DEFAULT_SPAN_RATIO
    span_mode: str = DEFAULT_SPAN_MODE
    seed: int = DEFAULT_SEED
    num_ddpm_seeds: int = DEFAULT_NUM_DDPM_SEEDS
    max_trajectories: int = DEFAULT_MAX_TRAJECTORIES
    ddpm_objective: str = "optimization_best"
    ddpm_train_seed: int = 42
    ddpm_train_epochs: int = 100
    ddpm_timesteps: int = 100
    ddpm_hidden_dim: int = 128
    ddpm_device: str = "auto"
    kalman_dt: float = 1.0
    kalman_process_var: float = 1e-3
    kalman_measure_var: float = 1e-2
    savgol_window_length: int = 5
    savgol_polyorder: int = 2


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


def ensure_output_dirs() -> dict[str, Path]:
    dirs = {
        "output_root": OUTPUT_ROOT,
        "config": CONFIG_DIR,
        "tables": TABLES_DIR,
        "figures": FIGURES_DIR,
        "raw": RAW_DIR,
        "logs": LOGS_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_absolute_trajectories(input_path: Path, max_trajectories: int | None = None) -> np.ndarray:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    traj_abs = np.load(input_path, allow_pickle=False).astype(np.float32)
    if traj_abs.ndim != 3 or traj_abs.shape[-1] != 2:
        raise ValueError(f"Expected [N, T, 2] absolute trajectories, got {traj_abs.shape}")
    if max_trajectories is not None:
        traj_abs = traj_abs[:max_trajectories]
    return traj_abs


def resolve_span_length(seq_len: int, span_ratio: float) -> int:
    if not 0.0 < span_ratio < 1.0:
        raise ValueError(f"span_ratio must be in (0, 1), got {span_ratio}")
    span_len = int(round(seq_len * span_ratio))
    span_len = max(1, span_len)
    span_len = min(span_len, seq_len - 2)
    if span_len <= 0:
        raise ValueError(f"Sequence length {seq_len} is too short for span_ratio {span_ratio}")
    return span_len


def sample_spans(num_samples: int, seq_len: int, span_len: int, span_mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    span_start = np.zeros(num_samples, dtype=np.int64)
    span_end = np.zeros(num_samples, dtype=np.int64)
    max_start = seq_len - span_len - 1
    if max_start < 1:
        raise ValueError(f"Cannot place a bounded interior span length {span_len} in T={seq_len}")

    if span_mode == "fixed":
        start = (seq_len - span_len) // 2
        end = start + span_len - 1
        span_start.fill(start)
        span_end.fill(end)
        return span_start, span_end

    if span_mode != "random":
        raise ValueError(f"Unsupported span_mode: {span_mode}")

    for idx in range(num_samples):
        start = int(rng.integers(low=1, high=max_start + 1))
        end = start + span_len - 1
        span_start[idx] = start
        span_end[idx] = end
    return span_start, span_end


def build_missing_only_dataset(traj_abs: np.ndarray, span_ratio: float, span_mode: str, seed: int) -> dict[str, np.ndarray]:
    num_samples, seq_len, _ = traj_abs.shape
    span_len = resolve_span_length(seq_len, span_ratio)
    span_start, span_end = sample_spans(num_samples, seq_len, span_len, span_mode, seed)

    obs_mask = np.ones((num_samples, seq_len), dtype=np.uint8)
    traj_obs = traj_abs.copy()
    for idx in range(num_samples):
        start = int(span_start[idx])
        end = int(span_end[idx])
        obs_mask[idx, start : end + 1] = 0
        traj_obs[idx, start : end + 1] = np.nan

    return {
        "traj_abs": traj_abs.astype(np.float32),
        "traj_obs": traj_obs.astype(np.float32),
        "obs_mask": obs_mask,
        "span_start": span_start,
        "span_end": span_end,
    }


def maybe_import_savgol_filter():
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return None
    return savgol_filter


def run_linear_interp(traj_obs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(traj_obs, dtype=np.float32)
    for idx in range(traj_obs.shape[0]):
        out[idx] = interpolate_sample(traj_obs[idx], obs_mask[idx], index=idx)
    return out


def run_savgol(traj_obs: np.ndarray, obs_mask: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    savgol_filter = maybe_import_savgol_filter()
    if savgol_filter is None:
        raise RuntimeError("scipy.signal.savgol_filter is unavailable in the current environment.")
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


def run_ddpm_v3(traj_obs: np.ndarray, obs_mask: np.ndarray, config: ExperimentConfig) -> np.ndarray:
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
        seed_base=config.seed,
        config=ddpm_config,
    ).astype(np.float32)


def anchor_missing_spans(pred_ns: np.ndarray, observed_abs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    pred_ns = pred_ns.astype(np.float32, copy=True)
    observed_abs = observed_abs.astype(np.float32)
    n, s, t, _ = pred_ns.shape
    for traj_idx in range(n):
        missing = np.where(obs_mask[traj_idx] == 0)[0]
        observed_mask = obs_mask[traj_idx] == 1
        if missing.size == 0:
            pred_ns[traj_idx] = observed_abs[traj_idx][None, :, :]
            continue
        left = int(missing[0] - 1)
        right = int(missing[-1] + 1)
        if left < 0 or right >= t:
            raise ValueError("Expected a bounded contiguous missing span.")
        left_obs = observed_abs[traj_idx, left]
        right_obs = observed_abs[traj_idx, right]
        for seed_idx in range(s):
            traj = pred_ns[traj_idx, seed_idx]
            segment = traj[left : right + 1].copy()
            seg_len = right - left
            left_delta = left_obs - segment[0]
            right_delta = right_obs - segment[-1]
            for local_idx in range(seg_len + 1):
                lam = local_idx / seg_len if seg_len > 0 else 0.0
                segment[local_idx] = segment[local_idx] + (1.0 - lam) * left_delta + lam * right_delta
            segment[0] = left_obs
            segment[-1] = right_obs
            traj[left : right + 1] = segment
            traj[observed_mask] = observed_abs[traj_idx, observed_mask]
            pred_ns[traj_idx, seed_idx] = traj
    return pred_ns


def masked_point_error(clean: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    point_error = np.linalg.norm(pred - clean, axis=-1)
    return point_error[mask == 0]


def compute_endpoint_error(clean: np.ndarray, pred: np.ndarray, span_end: int) -> float:
    return float(np.linalg.norm(pred[span_end] - clean[span_end]))


def path_length(traj: np.ndarray) -> float:
    if traj.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=-1).sum())


def compute_path_length_error(clean: np.ndarray, pred: np.ndarray, span_start: int, span_end: int) -> float:
    left = max(span_start - 1, 0)
    right = min(span_end + 1, clean.shape[0] - 1)
    clean_seg = clean[left : right + 1]
    pred_seg = pred[left : right + 1]
    return float(abs(path_length(pred_seg) - path_length(clean_seg)))


def acceleration_sequence(traj: np.ndarray) -> np.ndarray:
    if traj.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    vel = np.diff(traj, axis=0)
    return np.diff(vel, axis=0).astype(np.float32)


def compute_acceleration_error(clean: np.ndarray, pred: np.ndarray) -> float:
    acc_clean = acceleration_sequence(clean)
    acc_pred = acceleration_sequence(pred)
    if acc_clean.size == 0:
        return 0.0
    diff = acc_pred - acc_clean
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_case_metrics(clean: np.ndarray, pred: np.ndarray, obs_mask: np.ndarray, span_start: int, span_end: int) -> dict[str, float]:
    missing_values = masked_point_error(clean, pred, obs_mask)
    diff = pred - clean
    missing_diff = diff[obs_mask == 0]
    if missing_values.size == 0:
        masked_ade = 0.0
        masked_rmse = 0.0
    else:
        masked_ade = float(missing_values.mean())
        masked_rmse = float(np.sqrt(np.mean(missing_diff ** 2)))
    return {
        "masked_ADE": masked_ade,
        "masked_RMSE": masked_rmse,
        "endpoint_error": compute_endpoint_error(clean, pred, span_end),
        "path_length_error": compute_path_length_error(clean, pred, span_start, span_end),
        "acceleration_error": compute_acceleration_error(clean, pred),
    }


def summarize_metric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        nan = float("nan")
        return {
            "N": 0,
            "mean": nan,
            "std": nan,
            "median": nan,
            "min": nan,
            "max": nan,
            "p05": nan,
            "p25": nan,
            "p75": nan,
            "p95": nan,
        }
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


def format_float(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_float(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def method_population_note(method: str, num_trajectories: int, num_ddpm_seeds: int) -> str:
    if method.startswith("ddpm_v3"):
        return (
            f"N counts trajectory-seed cases for `{method}`: "
            f"{num_trajectories} trajectories × {num_ddpm_seeds} DDPM seeds = {num_trajectories * num_ddpm_seeds}."
        )
    return f"N counts trajectories for `{method}`: {num_trajectories}."


def get_prior_metadata(config: ExperimentConfig) -> dict[str, Any]:
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
        "recommended_paths": prior_paths,
        "run_metadata": run_meta,
    }


def plot_segments(ax, traj: np.ndarray, obs_mask: np.ndarray | None, color: str, linewidth: float = 2.0, alpha: float = 1.0):
    if obs_mask is None:
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=linewidth, alpha=alpha)
        return
    started = False
    for idx in range(traj.shape[0] - 1):
        if obs_mask[idx] == 1 and obs_mask[idx + 1] == 1:
            ax.plot(
                traj[idx : idx + 2, 0],
                traj[idx : idx + 2, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
            started = True
    if not started:
        observed_idx = np.flatnonzero(obs_mask == 1)
        if observed_idx.size > 0:
            obs = traj[observed_idx]
            ax.scatter(obs[:, 0], obs[:, 1], color=color, s=10, alpha=alpha)


def highlight_missing(ax, traj: np.ndarray, span_start: int, span_end: int, color: str):
    ax.plot(
        traj[span_start : span_end + 1, 0],
        traj[span_start : span_end + 1, 1],
        "o-",
        color=color,
        linewidth=2.0,
        markersize=4,
        alpha=0.9,
    )


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
    notes: list[str] | None = None,
):
    all_trajs = [clean, coarse, ddpm_candidate, final_refined]
    finite_points = [arr[np.isfinite(arr).all(axis=1)] for arr in all_trajs]
    xy = np.concatenate(finite_points, axis=0)
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    pad_x = max((x_max - x_min) * 0.08, 0.1)
    pad_y = max((y_max - y_min) * 0.08, 0.1)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), squeeze=True)
    column_titles = [
        "clean target",
        "degraded input",
        f"coarse reconstruction\n(reference: {COARSE_REFERENCE_METHOD})",
        "DDPM candidate",
        "final refined output",
    ]
    colors = {
        "clean": "#1f77b4",
        "degraded": "#7f7f7f",
        "coarse": "#d62728",
        "candidate": "#ff7f0e",
        "final": "#2ca02c",
        "gap": "#111111",
    }

    plot_segments(axes[0], clean, None, colors["clean"], linewidth=2.2)
    highlight_missing(axes[0], clean, span_start, span_end, colors["clean"])

    plot_segments(axes[1], degraded, obs_mask, colors["degraded"], linewidth=2.2)
    highlight_missing(axes[1], clean, span_start, span_end, colors["gap"])

    plot_segments(axes[2], coarse, None, colors["coarse"], linewidth=2.2)
    highlight_missing(axes[2], coarse, span_start, span_end, colors["coarse"])

    plot_segments(axes[3], ddpm_candidate, None, colors["candidate"], linewidth=2.2)
    highlight_missing(axes[3], ddpm_candidate, span_start, span_end, colors["candidate"])

    plot_segments(axes[4], final_refined, None, colors["final"], linewidth=2.2)
    highlight_missing(axes[4], final_refined, span_start, span_end, colors["final"])

    for ax, col_title in zip(axes, column_titles):
        ax.set_title(col_title, fontsize=9)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if notes:
        title = title + " | " + " ; ".join(notes)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_output_readme(
    config: ExperimentConfig,
    dataset_payload: dict[str, np.ndarray],
    prior_meta: dict[str, Any],
    included_methods: list[str],
    skipped_methods: list[dict[str, str]],
    stats_rows: list[dict[str, Any]],
    selected_cases_payload: dict[str, Any],
    warnings: list[str],
) -> str:
    num_trajectories = int(dataset_payload["traj_abs"].shape[0])
    seq_len = int(dataset_payload["traj_abs"].shape[1])
    span_len = int(dataset_payload["span_end"][0] - dataset_payload["span_start"][0] + 1)
    metric_defs = [
        {
            "metric": "masked_ADE",
            "definition": "Mean Euclidean point error on masked frames only.",
        },
        {
            "metric": "masked_RMSE",
            "definition": "Root mean square coordinate error on masked frames only.",
        },
        {
            "metric": "endpoint_error",
            "definition": "Euclidean error at the last missing frame of the contiguous span.",
        },
        {
            "metric": "path_length_error",
            "definition": "Absolute path-length difference on the bounded subtrajectory covering the missing span and its two observed anchors.",
        },
        {
            "metric": "acceleration_error",
            "definition": "RMSE between full-trajectory second finite differences, used as a smoothness proxy.",
        },
    ]
    metric_rows = [{"metric": item["metric"], "definition": item["definition"]} for item in metric_defs]

    key_rows = [
        row
        for row in stats_rows
        if row["metric"] in {"masked_ADE", "masked_RMSE", "endpoint_error"}
    ]

    lines = [
        f"# {config.experiment_name}",
        "",
        "## 1. Experiment name",
        "",
        f"`{config.experiment_name}`",
        "",
        "## 2. Purpose",
        "",
        "Quick in-domain sanity check of Stage 3 missing reconstruction on ETH+UCY public trajectories.",
        "",
        "## 3. What this experiment tests",
        "",
        "Whether the existing DDPM v3 reconstruction interface works when evaluation data come from the same public trajectory domain as the prior.",
        "",
        "## 4. What this experiment does NOT test",
        "",
        "- real Room3 performance",
        "- real sensor deployment",
        "- strict held-out generalization, unless a held-out split is explicitly implemented",
        "- geometry-aware reconstruction",
        "",
        "## 5. Dataset paths used",
        "",
        f"- absolute trajectories: `{INPUT_ABS_PATH}`",
        f"- relative trajectories reference: `{INPUT_REL_PATH}`",
    ]
    if INPUT_META_PATH.exists():
        lines.append(f"- metadata csv: `{INPUT_META_PATH}`")
    if INPUT_SUMMARY_PATH.exists():
        lines.append(f"- summary csv: `{INPUT_SUMMARY_PATH}`")
    lines += [
        "",
        "Dataset notes:",
        f"- natural coordinate scale, no Room3 normalization",
        f"- first `{num_trajectories}` trajectories used for this quick run",
        f"- sequence length `T={seq_len}`",
        f"- contiguous missing span length `{span_len}` frames under `{config.span_mode}` placement",
        "",
        "## 6. Prior checkpoint path used",
        "",
        f"- objective: `{prior_meta['objective']}`",
        f"- recommended prior variant: `{prior_meta['variant']}`",
        f"- checkpoint path: `{prior_meta['run_metadata']['ckpt_path']}`",
        "",
        "## 7. Method list",
        "",
    ]
    lines.extend(f"- `{method}`" for method in included_methods)
    if skipped_methods:
        lines += ["", "Skipped methods:"]
        lines.extend(f"- `{item['method']}`: {item['reason']}" for item in skipped_methods)
    lines += [
        "",
        "## 8. Missing condition",
        "",
        f"- `{config.missing_condition}` only",
        "- no added observation noise",
        "- no added drift",
        "",
        "## 9. Metric definitions",
        "",
        markdown_table(metric_rows, ["metric", "definition"]),
        "",
        "## 10. Statistical population",
        "",
        f"- deterministic methods: `N = {num_trajectories}` trajectories",
        f"- DDPM methods: `N = {num_trajectories} × {config.num_ddpm_seeds} = {num_trajectories * config.num_ddpm_seeds}` trajectory-seed cases",
        "- one contiguous missing span is generated per trajectory in this run",
        "- `per_case_results.csv` stores deterministic rows at trajectory level and DDPM rows at trajectory-seed level",
        "",
        "## 11. Known limitations",
        "",
        "- Existing checkpoint may have been trained on the same public trajectory corpus, so this is a pipeline sanity check, not a strict generalization test.",
        "- The run is intentionally quick and therefore uses a subset of the available ETH+UCY windows rather than a full-corpus exhaustive evaluation.",
        "- `ddpm_v3_inpainting` currently generates directly from degraded observed input; the saved five-column figure uses `linear_interp` as a reference coarse reconstruction column rather than a true upstream dependency of v3.",
        "",
        "## Key Stats Snapshot",
        "",
        markdown_table(key_rows, ["method", "missing_condition", "metric", "N", "mean", "std", "median", "p25", "p75"]),
        "",
        "## Representative Figures",
        "",
        f"- `{MEDIAN_FIG_PATH}`",
        f"- `{BEST_FIG_PATH}`",
        f"- `{WORST_FIG_PATH}`",
        "",
        "## Selected Case Notes",
        "",
        f"- coarse reference method for figure columns: `{selected_cases_payload['coarse_reference_method']}`",
        "- missing exact intermediate note: the current v3 interface does not expose a separate coarse-dependent DDPM candidate generation stage; see `selected_cases.json` for the explicit note stored with each case.",
    ]
    if warnings:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def build_output_changelog(
    run_date: str,
    included_methods: list[str],
    skipped_methods: list[dict[str, str]],
    generated_paths: list[Path],
) -> str:
    lines = [
        "# CHANGELOG",
        "",
        f"## {run_date}",
        "",
        "Files created:",
        f"- `{CODE_ROOT / 'run_exp01.py'}`",
        f"- `{CODE_ROOT / 'utils.py'}`",
        f"- `{CODE_ROOT / 'README.md'}`",
        f"- `{CODE_ROOT / 'CHANGELOG.md'}`",
    ]
    lines.extend(f"- `{path}`" for path in generated_paths)
    lines += [
        "",
        "Purpose of the experiment:",
        "- quick in-domain sanity check for Stage 3 missing reconstruction on ETH+UCY public trajectories",
        "",
        "Methods included:",
    ]
    lines.extend(f"- `{method}`" for method in included_methods)
    lines += [
        "",
        "Outputs generated:",
        f"- `{CONFIG_JSON_PATH}`",
        f"- `{PER_CASE_CSV_PATH}`",
        f"- `{SELECTED_CASES_JSON_PATH}`",
        f"- `{FULL_STATS_CSV_PATH}`",
        f"- `{FULL_STATS_MD_PATH}`",
        f"- `{SUMMARY_MD_PATH}`",
        f"- `{MEDIAN_FIG_PATH}`",
        f"- `{BEST_FIG_PATH}`",
        f"- `{WORST_FIG_PATH}`",
        f"- `{RUN_LOG_PATH}`",
        f"- `{OUTPUT_README_PATH}`",
        f"- `{OUTPUT_CHANGELOG_PATH}`",
    ]
    if skipped_methods:
        lines += ["", "Missing methods or missing intermediate outputs:"]
        lines.extend(f"- `{item['method']}`: {item['reason']}" for item in skipped_methods)
    else:
        lines += ["", "Missing methods or missing intermediate outputs:", "- no method implementation was missing at runtime"]
    lines += [
        "- five-column figures use `linear_interp` as the reference coarse column because the current v3 interface does not expose a coarse-dependent internal stage",
    ]
    return "\n".join(lines) + "\n"


def build_summary_markdown(
    config: ExperimentConfig,
    included_methods: list[str],
    skipped_methods: list[dict[str, str]],
    stats_rows: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    key_metrics = [row for row in stats_rows if row["metric"] in {"masked_ADE", "masked_RMSE", "endpoint_error"}]
    lines = [
        f"# {config.experiment_name} Summary",
        "",
        f"- missing condition: `{config.missing_condition}`",
        f"- methods evaluated: {', '.join(f'`{m}`' for m in included_methods)}",
        f"- DDPM seeds per trajectory: `{config.num_ddpm_seeds}`",
        f"- max trajectories used in this quick run: `{config.max_trajectories}`",
        "",
        markdown_table(key_metrics, ["method", "metric", "N", "mean", "std", "median", "p05", "p95"]),
    ]
    if skipped_methods:
        lines += ["", "## Skipped", ""]
        lines.extend(f"- `{item['method']}`: {item['reason']}" for item in skipped_methods)
    if warnings:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"
