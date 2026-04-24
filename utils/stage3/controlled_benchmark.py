from __future__ import annotations

from pathlib import Path

from utils.stage3.paths import OUTPUT_DIR


DEFAULT_SEED = 42
DEFAULT_SPAN_RATIO = 0.2
DEFAULT_SPAN_MODE = "fixed"
DEFAULT_SAMPLE_INDEX = 0
DEFAULT_NOISE_STD = 0.03
DEFAULT_DRIFT_AMP = 0.05

DEGRADATION_NAMES = [
    "missing_only",
    "missing_noise",
    "missing_drift",
    "missing_noise_drift",
]
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
DEGRADATION_LABELS = {
    "missing_only": "Missing only",
    "missing_noise": "Missing + noise",
    "missing_drift": "Missing + drift",
    "missing_noise_drift": "Missing + noise + drift",
}

DEGRADATION_DIR = OUTPUT_DIR / "degradation"
RECONSTRUCTION_DIR = OUTPUT_DIR / "reconstruction"
CONTROLLED_EVAL_DIR = OUTPUT_DIR / "eval"
CONTROLLED_FIGURE_DIR = OUTPUT_DIR / "figures"


def ensure_controlled_dirs():
    dirs = {
        "degradation": DEGRADATION_DIR,
        "reconstruction": RECONSTRUCTION_DIR,
        "eval": CONTROLLED_EVAL_DIR,
        "figures": CONTROLLED_FIGURE_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def experiment_tag(span_ratio: float, span_mode: str, seed: int):
    span_percent = int(round(span_ratio * 100))
    return f"span{span_percent}_{span_mode}_seed{seed}"


def clean_path():
    return DEGRADATION_DIR / "clean.npy"


def mask_path(tag: str):
    return DEGRADATION_DIR / f"mask_{tag}.npy"


def degraded_path(degradation_name: str, tag: str):
    return DEGRADATION_DIR / f"degraded_{degradation_name}_{tag}.npy"


def degradation_metadata_path(tag: str):
    return DEGRADATION_DIR / f"metadata_{tag}.json"


def reconstruction_path(degradation_name: str, method: str, tag: str):
    return RECONSTRUCTION_DIR / f"recon_{degradation_name}_{method}_{tag}.npy"


def metrics_csv_path():
    return CONTROLLED_EVAL_DIR / "metrics_summary.csv"


def metrics_json_path():
    return CONTROLLED_EVAL_DIR / "metrics_summary.json"


def trajectory_figure_path(degradation_name: str, sample_idx: int):
    return CONTROLLED_FIGURE_DIR / f"trajectory_example_{degradation_name}_sample{sample_idx}.png"


def ade_bar_path():
    return CONTROLLED_FIGURE_DIR / "bar_ADE_by_degradation.png"


def rmse_bar_path():
    return CONTROLLED_FIGURE_DIR / "bar_RMSE_by_degradation.png"


def masked_ade_bar_path():
    return CONTROLLED_FIGURE_DIR / "bar_masked_ADE_by_degradation.png"
