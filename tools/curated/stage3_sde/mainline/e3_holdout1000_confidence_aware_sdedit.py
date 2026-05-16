from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3_holdout1000_sdedit_mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D
from tools.stage3_indoor import degrade as stage3_degrade
from tools.stage3_indoor.generate_indoor_trajs import (
    BEHAVIORS,
    CLIP_MAX,
    CLIP_MIN,
    FPS,
    ROOM_MAX,
    ROOM_MIN,
    T,
    add_micro_jitter,
    generate_boundary_walk,
    generate_goal_directed,
    generate_multi_goal,
    generate_pacing,
    generate_stationary,
)
from tools.stage3_indoor.sdedit_gaussian_full import run_sdedit

try:
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False


CONDITIONS = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
T_STARTS = [1, 2, 3]
SDEDIT_SEEDS = [42, 43, 44, 45, 46]
SEED_START = 13000
SEED_END = 13999
N_HOLDOUT = 1000
TAU_HIGH = 0.7
GAMMA = 2
GLOBAL_T_START = 1
PREVIOUS_BEST_T_START = {
    "gaussian_medium": 2,
    "drift_medium": 1,
    "burst_medium": 1,
    "bias_medium": 1,
    "jump_medium": 3,
    "combined_medium": 1,
}

DATA_DIR = PROJECT_ROOT / "data" / "stage4" / "e3_holdout_1000"
DATA_CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
DATA_META_PATH = DATA_DIR / "metadata.json"
DEGRADATION_META_PATH = DATA_DIR / "degradation_metadata.json"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_holdout1000_confidence_aware_sdedit"
ARRAY_DIR = OUT_DIR / "arrays"
FIG_DIR = OUT_DIR / "figures"

NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    for path in [DATA_DIR, OUT_DIR, ARRAY_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(path)} written")


def save_csv(path: Path, rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[FILE] {rel(path)} written")
    return df


def generate_one_from_seed(seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    behavior_names = [name for name, _ in BEHAVIORS]
    behavior_probs = [weight for _, weight in BEHAVIORS]
    behavior = str(rng.choice(behavior_names, p=behavior_probs))
    if behavior == "goal_directed":
        traj = generate_goal_directed(rng)
    elif behavior == "multi_goal":
        traj = generate_multi_goal(rng)
    elif behavior == "pacing":
        traj = generate_pacing(rng)
    elif behavior == "stationary":
        traj = generate_stationary(rng)
    elif behavior == "boundary_walk":
        traj = generate_boundary_walk(rng)
    else:
        raise ValueError(f"Unsupported behavior: {behavior}")
    traj = add_micro_jitter(traj, rng)
    traj = np.clip(traj, CLIP_MIN, CLIP_MAX).astype(np.float32)
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    meta = {
        "seed": int(seed),
        "behavior": behavior,
        "mean_speed": float(steps.mean() * FPS),
    }
    return traj, meta


def create_holdout_clean() -> tuple[np.ndarray, list[dict]]:
    clean = np.zeros((N_HOLDOUT, T, 2), dtype=np.float32)
    meta_rows: list[dict] = []
    for idx, seed in enumerate(range(SEED_START, SEED_END + 1)):
        traj, meta = generate_one_from_seed(seed)
        clean[idx] = traj
        meta["trajectory_id"] = idx
        meta_rows.append(meta)
    if clean.shape != (1000, 20, 2):
        raise ValueError(f"Unexpected generated clean shape: {clean.shape}")
    np.save(DATA_CLEAN_PATH, clean)
    behavior_counts = {name: 0 for name, _ in BEHAVIORS}
    for row in meta_rows:
        behavior_counts[row["behavior"]] += 1
    payload = {
        "seed_start": SEED_START,
        "seed_end": SEED_END,
        "n_trajectories": N_HOLDOUT,
        "trajectory_length": T,
        "shape": list(clean.shape),
        "dtype": str(clean.dtype),
        "room_size": [ROOM_MIN, ROOM_MAX],
        "clip_range": [CLIP_MIN, CLIP_MAX],
        "fps": FPS,
        "behavior_families": [{"name": name, "weight": weight} for name, weight in BEHAVIORS],
        "behavior_counts": behavior_counts,
        "generation_script_wrapper_path": rel(Path(__file__).resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overlap_check": {
            "train_trajs_seed_range": [1000, 10999],
            "val_trajs_seed_range": [11000, 12999],
            "this_seed_range_overlaps_train_or_val": False,
        },
    }
    write_json(DATA_META_PATH, payload)
    print(f"[FILE] {rel(DATA_CLEAN_PATH)} written")
    return clean, meta_rows


def load_or_create_clean() -> np.ndarray:
    clean, _ = create_holdout_clean()
    return clean


def generate_degradations(clean: np.ndarray) -> dict[str, np.ndarray]:
    degraded_map = {
        "gaussian_medium": stage3_degrade.apply_gaussian(clean, sigma=0.05),
        "drift_medium": stage3_degrade.apply_drift(clean, sigma_step=0.010),
        "burst_medium": stage3_degrade.apply_burst(clean, burst_sigma=0.25, background_sigma=0.01),
        "bias_medium": stage3_degrade.apply_bias(clean, sigma=0.15),
        "jump_medium": stage3_degrade.apply_jump(clean),
        "combined_medium": stage3_degrade.apply_combined(clean, sigma_g=0.05, sigma_b=0.15, sigma_d=0.010),
    }
    degradation_meta: dict[str, dict] = {}
    for condition, degraded in degraded_map.items():
        if degraded.shape != clean.shape:
            raise ValueError(f"{condition}: degraded shape {degraded.shape} != clean {clean.shape}")
        path = DATA_DIR / f"degraded_{condition}.npy"
        np.save(path, degraded.astype(np.float32))
        print(f"[FILE] {rel(path)} written")
        degradation_meta[condition] = {
            "output_path": rel(path),
            "shape": list(degraded.shape),
            "function_name": {
                "gaussian_medium": "apply_gaussian",
                "drift_medium": "apply_drift",
                "burst_medium": "apply_burst",
                "bias_medium": "apply_bias",
                "jump_medium": "apply_jump",
                "combined_medium": "apply_combined",
            }[condition],
        }
    degradation_meta.update(
        {
            "input_clean_path": rel(DATA_CLEAN_PATH),
            "no_old_files_overwritten": True,
            "seed_rule_default": f"{stage3_degrade.BASE_SEED} + i",
            "combined_seed_rules": {
                "gaussian": "42 + i",
                "bias": "42000 + i",
                "drift": "84000 + i",
            },
            "parameters": {
                "gaussian_medium": {"sigma": 0.05},
                "drift_medium": {"sigma_step": 0.010},
                "burst_medium": {"burst_sigma": 0.25, "background_sigma": 0.01},
                "bias_medium": {"sigma": 0.15},
                "jump_medium": {
                    "n_jump_range_inclusive": [2, 4],
                    "jump_frame_range_inclusive": [1, T - 2],
                    "magnitude_range_m": [0.2, 0.5],
                    "offset_mode": "piecewise_constant_non_cumulative",
                },
                "combined_medium": {"sigma_g": 0.05, "sigma_b": 0.15, "sigma_d": 0.010},
            },
        }
    )
    write_json(DEGRADATION_META_PATH, degradation_meta)
    return degraded_map


def load_model(device: torch.device) -> tuple[TemporalDenoiser1D, DDPMForwardProcess, np.ndarray, np.ndarray]:
    for path in [CHECKPOINT_PATH, NORM_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    norm = np.load(NORM_PATH)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    diffusion = DDPMForwardProcess(timesteps=100, device=device)
    model = TemporalDenoiser1D(max_timesteps=100, in_channels=2, hidden_dim=128).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, diffusion, rel_mean, rel_std


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def per_traj_ade(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return frame_error(pred, clean).mean(axis=1)


def per_traj_rmse(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    err = frame_error(pred, clean)
    return np.sqrt(np.mean(err**2, axis=1)).astype(np.float32)


def acceleration_rms(pred: np.ndarray) -> np.ndarray:
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    acc_norm = np.linalg.norm(acc, axis=-1)
    return np.sqrt(np.mean(acc_norm**2, axis=1)).astype(np.float32)


def automatic_delta0(degraded: np.ndarray, clean: np.ndarray) -> float:
    delta0 = float(np.median(frame_error(degraded, clean).reshape(-1)))
    if not np.isfinite(delta0) or delta0 <= 0.0:
        raise ValueError(f"Invalid automatic delta0: {delta0}")
    return delta0


def compute_confidence(degraded: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, float]:
    delta0 = automatic_delta0(degraded, clean)
    confidence = np.exp(-frame_error(degraded, clean) / delta0).astype(np.float32)
    return confidence, delta0


def masks(confidence: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "high": confidence > 0.7,
        "mid": (confidence >= 0.3) & (confidence <= 0.7),
        "low": confidence < 0.3,
    }


def masked_ade(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    err = frame_error(pred, clean)
    return float(np.mean(err[mask])) if int(mask.sum()) else math.nan


def hard_fuse(degraded: np.ndarray, sdedit: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    correction = sdedit - degraded
    lam = np.power(1.0 - confidence, GAMMA).astype(np.float32)
    lam = np.where(confidence > TAU_HIGH, 0.0, lam).astype(np.float32)
    return (degraded + lam[..., None] * correction).astype(np.float32)


def run_or_load_sdedit(
    condition: str,
    t_start: int,
    degraded: np.ndarray,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    out_path = ARRAY_DIR / f"{condition}_sdedit_t{t_start}.npy"
    if out_path.is_file():
        cached = np.load(out_path).astype(np.float32)
        if cached.shape == degraded.shape:
            print(f"[CACHE] {rel(out_path)}")
            return cached
    seed_preds = []
    for sdedit_seed in SDEDIT_SEEDS:
        pred = run_sdedit(
            degraded_abs=degraded,
            model=model,
            diffusion=diffusion,
            rel_mean=rel_mean,
            rel_std=rel_std,
            t_start=t_start,
            sdedit_seed=sdedit_seed,
            device=device,
        )
        seed_preds.append(pred.astype(np.float32))
    out = np.mean(np.stack(seed_preds, axis=0), axis=0).astype(np.float32)
    np.save(out_path, out)
    print(f"[FILE] {rel(out_path)} written")
    return out


def wilcoxon_p(delta: np.ndarray) -> float:
    if not SCIPY_AVAILABLE:
        return math.nan
    try:
        return float(wilcoxon(delta, alternative="less").pvalue)
    except Exception:
        return math.nan


def method_row(
    condition: str,
    method: str,
    protocol: str,
    pred: np.ndarray,
    clean: np.ndarray,
    degraded: np.ndarray,
    confidence: np.ndarray,
    noisy_ade_traj: np.ndarray,
    t_start: int | float,
    vanilla_ref: np.ndarray | None = None,
    correction_ref: np.ndarray | None = None,
    delta0: float | None = None,
) -> dict:
    ade_traj = per_traj_ade(pred, clean)
    rmse_traj = per_traj_rmse(pred, clean)
    acc_traj = acceleration_rms(pred)
    bin_masks = masks(confidence)
    if correction_ref is not None:
        denom = float(np.mean(np.linalg.norm(correction_ref - degraded, axis=-1)))
    else:
        denom = float(np.mean(np.linalg.norm(pred - degraded, axis=-1)))
    usage_num = float(np.mean(np.linalg.norm(pred - degraded, axis=-1)))
    motion_usage = usage_num / denom if denom > 1e-12 else 0.0
    row = {
        "condition": condition,
        "method": method,
        "protocol": protocol,
        "t_start": t_start,
        "tau_high": TAU_HIGH if "fused" in method else math.nan,
        "gamma": GAMMA if "fused" in method else math.nan,
        "N": int(pred.shape[0]),
        "delta0_auto": delta0,
        "ADE": float(np.mean(ade_traj)),
        "RMSE": float(np.mean(rmse_traj)),
        "acceleration_RMS": float(np.mean(acc_traj)),
        "high_conf_ADE": masked_ade(pred, clean, bin_masks["high"]),
        "mid_conf_ADE": masked_ade(pred, clean, bin_masks["mid"]),
        "low_conf_ADE": masked_ade(pred, clean, bin_masks["low"]),
        "noisy_reversion_gap": float(np.mean(ade_traj)) - float(np.mean(noisy_ade_traj)),
        "motion_usage_ratio": motion_usage,
        "win_rate_vs_noisy_input": float(np.mean(ade_traj < noisy_ade_traj)),
        "paired_mean_delta_ADE_vs_noisy": float(np.mean(ade_traj - noisy_ade_traj)),
        "paired_median_delta_ADE_vs_noisy": float(np.median(ade_traj - noisy_ade_traj)),
        "paired_wilcoxon_p_vs_noisy": wilcoxon_p(ade_traj - noisy_ade_traj),
        "percent_improved_vs_noisy": float(np.mean(ade_traj < noisy_ade_traj) * 100.0),
    }
    if vanilla_ref is not None:
        vanilla_ade = per_traj_ade(vanilla_ref, clean)
        row["win_rate_vs_vanilla_SDEdit"] = float(np.mean(ade_traj < vanilla_ade))
        row["paired_mean_delta_ADE_vs_vanilla"] = float(np.mean(ade_traj - vanilla_ade))
    else:
        row["win_rate_vs_vanilla_SDEdit"] = math.nan
        row["paired_mean_delta_ADE_vs_vanilla"] = math.nan
    return row


def confidence_bin_rows(
    condition: str,
    method: str,
    protocol: str,
    pred: np.ndarray,
    clean: np.ndarray,
    confidence: np.ndarray,
    t_start: int | float,
) -> list[dict]:
    err = frame_error(pred, clean)
    rows: list[dict] = []
    for bin_name, mask in masks(confidence).items():
        vals = err[mask]
        rows.append(
            {
                "condition": condition,
                "method": method,
                "protocol": protocol,
                "t_start": t_start,
                "bin": bin_name,
                "N_frames": int(mask.sum()),
                "ADE_mean": float(np.mean(vals)) if vals.size else math.nan,
                "ADE_std": float(np.std(vals)) if vals.size else math.nan,
                "ADE_median": float(np.median(vals)) if vals.size else math.nan,
                "ADE_min": float(np.min(vals)) if vals.size else math.nan,
                "ADE_max": float(np.max(vals)) if vals.size else math.nan,
                "ADE_p25": float(np.percentile(vals, 25)) if vals.size else math.nan,
                "ADE_p75": float(np.percentile(vals, 75)) if vals.size else math.nan,
            }
        )
    return rows


def curve_type(sub: pd.DataFrame) -> str:
    rows = sub.sort_values("t_start")
    ades = rows["ADE"].to_numpy(dtype=float)
    noisy = float(rows["noisy_ADE"].iloc[0])
    if np.all(ades > noisy) and np.all(np.diff(ades) >= -1e-12):
        return "monotonic_worse"
    if np.all(np.diff(ades) <= 1e-12):
        return "monotonic_better"
    best_idx = int(np.argmin(ades))
    best_impr = (noisy - float(np.min(ades))) / max(noisy, 1e-12) * 100.0
    if best_idx == 0 and best_impr > 0:
        return "tiny-t_only_improvement"
    if 0 < best_idx < len(ades) - 1 and best_impr > 0:
        return "small_U-shaped" if best_impr < 5.0 else "U-shaped"
    if np.max(np.abs(ades - noisy)) / max(noisy, 1e-12) < 0.02:
        return "flat"
    return "mixed"


def plot_overall(metrics: pd.DataFrame, protocol: str, filename: str) -> None:
    sub = metrics[(metrics["protocol"] == protocol) & (metrics["method"].isin(["noisy_input", "vanilla_SDEdit", "fused_tau07_gamma2"]))]
    x = np.arange(len(CONDITIONS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for offset, method in zip([-1, 0, 1], ["noisy_input", "vanilla_SDEdit", "fused_tau07_gamma2"]):
        vals = sub[sub["method"] == method].set_index("condition").loc[CONDITIONS]["ADE"]
        ax.bar(x + offset * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=25, ha="right")
    ax.set_ylabel("ADE")
    ax.set_title(f"Overall ADE comparison ({protocol})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[FILE] {rel(path)} written")


def plot_tstart_sweep(vanilla_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for ax, condition in zip(axes.flatten(), CONDITIONS):
        sub = vanilla_df[vanilla_df["condition"] == condition].sort_values("t_start")
        ax.plot(sub["t_start"], sub["ADE"], marker="o", label="vanilla SDEdit")
        ax.axhline(float(sub["noisy_ADE"].iloc[0]), color="gray", linestyle="--", label="noisy")
        ax.set_title(f"{condition}\n{sub['curve_type'].iloc[0]}")
        ax.set_xlabel("t_start")
        ax.set_ylabel("ADE")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    path = FIG_DIR / "tstart_sweep_by_condition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[FILE] {rel(path)} written")


def plot_high_low(metrics: pd.DataFrame, protocol: str, filename: str) -> None:
    sub = metrics[(metrics["protocol"] == protocol) & (metrics["method"].isin(["noisy_input", "vanilla_SDEdit", "fused_tau07_gamma2"]))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    for ax, col, title in [
        (axes[0], "high_conf_ADE", "High-confidence ADE"),
        (axes[1], "low_conf_ADE", "Low-confidence ADE"),
    ]:
        x = np.arange(len(CONDITIONS))
        width = 0.25
        for offset, method in zip([-1, 0, 1], ["noisy_input", "vanilla_SDEdit", "fused_tau07_gamma2"]):
            vals = sub[sub["method"] == method].set_index("condition").loc[CONDITIONS][col]
            ax.bar(x + offset * width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=25, ha="right")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[FILE] {rel(path)} written")


def plot_representative(
    condition: str,
    clean: np.ndarray,
    degraded: np.ndarray,
    confidence: np.ndarray,
    sdedit: np.ndarray,
    fused: np.ndarray,
    t_start: int,
) -> None:
    noisy_ade = per_traj_ade(degraded, clean)
    fused_ade = per_traj_ade(fused, clean)
    idx = int(np.argmax(np.abs(fused_ade - noisy_ade)))
    frames = np.arange(clean.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(clean[idx, :, 0], clean[idx, :, 1], "k-", label="clean")
    ax.plot(degraded[idx, :, 0], degraded[idx, :, 1], color="tab:orange", label="degraded")
    ax.plot(sdedit[idx, :, 0], sdedit[idx, :, 1], color="tab:blue", label=f"SDEdit t={t_start}")
    ax.plot(fused[idx, :, 0], fused[idx, :, 1], color="tab:red", label="fused tau0.7 gamma2")
    ax.set_title(f"{condition} representative idx={idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(frames, np.linalg.norm(degraded[idx] - clean[idx], axis=-1), color="tab:orange", label="noisy")
    ax.plot(frames, np.linalg.norm(sdedit[idx] - clean[idx], axis=-1), color="tab:blue", label="SDEdit")
    ax.plot(frames, np.linalg.norm(fused[idx] - clean[idx], axis=-1), color="tab:red", label="fused")
    ax.set_title("Per-frame error")
    ax.set_xlabel("frame")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(frames, confidence[idx], color="tab:purple")
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.3, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Oracle confidence")
    ax.set_xlabel("frame")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = FIG_DIR / f"representative_{condition}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"[FILE] {rel(path)} written")


def build_trend_summary(global_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol_name, df in [("global_t1", global_df), ("per_condition_best", best_df)]:
        merged = df.pivot(index="condition", columns="method", values=["ADE", "high_conf_ADE", "low_conf_ADE", "motion_usage_ratio"])
        n_better = 0
        n_high_no_harm = 0
        low_eligible = 0
        low_preserved = 0
        for condition in CONDITIONS:
            noisy_ade = float(merged.loc[condition, ("ADE", "noisy_input")])
            fused_ade = float(merged.loc[condition, ("ADE", "fused_tau07_gamma2")])
            noisy_high = float(merged.loc[condition, ("high_conf_ADE", "noisy_input")])
            fused_high = float(merged.loc[condition, ("high_conf_ADE", "fused_tau07_gamma2")])
            noisy_low = float(merged.loc[condition, ("low_conf_ADE", "noisy_input")])
            vanilla_low = float(merged.loc[condition, ("low_conf_ADE", "vanilla_SDEdit")])
            fused_low = float(merged.loc[condition, ("low_conf_ADE", "fused_tau07_gamma2")])
            n_better += int(fused_ade < noisy_ade)
            n_high_no_harm += int(fused_high / noisy_high <= 1.05) if noisy_high > 1e-12 else 0
            if vanilla_low < noisy_low:
                low_eligible += 1
                low_preserved += int(fused_low < noisy_low)
        motion = df[df["method"] == "fused_tau07_gamma2"]["motion_usage_ratio"].mean()
        rows.append(
            {
                "protocol": protocol_name,
                "fused_better_than_noisy_conditions": n_better,
                "fused_better_than_noisy_ge4of6": bool(n_better >= 4),
                "high_conf_no_harm_conditions": n_high_no_harm,
                "high_conf_no_harm_recovered": bool(n_high_no_harm >= 5),
                "low_conf_preserved_conditions": low_preserved,
                "low_conf_eligible_conditions": low_eligible,
                "low_conf_preservation_pass": bool(low_eligible > 0 and low_preserved >= max(1, low_eligible)),
                "mean_motion_usage_ratio": float(motion),
                "not_noisy_reversion": bool(motion > 0.1),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    vanilla_df: pd.DataFrame,
    global_df: pd.DataFrame,
    best_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    trend_df: pd.DataFrame,
) -> None:
    best_t_rows = []
    for condition in CONDITIONS:
        sub = vanilla_df[vanilla_df["condition"] == condition].sort_values("t_start")
        known_t = PREVIOUS_BEST_T_START[condition]
        row = sub[sub["t_start"] == known_t].iloc[0]
        best_t_rows.append(
            {
                "condition": condition,
                "previous_best_t_start": known_t,
                "curve_type_on_holdout": row["curve_type"],
                "noisy_ADE": row["noisy_ADE"],
                "selected_sdedit_ADE": row["ADE"],
                "selected_delta_ADE": row["ADE"] - row["noisy_ADE"],
            }
        )
    best_t_table = pd.DataFrame(best_t_rows)

    global_trend = trend_df[trend_df["protocol"] == "global_t1"].iloc[0]
    best_trend = trend_df[trend_df["protocol"] == "per_condition_best"].iloc[0]
    global_fused = global_df[global_df["method"] == "fused_tau07_gamma2"].set_index("condition").loc[CONDITIONS]
    best_fused = best_df[best_df["method"] == "fused_tau07_gamma2"].set_index("condition").loc[CONDITIONS]

    drift_stable = bool(
        global_fused.loc["drift_medium", "ADE"] < global_df[(global_df["condition"] == "drift_medium") & (global_df["method"] == "noisy_input")]["ADE"].iloc[0]
    )
    burst_stable = bool(
        global_fused.loc["burst_medium", "ADE"] < global_df[(global_df["condition"] == "burst_medium") & (global_df["method"] == "noisy_input")]["ADE"].iloc[0]
    )

    lines = [
        "# E3 Holdout-1000 Confidence-Aware SDEdit Expanded Validation",
        "",
        "This validation uses a new independent clean hold-out set with seeds 13000-13999. It does not use old `clean_trajs[200:]`, old degraded arrays, retraining, checkpoint modification, or a changed SDEdit reverse process.",
        "",
        "## Data",
        "",
        f"- clean hold-out: `{rel(DATA_CLEAN_PATH)}`",
        f"- clean metadata: `{rel(DATA_META_PATH)}`",
        f"- degradation metadata: `{rel(DEGRADATION_META_PATH)}`",
        f"- output arrays: `{rel(ARRAY_DIR)}`",
        f"- shape: `(1000, 20, 2)`",
        f"- seed range: `{SEED_START}-{SEED_END}`",
        "",
        "## Method",
        "",
        f"- checkpoint: `{rel(CHECKPOINT_PATH)}`",
        f"- normalization: `{rel(NORM_PATH)}`",
        f"- SDEdit seeds averaged: `{SDEDIT_SEEDS}`",
        f"- t_start sweep: `{T_STARTS}`",
        f"- global fixed t_start: `{GLOBAL_T_START}`",
        f"- fusion: `tau_high={TAU_HIGH}`, `gamma={GAMMA}`",
        "- confidence: `c_t = exp(- ||y_t - x*_t|| / delta_0)`, with per-condition `delta_0` equal to median degraded-frame error on this hold-out set.",
        "",
        "## Fixed Previous Best t_start Diagnostic Upper Bound",
        "",
        best_t_table.to_string(index=False),
        "",
        "## Trend Stability Checks",
        "",
        trend_df.to_string(index=False),
        "",
        "## Answers",
        "",
        f"1. New hold-out clean trajectories generated: `True`, shape `(1000, 20, 2)`, seeds `{SEED_START}-{SEED_END}`.",
        "2. Six degradations generated: `True`.",
        "3. SDEdit t_start {1,2,3} completed for all six conditions: `True`.",
        f"4. Global t=1 fused >=4/6 overall improvement: `{bool(global_trend['fused_better_than_noisy_ge4of6'])}` ({int(global_trend['fused_better_than_noisy_conditions'])}/6).",
        f"5. Global t=1 high-conf no-harm recovered: `{bool(global_trend['high_conf_no_harm_recovered'])}` ({int(global_trend['high_conf_no_harm_conditions'])}/6).",
        f"6. Global t=1 low-conf preservation: `{bool(global_trend['low_conf_preservation_pass'])}` ({int(global_trend['low_conf_preserved_conditions'])}/{int(global_trend['low_conf_eligible_conditions'])} eligible).",
        f"7. Per-condition best fused >=4/6 overall improvement: `{bool(best_trend['fused_better_than_noisy_ge4of6'])}` ({int(best_trend['fused_better_than_noisy_conditions'])}/6).",
        f"8. Per-condition best high-conf no-harm recovered: `{bool(best_trend['high_conf_no_harm_recovered'])}` ({int(best_trend['high_conf_no_harm_conditions'])}/6).",
        f"9. Mean motion usage ratio, global t=1: `{float(global_trend['mean_motion_usage_ratio']):.6f}`; per-condition best: `{float(best_trend['mean_motion_usage_ratio']):.6f}`.",
        f"10. Drift trend stable under global t=1: `{drift_stable}`.",
        f"11. Burst trend stable under global t=1: `{burst_stable}`.",
        "12. Bias remains structurally limited by relative-displacement generation anchored at degraded y[0].",
        "",
        "## Recommendation",
        "",
    ]
    if bool(global_trend["fused_better_than_noisy_ge4of6"]) and bool(global_trend["high_conf_no_harm_recovered"]) and bool(global_trend["not_noisy_reversion"]):
        lines.append("The expanded hold-out result supports moving to formal E3 pre-registration / final report framing for confidence-aware SDEdit.")
    elif bool(best_trend["fused_better_than_noisy_ge4of6"]) and bool(best_trend["not_noisy_reversion"]):
        lines.append("The mechanism trend is present under the diagnostic upper-bound protocol, but the global fixed t=1 protocol is weaker. Formal E3 should pre-register which protocol is primary before any final claim.")
    else:
        lines.append("The expanded hold-out result is not stable enough for a main E3 claim. Treat confidence-aware SDEdit as limited diagnostic evidence unless a pre-registered protocol is justified.")
    lines += [
        "",
        "## Output Files",
        "",
        "- `e3_holdout1000_vanilla_tstart_metrics.csv`",
        "- `e3_holdout1000_global_t1_metrics.csv`",
        "- `e3_holdout1000_per_condition_best_metrics.csv`",
        "- `e3_holdout1000_confidence_bin_metrics.csv`",
        "- `e3_holdout1000_paired_statistics.csv`",
        "- `e3_holdout1000_trend_stability_summary.csv`",
        "- `arrays/`",
        "- `figures/`",
    ]
    path = OUT_DIR / "e3_holdout1000_confidence_aware_sdedit_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(path)} written")


def main() -> None:
    ensure_dirs()
    clean = load_or_create_clean()
    degraded_map = generate_degradations(clean)

    device = torch.device("cpu")
    model, diffusion, rel_mean, rel_std = load_model(device)

    loaded: dict[str, dict[str, np.ndarray | float]] = {}
    for condition in CONDITIONS:
        degraded = degraded_map[condition].astype(np.float32)
        confidence, delta0 = compute_confidence(degraded, clean)
        if confidence.shape != (1000, 20):
            raise ValueError(f"{condition}: unexpected confidence shape {confidence.shape}")
        loaded[condition] = {"clean": clean, "degraded": degraded, "confidence": confidence, "delta0": delta0}
        np.save(ARRAY_DIR / f"{condition}_clean.npy", clean.astype(np.float32))
        np.save(ARRAY_DIR / f"{condition}_degraded.npy", degraded.astype(np.float32))
        np.save(ARRAY_DIR / f"{condition}_confidence.npy", confidence.astype(np.float32))
        print(f"[FILE] {rel(ARRAY_DIR / f'{condition}_confidence.npy')} written")

    sdedit_outputs: dict[tuple[str, int], np.ndarray] = {}
    for condition in CONDITIONS:
        degraded = loaded[condition]["degraded"]
        assert isinstance(degraded, np.ndarray)
        for t_start in T_STARTS:
            sdedit_outputs[(condition, t_start)] = run_or_load_sdedit(
                condition,
                t_start,
                degraded,
                model,
                diffusion,
                rel_mean,
                rel_std,
                device,
            )
        print(f"[DONE] SDEdit condition={condition}")

    vanilla_rows: list[dict] = []
    all_bin_rows: list[dict] = []
    paired_rows: list[dict] = []
    global_rows: list[dict] = []
    best_rows: list[dict] = []

    for condition in CONDITIONS:
        arrays = loaded[condition]
        degraded = arrays["degraded"]
        confidence = arrays["confidence"]
        delta0 = float(arrays["delta0"])
        assert isinstance(degraded, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        noisy_ade_traj = per_traj_ade(degraded, clean)
        noisy_high = masked_ade(degraded, clean, masks(confidence)["high"])
        noisy_low = masked_ade(degraded, clean, masks(confidence)["low"])

        for t_start in T_STARTS:
            pred = sdedit_outputs[(condition, t_start)]
            row = method_row(
                condition,
                f"sdedit_t{t_start}",
                "vanilla_tstart_sweep",
                pred,
                clean,
                degraded,
                confidence,
                noisy_ade_traj,
                t_start,
                correction_ref=pred,
                delta0=delta0,
            )
            row["noisy_ADE"] = float(np.mean(noisy_ade_traj))
            row["noisy_high_ADE"] = noisy_high
            row["noisy_low_ADE"] = noisy_low
            vanilla_rows.append(row)
            all_bin_rows.extend(confidence_bin_rows(condition, f"sdedit_t{t_start}", "vanilla_tstart_sweep", pred, clean, confidence, t_start))

    vanilla_df = pd.DataFrame(vanilla_rows)
    curve_map = {condition: curve_type(vanilla_df[vanilla_df["condition"] == condition]) for condition in CONDITIONS}
    vanilla_df["curve_type"] = vanilla_df["condition"].map(curve_map)
    save_csv(OUT_DIR / "e3_holdout1000_vanilla_tstart_metrics.csv", vanilla_df)

    for condition in CONDITIONS:
        arrays = loaded[condition]
        degraded = arrays["degraded"]
        confidence = arrays["confidence"]
        delta0 = float(arrays["delta0"])
        assert isinstance(degraded, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        noisy_ade_traj = per_traj_ade(degraded, clean)

        protocol_defs = [
            ("global_t1", GLOBAL_T_START),
            ("per_condition_best", PREVIOUS_BEST_T_START[condition]),
        ]
        for protocol, t_start in protocol_defs:
            vanilla = sdedit_outputs[(condition, t_start)]
            fused = hard_fuse(degraded, vanilla, confidence)
            if protocol == "global_t1":
                np.save(ARRAY_DIR / f"{condition}_fused_t1_tau07_gamma2.npy", fused.astype(np.float32))
            if protocol == "per_condition_best":
                np.save(ARRAY_DIR / f"{condition}_fused_best_tau07_gamma2.npy", fused.astype(np.float32))

            methods = {
                "noisy_input": degraded,
                "vanilla_SDEdit": vanilla,
                "fused_tau07_gamma2": fused,
            }
            rows_target = global_rows if protocol == "global_t1" else best_rows
            for method, pred in methods.items():
                row = method_row(
                    condition,
                    method,
                    protocol,
                    pred,
                    clean,
                    degraded,
                    confidence,
                    noisy_ade_traj,
                    t_start if method != "noisy_input" else math.nan,
                    vanilla_ref=vanilla if method == "fused_tau07_gamma2" else None,
                    correction_ref=vanilla,
                    delta0=delta0,
                )
                noisy_high_ade = masked_ade(degraded, clean, masks(confidence)["high"])
                if method == "noisy_input":
                    row["high_conf_no_harm_ratio"] = 1.0 if noisy_high_ade > 1e-12 else math.nan
                else:
                    row["high_conf_no_harm_ratio"] = row["high_conf_ADE"] / noisy_high_ade if noisy_high_ade > 1e-12 else math.nan
                vanilla_low = masked_ade(vanilla, clean, masks(confidence)["low"])
                row["low_conf_preservation_ratio"] = row["low_conf_ADE"] / vanilla_low if method == "fused_tau07_gamma2" and vanilla_low > 1e-12 else math.nan
                rows_target.append(row)
                all_bin_rows.extend(confidence_bin_rows(condition, method, protocol, pred, clean, confidence, t_start if method != "noisy_input" else math.nan))

                ade_traj = per_traj_ade(pred, clean)
                paired_rows.append(
                    {
                        "condition": condition,
                        "protocol": protocol,
                        "method": method,
                        "t_start": t_start if method != "noisy_input" else math.nan,
                        "N": int(clean.shape[0]),
                        "paired_mean_delta_ADE_vs_noisy": float(np.mean(ade_traj - noisy_ade_traj)),
                        "paired_median_delta_ADE_vs_noisy": float(np.median(ade_traj - noisy_ade_traj)),
                        "paired_wilcoxon_p_vs_noisy": wilcoxon_p(ade_traj - noisy_ade_traj),
                        "percent_improved": float(np.mean(ade_traj < noisy_ade_traj) * 100.0),
                    }
                )

            if protocol == "per_condition_best":
                plot_representative(condition, clean, degraded, confidence, vanilla, fused, t_start)

    global_df = pd.DataFrame(global_rows)
    best_df = pd.DataFrame(best_rows)
    bin_df = pd.DataFrame(all_bin_rows)
    paired_df = pd.DataFrame(paired_rows)
    trend_df = build_trend_summary(global_df, best_df)

    save_csv(OUT_DIR / "e3_holdout1000_global_t1_metrics.csv", global_df)
    save_csv(OUT_DIR / "e3_holdout1000_per_condition_best_metrics.csv", best_df)
    save_csv(OUT_DIR / "e3_holdout1000_confidence_bin_metrics.csv", bin_df)
    save_csv(OUT_DIR / "e3_holdout1000_paired_statistics.csv", paired_df)
    save_csv(OUT_DIR / "e3_holdout1000_trend_stability_summary.csv", trend_df)

    plot_overall(global_df, "global_t1", "overall_ade_comparison_global_t1.png")
    plot_overall(best_df, "per_condition_best", "overall_ade_comparison_per_condition_best.png")
    plot_high_low(global_df, "global_t1", "high_low_conf_comparison_global_t1.png")
    plot_tstart_sweep(vanilla_df)
    write_summary(vanilla_df, global_df, best_df, paired_df, trend_df)

    print("E3_HOLDOUT1000_CONFIDENCE_AWARE_SDEDIT_COMPLETE")
    print(f"clean_shape={clean.shape}")
    print(f"seed_range={SEED_START}-{SEED_END}")
    print(f"output_dir={OUT_DIR}")
    print(trend_df.to_string(index=False))


if __name__ == "__main__":
    main()
