from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_sdedit_hardfusion_mpl")

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
from tools.stage3_indoor.sdedit_gaussian_full import run_sdedit


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
TAU_GAMMA_SETTINGS = [
    (0.5, 1),
    (0.5, 2),
    (0.7, 1),
    (0.7, 2),
    (0.9, 1),
    (0.9, 2),
]

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "stage3_sdedit_tstart123_confidence_hardfusion"
ARRAY_DIR = OUT_DIR / "arrays"
FIG_DIR = OUT_DIR / "figures"

CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"
GENERALIZATION_DIR = (
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42"
)
CONF_CACHE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage4"
    / "e1_oracle_residual_gating_6conditions"
    / "confidence_cache"
)
LEGACY_GAUSSIAN_CACHE = (
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "cache" / "gaussian_uncond_sdedit_t2_refined.npy"
)

DEGRADED_PATHS = {
    "gaussian_medium": GENERALIZATION_DIR / "generalization_degraded_gaussian.npy",
    "drift_medium": GENERALIZATION_DIR / "generalization_degraded_drift.npy",
    "burst_medium": GENERALIZATION_DIR / "generalization_degraded_burst.npy",
    "bias_medium": GENERALIZATION_DIR / "generalization_degraded_bias.npy",
    "jump_medium": GENERALIZATION_DIR / "generalization_degraded_jump.npy",
    "combined_medium": GENERALIZATION_DIR / "generalization_degraded_combined.npy",
}


def ensure_dirs() -> None:
    for path in [OUT_DIR, ARRAY_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_model(device: torch.device) -> tuple[TemporalDenoiser1D, DDPMForwardProcess, np.ndarray, np.ndarray]:
    required = [CHECKPOINT_PATH, NORM_PATH]
    for path in required:
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


def load_inputs() -> dict[str, dict[str, np.ndarray]]:
    if not CLEAN_PATH.is_file():
        raise FileNotFoundError(f"Missing clean trajectories: {CLEAN_PATH}")
    clean = np.load(CLEAN_PATH).astype(np.float32)[:200]
    if clean.shape != (200, 20, 2):
        raise ValueError(f"Expected clean shape (200,20,2), got {clean.shape}")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for condition in CONDITIONS:
        degraded_path = DEGRADED_PATHS[condition]
        conf_path = CONF_CACHE_DIR / f"{condition}_confidence.npy"
        if not degraded_path.is_file():
            raise FileNotFoundError(f"Missing degraded input for {condition}: {degraded_path}")
        if not conf_path.is_file():
            raise FileNotFoundError(f"Missing validated confidence cache for {condition}: {conf_path}")
        degraded = np.load(degraded_path).astype(np.float32)[:200]
        confidence = np.load(conf_path).astype(np.float32)[:200]
        if degraded.shape != clean.shape:
            raise ValueError(f"{condition}: degraded shape {degraded.shape} != clean {clean.shape}")
        if confidence.shape != clean.shape[:2]:
            raise ValueError(f"{condition}: confidence shape {confidence.shape} != {(200,20)}")
        if not np.all((confidence >= 0.0) & (confidence <= 1.0)):
            raise ValueError(f"{condition}: confidence outside [0,1]")
        loaded[condition] = {"clean": clean, "degraded": degraded, "confidence": confidence}
    return loaded


def save_base_arrays(loaded: dict[str, dict[str, np.ndarray]]) -> None:
    for condition, arrays in loaded.items():
        np.save(ARRAY_DIR / f"{condition}_clean.npy", arrays["clean"].astype(np.float32))
        np.save(ARRAY_DIR / f"{condition}_degraded.npy", arrays["degraded"].astype(np.float32))
        np.save(ARRAY_DIR / f"{condition}_confidence.npy", arrays["confidence"].astype(np.float32))


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
        arr = np.load(out_path).astype(np.float32)
        if arr.shape == degraded.shape:
            print(f"[CACHE] {rel(out_path)}")
            return arr
    seed_preds = []
    for seed in SDEDIT_SEEDS:
        pred = run_sdedit(
            degraded_abs=degraded,
            model=model,
            diffusion=diffusion,
            rel_mean=rel_mean,
            rel_std=rel_std,
            t_start=t_start,
            sdedit_seed=seed,
            device=device,
        )
        seed_preds.append(pred.astype(np.float32))
    out = np.mean(np.stack(seed_preds, axis=0), axis=0).astype(np.float32)
    np.save(out_path, out)
    print(f"[FILE] {rel(out_path)} written")
    return out


def acceleration_rms(traj: np.ndarray) -> np.ndarray:
    acc = traj[:, 2:, :] - 2.0 * traj[:, 1:-1, :] + traj[:, :-2, :]
    acc_norm = np.linalg.norm(acc, axis=-1)
    return np.sqrt(np.mean(acc_norm**2, axis=1))


def per_traj_ade(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).mean(axis=1)


def per_traj_rmse(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    err = np.linalg.norm(pred - clean, axis=-1)
    return np.sqrt(np.mean(err**2, axis=1))


def masked_ade(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    err = np.linalg.norm(pred - clean, axis=-1)
    return float(np.mean(err[mask])) if int(mask.sum()) > 0 else float("nan")


def curve_type(rows: pd.DataFrame) -> str:
    sub = rows.sort_values("t_start")
    ades = sub["ADE"].to_numpy(dtype=float)
    noisy = float(sub["noisy_ADE"].iloc[0])
    if np.all(ades > noisy) and np.all(np.diff(ades) >= -1e-12):
        return "monotonic_worse"
    if np.all(np.diff(ades) <= 1e-12):
        return "monotonic_better"
    best_idx = int(np.argmin(ades))
    best_impr = (noisy - float(np.min(ades))) / noisy * 100.0
    if best_idx == 0 and best_impr > 0:
        return "tiny-t_only_improvement"
    if 0 < best_idx < len(ades) - 1 and best_impr > 0:
        return "small_U-shaped" if best_impr < 5.0 else "U-shaped"
    if np.max(np.abs(ades - noisy)) / max(noisy, 1e-12) < 0.02:
        return "flat"
    return "mixed"


def compute_method_metrics(
    condition: str,
    method: str,
    pred: np.ndarray,
    clean: np.ndarray,
    degraded: np.ndarray,
    confidence: np.ndarray,
    noisy_ade_traj: np.ndarray,
    correction_denom: float,
    t_start: int | float = np.nan,
    tau_high: float | float = np.nan,
    gamma: int | float = np.nan,
    vanilla_best: np.ndarray | None = None,
) -> dict:
    ade_traj = per_traj_ade(pred, clean)
    rmse_traj = per_traj_rmse(pred, clean)
    acc_traj = acceleration_rms(pred)
    noisy_ade = float(np.mean(noisy_ade_traj))
    correction_norm = float(np.mean(np.linalg.norm(pred - degraded, axis=-1)))
    motion_usage = correction_norm / correction_denom if correction_denom > 1e-12 else np.nan
    high = confidence > 0.7
    mid = (confidence >= 0.3) & (confidence <= 0.7)
    low = confidence < 0.3
    row = {
        "condition": condition,
        "method": method,
        "t_start": t_start,
        "tau_high": tau_high,
        "gamma": gamma,
        "ADE": float(np.mean(ade_traj)),
        "RMSE": float(np.mean(rmse_traj)),
        "acceleration_RMS": float(np.mean(acc_traj)),
        "high_conf_ADE": masked_ade(pred, clean, high),
        "mid_conf_ADE": masked_ade(pred, clean, mid),
        "low_conf_ADE": masked_ade(pred, clean, low),
        "noisy_reversion_gap": float(np.mean(ade_traj)) - noisy_ade,
        "motion_usage_ratio": motion_usage,
        "win_rate_vs_noisy_input": float(np.mean(ade_traj < noisy_ade_traj)),
    }
    if vanilla_best is not None:
        vanilla_ade_traj = per_traj_ade(vanilla_best, clean)
        row["win_rate_vs_vanilla_best"] = float(np.mean(ade_traj < vanilla_ade_traj))
    return row


def confidence_bin_rows(
    condition: str,
    method: str,
    pred: np.ndarray,
    clean: np.ndarray,
    confidence: np.ndarray,
    t_start: int | float = np.nan,
    tau_high: float | float = np.nan,
    gamma: int | float = np.nan,
) -> list[dict]:
    err = np.linalg.norm(pred - clean, axis=-1)
    masks = {
        "high": confidence > 0.7,
        "mid": (confidence >= 0.3) & (confidence <= 0.7),
        "low": confidence < 0.3,
    }
    rows = []
    for bin_name, mask in masks.items():
        vals = err[mask]
        rows.append(
            {
                "condition": condition,
                "method": method,
                "t_start": t_start,
                "tau_high": tau_high,
                "gamma": gamma,
                "bin": bin_name,
                "N_frames": int(mask.sum()),
                "ADE_mean": float(np.mean(vals)) if vals.size else np.nan,
                "ADE_std": float(np.std(vals)) if vals.size else np.nan,
                "ADE_median": float(np.median(vals)) if vals.size else np.nan,
                "ADE_min": float(np.min(vals)) if vals.size else np.nan,
                "ADE_max": float(np.max(vals)) if vals.size else np.nan,
                "ADE_p25": float(np.percentile(vals, 25)) if vals.size else np.nan,
                "ADE_p75": float(np.percentile(vals, 75)) if vals.size else np.nan,
            }
        )
    return rows


def setting_slug(tau_high: float, gamma: int) -> str:
    tau = int(round(tau_high * 10))
    return f"tau0{tau}_gamma{gamma}"


def hard_fuse(degraded: np.ndarray, sdedit: np.ndarray, confidence: np.ndarray, tau_high: float, gamma: int) -> np.ndarray:
    correction = sdedit - degraded
    lam = np.power(1.0 - confidence, gamma).astype(np.float32)
    lam = np.where(confidence > tau_high, 0.0, lam).astype(np.float32)
    return (degraded + lam[..., None] * correction).astype(np.float32)


def save_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[FILE] {rel(path)} written")


def plot_tstart_sweeps(vanilla_df: pd.DataFrame) -> None:
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
    fig.savefig(FIG_DIR / "tstart_sweep_ade_by_condition.png", dpi=180)
    plt.close(fig)
    print(f"[FILE] {rel(FIG_DIR / 'tstart_sweep_ade_by_condition.png')} written")


def plot_overall_comparison(best_df: pd.DataFrame, fusion_df: pd.DataFrame, best_setting: str) -> None:
    rows = []
    best_fusion = fusion_df[fusion_df["setting"] == best_setting]
    for condition in CONDITIONS:
        b = best_df[best_df["condition"] == condition].iloc[0]
        f = best_fusion[best_fusion["condition"] == condition].iloc[0]
        rows.extend(
            [
                {"condition": condition, "method": "noisy", "ADE": b["noisy_ADE"]},
                {"condition": condition, "method": "vanilla_best", "ADE": b["best_sdedit_ADE"]},
                {"condition": condition, "method": best_setting, "ADE": f["ADE"]},
            ]
        )
    df = pd.DataFrame(rows)
    x = np.arange(len(CONDITIONS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for i, method in enumerate(["noisy", "vanilla_best", best_setting]):
        sub = df[df["method"] == method].set_index("condition").loc[CONDITIONS]
        ax.bar(x + (i - 1) * width, sub["ADE"], width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=25, ha="right")
    ax.set_ylabel("ADE")
    ax.set_title("Overall ADE: noisy vs vanilla best vs best fusion")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overall_ade_noisy_vanilla_best_fusion.png", dpi=180)
    plt.close(fig)
    print(f"[FILE] {rel(FIG_DIR / 'overall_ade_noisy_vanilla_best_fusion.png')} written")


def plot_high_low_bars(best_df: pd.DataFrame, fusion_df: pd.DataFrame, best_setting: str) -> None:
    best_fusion = fusion_df[fusion_df["setting"] == best_setting]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    for ax, bin_name, noisy_col, vanilla_col, fusion_col in [
        (axes[0], "high", "noisy_high_ADE", "best_high_ADE", "high_conf_ADE"),
        (axes[1], "low", "noisy_low_ADE", "best_low_ADE", "low_conf_ADE"),
    ]:
        x = np.arange(len(CONDITIONS))
        width = 0.25
        noisy_vals = best_df.set_index("condition").loc[CONDITIONS][noisy_col]
        vanilla_vals = best_df.set_index("condition").loc[CONDITIONS][vanilla_col]
        fusion_vals = best_fusion.set_index("condition").loc[CONDITIONS][fusion_col]
        ax.bar(x - width, noisy_vals, width=width, label="noisy")
        ax.bar(x, vanilla_vals, width=width, label="vanilla best")
        ax.bar(x + width, fusion_vals, width=width, label=best_setting)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=25, ha="right")
        ax.set_ylabel(f"{bin_name}-confidence ADE")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "high_low_conf_ade_noisy_vanilla_fusion.png", dpi=180)
    plt.close(fig)
    print(f"[FILE] {rel(FIG_DIR / 'high_low_conf_ade_noisy_vanilla_fusion.png')} written")


def plot_representative(
    condition: str,
    arrays: dict[str, np.ndarray],
    best_t: int,
    fused_g1: np.ndarray,
    fused_g2: np.ndarray,
) -> None:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    sdedit = np.load(ARRAY_DIR / f"{condition}_sdedit_t{best_t}.npy").astype(np.float32)
    noisy_ade = per_traj_ade(degraded, clean)
    sdedit_ade = per_traj_ade(sdedit, clean)
    idx = int(np.argmax(np.abs(sdedit_ade - noisy_ade)))
    t = np.arange(clean.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(clean[idx, :, 0], clean[idx, :, 1], "k-", label="clean")
    ax.plot(degraded[idx, :, 0], degraded[idx, :, 1], color="tab:orange", label="degraded")
    ax.plot(sdedit[idx, :, 0], sdedit[idx, :, 1], color="tab:blue", label=f"SDEdit t={best_t}")
    ax.plot(fused_g1[idx, :, 0], fused_g1[idx, :, 1], color="tab:green", label="tau0.7 gamma1")
    ax.plot(fused_g2[idx, :, 0], fused_g2[idx, :, 1], color="tab:red", label="tau0.7 gamma2")
    ax.set_title(f"{condition} representative idx={idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(t, np.linalg.norm(degraded[idx] - clean[idx], axis=-1), color="tab:orange", label="noisy")
    ax.plot(t, np.linalg.norm(sdedit[idx] - clean[idx], axis=-1), color="tab:blue", label="SDEdit")
    ax.plot(t, np.linalg.norm(fused_g1[idx] - clean[idx], axis=-1), color="tab:green", label="fused g1")
    ax.plot(t, np.linalg.norm(fused_g2[idx] - clean[idx], axis=-1), color="tab:red", label="fused g2")
    ax.set_title("Per-frame error")
    ax.set_xlabel("frame")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(t, confidence[idx], color="tab:purple")
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


def write_summary(
    vanilla_df: pd.DataFrame,
    best_df: pd.DataFrame,
    fusion_df: pd.DataFrame,
    passfail_df: pd.DataFrame,
    best_setting: str,
    cache_ok: bool,
    legacy_anchor_delta: float | None,
) -> None:
    setting_row = passfail_df[passfail_df["setting"] == best_setting].iloc[0]
    best_rows = best_df.set_index("condition").loc[CONDITIONS]
    high_harm_count = int((best_rows["high_conf_delta"] > 0).sum())
    low_improve_count = int((best_rows["low_conf_delta"] < 0).sum())
    drift_fusion = fusion_df[(fusion_df["condition"] == "drift_medium") & (fusion_df["setting"] == best_setting)].iloc[0]
    burst_fusion = fusion_df[(fusion_df["condition"] == "burst_medium") & (fusion_df["setting"] == best_setting)].iloc[0]

    if bool(setting_row["case1_success"]):
        case = "Case 1: confidence-aware fusion succeeds"
    elif bool(setting_row["case2_high_fixed_overall_not"]):
        case = "Case 2: high-conf no-harm restored but overall ADE not improved enough"
    elif bool(setting_row["case3_high_still_harm"]):
        case = "Case 3: high-conf still harmed"
    elif bool(setting_row["case4_noisy_reversion"]):
        case = "Case 4: fusion close to noisy reversion"
    elif bool(setting_row["case5_condition_specific"]):
        case = "Case 5: condition-specific benefit"
    else:
        case = "Mixed / inconclusive"

    lines = [
        "# Stage 3 SDEdit t_start 1/2/3 + Confidence Hard-Fusion Diagnosis",
        "",
        "This is a mechanism diagnosis, not a formal E3 method. The indoor-v2 prior checkpoint and old vanilla SDEdit reverse process were left unchanged.",
        "",
        "## Setup",
        "",
        f"- checkpoint: `{rel(CHECKPOINT_PATH)}`",
        f"- normalization: `{rel(NORM_PATH)}`",
        f"- SDEdit seeds: `{SDEDIT_SEEDS}`",
        f"- t_start sweep: `{T_STARTS}`",
        f"- confidence source: validated reconstructed Formal E1 cache under `{rel(CONF_CACHE_DIR)}`",
        f"- 6 condition x 3 t_start cache generated successfully: `{cache_ok}`",
        f"- gaussian t2 regenerated-vs-legacy max abs delta: `{legacy_anchor_delta}`",
        "",
        "## Best t_start By Condition",
        "",
        best_df[
            [
                "condition",
                "best_t_start",
                "curve_type",
                "noisy_ADE",
                "best_sdedit_ADE",
                "delta_ADE",
                "relative_improvement_percent",
                "high_conf_delta",
                "low_conf_delta",
                "destructive_all_tstarts",
            ]
        ].to_string(index=False),
        "",
        "## Mechanism Answers",
        "",
        f"1. Cache complete: `{cache_ok}`.",
        "2. Best t_start values are listed above; the selection rule was min overall ADE, tie-broken by high-conf ADE.",
        "3. Curve types are condition-specific, not universally gaussian-like.",
        f"4. Vanilla high-conf harm count: `{high_harm_count}/6` conditions.",
        f"5. Vanilla low-conf improvement count: `{low_improve_count}/6` conditions.",
        f"6. Best fusion setting: `{best_setting}`.",
        f"7. Best fusion improves over noisy in `{int(setting_row['n_conditions_better_than_noisy'])}/6` conditions.",
        f"8. High-conf no-harm count under best fusion: `{int(setting_row['n_high_no_harm'])}/6` conditions.",
        f"9. Low-conf preservation count under best fusion: `{int(setting_row['n_low_preserved'])}/6` eligible conditions.",
        f"10. Mean motion usage ratio under best fusion: `{float(setting_row['mean_motion_usage_ratio']):.6f}`.",
        f"11. Drift best-fusion ADE delta vs noisy: `{float(drift_fusion['noisy_reversion_gap']):.6f}`; high no-harm ratio `{float(drift_fusion['high_conf_no_harm_ratio']):.6f}`.",
        f"12. Burst best-fusion ADE delta vs noisy: `{float(burst_fusion['noisy_reversion_gap']):.6f}`; high no-harm ratio `{float(burst_fusion['high_conf_no_harm_ratio']):.6f}`.",
        "13. Bias remains a structural limitation because the relative-displacement SDEdit reconstruction is anchored at degraded y[0].",
        f"14. Final case classification: `{case}`.",
        "",
        "## Interpretation",
        "",
    ]
    if case.startswith("Case 1"):
        lines.append(
            "Confidence-aware hard-threshold fusion supports the hypothesis that vanilla SDEdit's main weakness is missing confidence control. A formal confidence-aware SDEdit / E3 pre-registration is justified."
        )
    elif case.startswith("Case 2"):
        lines.append(
            "Confidence control reduces high-confidence harm, but the overall signal is not strong enough. This points to weak SDEdit correction strength or condition-specific prior mismatch rather than a pure gating problem."
        )
    elif case.startswith("Case 3"):
        lines.append(
            "High-confidence harm remains after hard-thresholding, so confidence alignment or frame correspondence should be audited before any formal confidence-aware SDEdit claim."
        )
    elif case.startswith("Case 4"):
        lines.append(
            "The best fusion mostly shuts off SDEdit and behaves like noisy input. That should not be treated as a successful prior-based correction."
        )
    elif case.startswith("Case 5"):
        lines.append(
            "The result is condition-specific: SDEdit/fusion can help some local-noise settings but does not support universal refinement across structured drift/burst/bias."
        )
    else:
        lines.append("The result is mixed and should be treated as diagnostic evidence only.")
    lines += [
        "",
        "## Output Files",
        "",
        "- `stage3_sdedit_vanilla_condition_metrics.csv`",
        "- `stage3_sdedit_vanilla_confidence_bin_metrics.csv`",
        "- `stage3_sdedit_tstart_sweep_summary.csv`",
        "- `stage3_sdedit_best_tstart_by_condition.csv`",
        "- `stage3_sdedit_hardfusion_condition_metrics.csv`",
        "- `stage3_sdedit_hardfusion_confidence_bin_metrics.csv`",
        "- `stage3_sdedit_hardfusion_passfail_diagnostic.csv`",
        "- `arrays/`",
        "- `figures/`",
    ]
    path = OUT_DIR / "stage3_sdedit_tstart123_hardfusion_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(path)} written")


def main() -> None:
    ensure_dirs()
    device = torch.device("cpu")
    loaded = load_inputs()
    save_base_arrays(loaded)
    model, diffusion, rel_mean, rel_std = load_model(device)

    sdedit_outputs: dict[tuple[str, int], np.ndarray] = {}
    for condition in CONDITIONS:
        degraded = loaded[condition]["degraded"]
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
        print(f"[DONE] vanilla SDEdit condition={condition}")

    legacy_delta = None
    if LEGACY_GAUSSIAN_CACHE.is_file():
        legacy = np.load(LEGACY_GAUSSIAN_CACHE).astype(np.float32)
        current = sdedit_outputs[("gaussian_medium", 2)]
        if legacy.shape == current.shape:
            legacy_delta = float(np.max(np.abs(legacy - current)))
            print(f"[CHECK] gaussian t2 regenerated-vs-legacy max abs delta = {legacy_delta:.6e}")

    vanilla_rows = []
    vanilla_bin_rows = []
    noisy_cache: dict[str, dict[str, float | np.ndarray]] = {}
    for condition in CONDITIONS:
        arrays = loaded[condition]
        clean = arrays["clean"]
        degraded = arrays["degraded"]
        confidence = arrays["confidence"]
        noisy_ade_traj = per_traj_ade(degraded, clean)
        noisy_cache[condition] = {
            "ade_traj": noisy_ade_traj,
            "high": masked_ade(degraded, clean, confidence > 0.7),
            "mid": masked_ade(degraded, clean, (confidence >= 0.3) & (confidence <= 0.7)),
            "low": masked_ade(degraded, clean, confidence < 0.3),
            "ADE": float(np.mean(noisy_ade_traj)),
        }
        for t_start in T_STARTS:
            pred = sdedit_outputs[(condition, t_start)]
            denom = float(np.mean(np.linalg.norm(pred - degraded, axis=-1)))
            row = compute_method_metrics(
                condition,
                f"sdedit_t{t_start}",
                pred,
                clean,
                degraded,
                confidence,
                noisy_ade_traj,
                correction_denom=denom,
                t_start=t_start,
            )
            row["noisy_ADE"] = noisy_cache[condition]["ADE"]
            vanilla_rows.append(row)
            vanilla_bin_rows.extend(confidence_bin_rows(condition, f"sdedit_t{t_start}", pred, clean, confidence, t_start=t_start))

    vanilla_df = pd.DataFrame(vanilla_rows)
    curve_map = {condition: curve_type(vanilla_df[vanilla_df["condition"] == condition]) for condition in CONDITIONS}
    vanilla_df["curve_type"] = vanilla_df["condition"].map(curve_map)
    save_csv(OUT_DIR / "stage3_sdedit_vanilla_condition_metrics.csv", vanilla_df)
    save_csv(OUT_DIR / "stage3_sdedit_vanilla_confidence_bin_metrics.csv", vanilla_bin_rows)

    sweep_summary_rows = []
    for condition in CONDITIONS:
        sub = vanilla_df[vanilla_df["condition"] == condition].sort_values("t_start")
        for _, row in sub.iterrows():
            sweep_summary_rows.append(
                {
                    "condition": condition,
                    "t_start": int(row["t_start"]),
                    "ADE": row["ADE"],
                    "noisy_ADE": row["noisy_ADE"],
                    "ADE_delta": row["ADE"] - row["noisy_ADE"],
                    "relative_improvement_percent": (row["noisy_ADE"] - row["ADE"]) / row["noisy_ADE"] * 100.0,
                    "high_conf_ADE": row["high_conf_ADE"],
                    "low_conf_ADE": row["low_conf_ADE"],
                    "acceleration_RMS": row["acceleration_RMS"],
                    "curve_type": curve_map[condition],
                }
            )
    sweep_df = pd.DataFrame(sweep_summary_rows)
    save_csv(OUT_DIR / "stage3_sdedit_tstart_sweep_summary.csv", sweep_df)

    best_rows = []
    best_t_map: dict[str, int] = {}
    for condition in CONDITIONS:
        sub = vanilla_df[vanilla_df["condition"] == condition].copy()
        min_ade = float(sub["ADE"].min())
        candidates = sub[np.abs(sub["ADE"] - min_ade) < 1e-4].copy()
        best = candidates.sort_values(["high_conf_ADE", "t_start"]).iloc[0]
        best_t = int(best["t_start"])
        best_t_map[condition] = best_t
        arrays = loaded[condition]
        clean = arrays["clean"]
        degraded = arrays["degraded"]
        confidence = arrays["confidence"]
        noisy_high = masked_ade(degraded, clean, confidence > 0.7)
        noisy_low = masked_ade(degraded, clean, confidence < 0.3)
        destructive = bool((sub["ADE"] > sub["noisy_ADE"]).all())
        best_rows.append(
            {
                "condition": condition,
                "best_t_start": best_t,
                "noisy_ADE": best["noisy_ADE"],
                "best_sdedit_ADE": best["ADE"],
                "delta_ADE": best["ADE"] - best["noisy_ADE"],
                "relative_improvement_percent": (best["noisy_ADE"] - best["ADE"]) / best["noisy_ADE"] * 100.0,
                "noisy_high_ADE": noisy_high,
                "best_high_ADE": best["high_conf_ADE"],
                "high_conf_delta": best["high_conf_ADE"] - noisy_high,
                "noisy_low_ADE": noisy_low,
                "best_low_ADE": best["low_conf_ADE"],
                "low_conf_delta": best["low_conf_ADE"] - noisy_low,
                "curve_type": curve_map[condition],
                "destructive_all_tstarts": destructive,
            }
        )
    best_df = pd.DataFrame(best_rows)
    save_csv(OUT_DIR / "stage3_sdedit_best_tstart_by_condition.csv", best_df)

    fusion_rows = []
    fusion_bin_rows = []
    for condition in CONDITIONS:
        arrays = loaded[condition]
        clean = arrays["clean"]
        degraded = arrays["degraded"]
        confidence = arrays["confidence"]
        noisy_ade_traj = noisy_cache[condition]["ade_traj"]
        best_t = best_t_map[condition]
        sdedit_best = sdedit_outputs[(condition, best_t)]
        denom = float(np.mean(np.linalg.norm(sdedit_best - degraded, axis=-1)))
        vanilla_low = float(best_df[best_df["condition"] == condition]["best_low_ADE"].iloc[0])
        noisy_high = float(best_df[best_df["condition"] == condition]["noisy_high_ADE"].iloc[0])
        for tau_high, gamma in TAU_GAMMA_SETTINGS:
            slug = setting_slug(tau_high, gamma)
            method = f"fused_{slug}"
            fused = hard_fuse(degraded, sdedit_best, confidence, tau_high, gamma)
            np.save(ARRAY_DIR / f"{condition}_fused_{slug}.npy", fused.astype(np.float32))
            row = compute_method_metrics(
                condition,
                method,
                fused,
                clean,
                degraded,
                confidence,
                noisy_ade_traj,
                correction_denom=denom,
                t_start=best_t,
                tau_high=tau_high,
                gamma=gamma,
                vanilla_best=sdedit_best,
            )
            row["setting"] = slug
            row["best_vanilla_t_start"] = best_t
            row["high_conf_no_harm_ratio"] = row["high_conf_ADE"] / noisy_high if noisy_high > 1e-12 else np.nan
            row["low_conf_preservation_ratio"] = row["low_conf_ADE"] / vanilla_low if vanilla_low > 1e-12 else np.nan
            fusion_rows.append(row)
            fusion_bin_rows.extend(
                confidence_bin_rows(condition, method, fused, clean, confidence, t_start=best_t, tau_high=tau_high, gamma=gamma)
            )
    fusion_df = pd.DataFrame(fusion_rows)
    save_csv(OUT_DIR / "stage3_sdedit_hardfusion_condition_metrics.csv", fusion_df)
    save_csv(OUT_DIR / "stage3_sdedit_hardfusion_confidence_bin_metrics.csv", fusion_bin_rows)

    passfail_rows = []
    for slug, sub in fusion_df.groupby("setting"):
        merged = sub.merge(best_df, on="condition", suffixes=("", "_best"))
        better_than_noisy = int((merged["ADE"] < merged["noisy_ADE"]).sum())
        high_no_harm = int((merged["high_conf_no_harm_ratio"] <= 1.05).sum())
        eligible_low = merged[merged["best_low_ADE"] < merged["noisy_low_ADE"]]
        low_preserved = int((eligible_low["low_conf_ADE"] < eligible_low["noisy_low_ADE"]).sum())
        mean_motion = float(merged["motion_usage_ratio"].mean())
        drift_ok = bool(
            (
                (merged["condition"] == "drift_medium")
                & (merged["ADE"] < merged["noisy_ADE"])
            ).any()
        )
        burst_ok = bool(
            (
                (merged["condition"] == "burst_medium")
                & (merged["ADE"] < merged["noisy_ADE"])
            ).any()
        )
        high_fixed_but_overall_not = high_no_harm >= 5 and better_than_noisy < 4
        high_still_harm = high_no_harm < 4
        noisy_reversion = bool(abs(float(merged["noisy_reversion_gap"].mean())) < 0.001 and mean_motion < 0.1)
        condition_specific = better_than_noisy < 6 and (drift_ok is False or burst_ok is False)
        case1 = bool(better_than_noisy >= 4 and high_no_harm >= 5 and low_preserved >= max(1, int(len(eligible_low) * 0.5)) and mean_motion >= 0.1)
        passfail_rows.append(
            {
                "setting": slug,
                "n_conditions_better_than_noisy": better_than_noisy,
                "n_high_no_harm": high_no_harm,
                "n_low_preserved": low_preserved,
                "n_low_eligible": int(len(eligible_low)),
                "mean_ADE": float(merged["ADE"].mean()),
                "mean_noisy_ADE": float(merged["noisy_ADE"].mean()),
                "mean_vanilla_best_ADE": float(merged["best_sdedit_ADE"].mean()),
                "mean_motion_usage_ratio": mean_motion,
                "mean_noisy_reversion_gap": float(merged["noisy_reversion_gap"].mean()),
                "drift_better_than_noisy": drift_ok,
                "burst_better_than_noisy": burst_ok,
                "case1_success": case1,
                "case2_high_fixed_overall_not": bool(high_fixed_but_overall_not and not case1),
                "case3_high_still_harm": bool(high_still_harm),
                "case4_noisy_reversion": noisy_reversion,
                "case5_condition_specific": bool(condition_specific and not case1 and not high_still_harm and not noisy_reversion),
            }
        )
    passfail_df = pd.DataFrame(passfail_rows).sort_values(
        ["n_conditions_better_than_noisy", "n_high_no_harm", "mean_ADE"],
        ascending=[False, False, True],
    )
    save_csv(OUT_DIR / "stage3_sdedit_hardfusion_passfail_diagnostic.csv", passfail_df)

    best_setting = str(passfail_df.iloc[0]["setting"])
    plot_tstart_sweeps(vanilla_df)
    plot_overall_comparison(best_df, fusion_df, best_setting)
    plot_high_low_bars(best_df, fusion_df, best_setting)
    for condition in CONDITIONS:
        arrays = loaded[condition]
        best_t = best_t_map[condition]
        fused_g1 = np.load(ARRAY_DIR / f"{condition}_fused_tau07_gamma1.npy").astype(np.float32)
        fused_g2 = np.load(ARRAY_DIR / f"{condition}_fused_tau07_gamma2.npy").astype(np.float32)
        plot_representative(condition, arrays, best_t, fused_g1, fused_g2)

    cache_ok = all((ARRAY_DIR / f"{condition}_sdedit_t{t}.npy").is_file() for condition in CONDITIONS for t in T_STARTS)
    write_summary(vanilla_df, best_df, fusion_df, passfail_df, best_setting, cache_ok, legacy_delta)

    print("STAGE3_SDEDIT_TSTART123_HARDFUSION_COMPLETE")
    print(f"output_dir={OUT_DIR}")
    print(f"cache_complete={cache_ok}")
    print("best_t_start_by_condition=" + json.dumps(best_t_map, sort_keys=True))
    print(f"best_fusion_setting={best_setting}")
    print(f"passfail={rel(OUT_DIR / 'stage3_sdedit_hardfusion_passfail_diagnostic.csv')}")


if __name__ == "__main__":
    main()
