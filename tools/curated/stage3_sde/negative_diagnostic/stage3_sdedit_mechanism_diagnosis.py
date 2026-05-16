from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_sdedit_mechanism_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "stage3_sdedit_mechanism_diagnosis"
FIG_DIR = OUT_DIR / "figures"

CONDITIONS = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
COND_TO_PREFIX = {
    "gaussian_medium": "gaussian",
    "drift_medium": "drift",
    "burst_medium": "burst",
    "bias_medium": "bias",
    "jump_medium": "jump",
    "combined_medium": "combined",
}

CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
FORMAL_E1_CONF_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage4"
    / "e1_oracle_residual_gating_6conditions"
    / "confidence_cache"
)

REPORT_CACHE = PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "cache"
GENERALIZATION_DIR = (
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42"
)
DDPM_V2_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42"
SDEDIT_DIAG_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "sdedit_diagnostic"
OLD_PRIOR_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_prior"

SEARCH_ROOTS = [
    PROJECT_ROOT / "outputs" / "stage3_indoor",
    PROJECT_ROOT / "docs" / "stage3",
    PROJECT_ROOT / "docs" / "stage4",
]
KEYWORDS = [
    "sdedit",
    "sde",
    "refined",
    "denoise",
    "gaussian",
    "drift",
    "burst",
    "bias",
    "jump",
    "combined",
    "t_start",
    "tau",
    "generalization",
    "diagnostic",
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def metric_arrays(pred: np.ndarray, clean: np.ndarray) -> dict[str, np.ndarray]:
    err = np.linalg.norm(pred - clean, axis=-1)
    ade = err.mean(axis=1)
    rmse = np.sqrt(np.mean(err**2, axis=1))
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    acc_norm = np.linalg.norm(acc, axis=-1)
    acc_mean = acc_norm.mean(axis=1)
    acc_rms = np.sqrt(np.mean(acc_norm**2, axis=1))
    return {"frame_error": err, "ADE": ade, "RMSE": rmse, "acc_mean": acc_mean, "acc_rms": acc_rms}


def summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p25": np.nan,
            "p75": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def search_artifacts() -> pd.DataFrame:
    rows = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            rows.append(
                {
                    "path": rel(root),
                    "exists": False,
                    "suffix": "",
                    "size_bytes": np.nan,
                    "matched_keywords": "",
                }
            )
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            matches = [kw for kw in KEYWORDS if kw in lower]
            if not matches:
                continue
            rows.append(
                {
                    "path": rel(path),
                    "exists": True,
                    "suffix": path.suffix,
                    "size_bytes": path.stat().st_size,
                    "matched_keywords": ";".join(matches),
                }
            )
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def inventory_refined_outputs() -> tuple[pd.DataFrame, dict[str, Path]]:
    rows = []
    found: dict[str, Path] = {}

    # Expected final six-condition t=2 cache pattern.
    for condition in CONDITIONS:
        prefix = COND_TO_PREFIX[condition]
        path = REPORT_CACHE / f"{prefix}_uncond_sdedit_t2_refined.npy"
        exists = path.is_file()
        if exists:
            found[f"{condition}:t2"] = path
        rows.append(
            {
                "scope": "six_condition_t2_expected",
                "condition": condition,
                "t_start": 2,
                "path": rel(path),
                "exists": exists,
                "shape": str(np.load(path).shape) if exists else "",
                "note": "expected per-frame uncond SDEdit t=2 output",
            }
        )

    # Expected gaussian full sweep cache pattern; these are not present in the legacy outputs.
    for t_start in [1, 2, 3, 5]:
        candidate_paths = [
            DDPM_V2_DIR / f"sdedit_t{t_start}_refined.npy",
            DDPM_V2_DIR / f"gaussian_sdedit_t{t_start}_refined.npy",
            DDPM_V2_DIR / f"gaussian_uncond_sdedit_t{t_start}_refined.npy",
            REPORT_CACHE / f"gaussian_uncond_sdedit_t{t_start}_refined.npy",
        ]
        existing = next((p for p in candidate_paths if p.is_file()), None)
        if existing is not None:
            found[f"gaussian_medium:t{t_start}"] = existing
        rows.append(
            {
                "scope": "gaussian_tstart_sweep_expected",
                "condition": "gaussian_medium",
                "t_start": t_start,
                "path": rel(existing) if existing is not None else "; ".join(rel(p) for p in candidate_paths),
                "exists": existing is not None,
                "shape": str(np.load(existing).shape) if existing is not None else "",
                "note": "candidate locations for per-frame gaussian SDEdit sweep output",
            }
        )

    return pd.DataFrame(rows), found


def build_condition_metrics() -> pd.DataFrame:
    rows = []
    gen_path = GENERALIZATION_DIR / "generalization_summary.csv"
    gen = load_csv(gen_path)
    if gen is not None:
        for condition in CONDITIONS:
            sub = gen[gen["degradation"] == condition]
            noisy = sub[sub["method"] == "noisy_input"]
            sdedit = sub[sub["method"] == "uncond_sdedit_t2"]
            if noisy.empty or sdedit.empty:
                continue
            noisy_row = noisy.iloc[0]
            sdedit_row = sdedit.iloc[0]
            noisy_ade = float(noisy_row["ADE_mean"])
            sdedit_ade = float(sdedit_row["ADE_mean"])
            rows.append(
                {
                    "source": rel(gen_path),
                    "diagnostic_level": "summary_level_only",
                    "condition": condition,
                    "t_start": 2,
                    "method": "uncond_sdedit_t2",
                    "noisy_ADE": noisy_ade,
                    "sdedit_ADE": sdedit_ade,
                    "ADE_delta": sdedit_ade - noisy_ade,
                    "ADE_relative_improvement_pct": (noisy_ade - sdedit_ade) / noisy_ade * 100.0,
                    "noisy_RMSE": np.nan,
                    "sdedit_RMSE": np.nan,
                    "RMSE_delta": np.nan,
                    "noisy_acceleration": float(noisy_row.get("smooth_mean", np.nan)),
                    "sdedit_acceleration": float(sdedit_row.get("smooth_mean", np.nan)),
                    "acceleration_delta": float(sdedit_row.get("smooth_mean", np.nan))
                    - float(noisy_row.get("smooth_mean", np.nan)),
                    "improved_fraction": float(sdedit_row.get("improved_fraction", np.nan)),
                    "wilcoxon_p": float(sdedit_row.get("wilcoxon_p_vs_noisy", np.nan)),
                    "note": "from saved Stage 3 generalization summary; no per-frame refined output required",
                }
            )

    full_path = DDPM_V2_DIR / "sdedit_gaussian_full_summary.csv"
    full = load_csv(full_path)
    if full is not None:
        noisy = full[full["method"] == "noisy_input"]
        if not noisy.empty:
            noisy_row = noisy.iloc[0]
            for _, sdedit_row in full[full["method"].str.startswith("sdedit_t", na=False)].iterrows():
                noisy_ade = float(noisy_row["ADE_mean"])
                sdedit_ade = float(sdedit_row["ADE_mean"])
                rows.append(
                    {
                        "source": rel(full_path),
                        "diagnostic_level": "summary_level_only",
                        "condition": "gaussian_medium",
                        "t_start": int(sdedit_row["t_start"]),
                        "method": sdedit_row["method"],
                        "noisy_ADE": noisy_ade,
                        "sdedit_ADE": sdedit_ade,
                        "ADE_delta": sdedit_ade - noisy_ade,
                        "ADE_relative_improvement_pct": (noisy_ade - sdedit_ade) / noisy_ade * 100.0,
                        "noisy_RMSE": float(noisy_row.get("RMSE_mean", np.nan)),
                        "sdedit_RMSE": float(sdedit_row.get("RMSE_mean", np.nan)),
                        "RMSE_delta": float(sdedit_row.get("RMSE_mean", np.nan))
                        - float(noisy_row.get("RMSE_mean", np.nan)),
                        "noisy_acceleration": float(noisy_row.get("smooth_mean", np.nan)),
                        "sdedit_acceleration": float(sdedit_row.get("smooth_mean", np.nan)),
                        "acceleration_delta": float(sdedit_row.get("smooth_mean", np.nan))
                        - float(noisy_row.get("smooth_mean", np.nan)),
                        "improved_fraction": float(sdedit_row.get("improved_fraction_ADE", np.nan)),
                        "wilcoxon_p": float(sdedit_row.get("wilcoxon_p_vs_noisy", np.nan)),
                        "note": "from saved gaussian full t_start sweep summary",
                    }
                )
    return pd.DataFrame(rows)


def build_tstart_sweep() -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    classifications: dict[str, str] = {}

    def add_group(source: Path, df: pd.DataFrame, condition_col: str, method_prefix: str) -> None:
        for condition, sub in df.groupby(condition_col):
            noisy = sub[sub["method"] == "noisy_input"]
            if noisy.empty:
                continue
            noisy_ade = float(noisy.iloc[0]["ADE_mean"])
            sd = sub[sub["method"].astype(str).str.startswith(method_prefix, na=False)].copy()
            if sd.empty:
                continue
            sd = sd.sort_values("t_start")
            ades = sd["ADE_mean"].astype(float).to_numpy()
            ts = sd["t_start"].astype(int).to_numpy()
            if np.all(ades > noisy_ade) and np.all(np.diff(ades) >= -1e-12):
                cls = "monotonic_worse"
            elif np.min(ades) < noisy_ade and len(ades) >= 3 and 0 < int(np.argmin(ades)) < len(ades) - 1:
                if np.max((noisy_ade - ades) / noisy_ade * 100.0) < 5.0:
                    cls = "small_U_shaped_sweet_spot"
                else:
                    cls = "U_shaped"
            elif np.max(np.abs(ades - noisy_ade)) / noisy_ade < 0.02:
                cls = "flat"
            elif np.min(ades) < noisy_ade and ts[int(np.argmin(ades))] <= 2:
                cls = "small_improvement_only_at_tiny_t"
            else:
                cls = "mixed"
            classifications[f"{rel(source)}::{condition}"] = cls
            for _, row in sd.iterrows():
                rows.append(
                    {
                        "source": rel(source),
                        "condition": condition,
                        "t_start": int(row["t_start"]),
                        "ADE": float(row["ADE_mean"]),
                        "noisy_ADE": noisy_ade,
                        "ADE_delta": float(row["ADE_mean"]) - noisy_ade,
                        "ADE_relative_improvement_pct": (noisy_ade - float(row["ADE_mean"])) / noisy_ade * 100.0,
                        "RMSE": float(row.get("RMSE_mean", np.nan)),
                        "acceleration": float(row.get("smooth_mean", np.nan)),
                        "classification": cls,
                    }
                )

    full = load_csv(DDPM_V2_DIR / "sdedit_gaussian_full_summary.csv")
    if full is not None:
        add_group(DDPM_V2_DIR / "sdedit_gaussian_full_summary.csv", full, "degradation", "sdedit_t")

    scout = load_csv(DDPM_V2_DIR / "sdedit_scout_results.csv")
    if scout is not None:
        add_group(DDPM_V2_DIR / "sdedit_scout_results.csv", scout, "degradation", "sdedit_t")

    diag = load_csv(SDEDIT_DIAG_DIR / "diagnostic_summary.csv")
    if diag is not None:
        add_group(SDEDIT_DIAG_DIR / "diagnostic_summary.csv", diag, "degradation", "ddpm_sdedit_t")

    return pd.DataFrame(rows), classifications


def confidence_bin_diagnosis(found_refined: dict[str, Path]) -> pd.DataFrame:
    rows = []
    key = "gaussian_medium:t2"
    if key not in found_refined:
        return pd.DataFrame(
            [
                {
                    "condition": "gaussian_medium",
                    "t_start": 2,
                    "bin": "unavailable",
                    "N_frames": 0,
                    "noisy_ADE": np.nan,
                    "sdedit_ADE": np.nan,
                    "ADE_delta": np.nan,
                    "note": "gaussian t2 refined cache not found",
                }
            ]
        )
    conf_path = FORMAL_E1_CONF_DIR / "gaussian_medium_confidence.npy"
    degraded_path = DDPM_V2_DIR / "gaussian_full_degraded.npy"
    if not (CLEAN_PATH.is_file() and conf_path.is_file() and degraded_path.is_file()):
        return pd.DataFrame(
            [
                {
                    "condition": "gaussian_medium",
                    "t_start": 2,
                    "bin": "unavailable",
                    "N_frames": 0,
                    "noisy_ADE": np.nan,
                    "sdedit_ADE": np.nan,
                    "ADE_delta": np.nan,
                    "note": "missing clean/degraded/confidence cache needed for bin diagnosis",
                }
            ]
        )

    clean = np.load(CLEAN_PATH).astype(np.float32)[:200]
    degraded = np.load(degraded_path).astype(np.float32)[:200]
    refined = np.load(found_refined[key]).astype(np.float32)
    conf = np.load(conf_path).astype(np.float32)[:200]
    if clean.shape != degraded.shape or clean.shape != refined.shape or conf.shape != clean.shape[:2]:
        return pd.DataFrame(
            [
                {
                    "condition": "gaussian_medium",
                    "t_start": 2,
                    "bin": "unavailable",
                    "N_frames": 0,
                    "noisy_ADE": np.nan,
                    "sdedit_ADE": np.nan,
                    "ADE_delta": np.nan,
                    "note": f"shape mismatch clean={clean.shape}, degraded={degraded.shape}, refined={refined.shape}, conf={conf.shape}",
                }
            ]
        )

    noisy_err = np.linalg.norm(degraded - clean, axis=-1)
    sdedit_err = np.linalg.norm(refined - clean, axis=-1)
    masks = {
        "high": conf > 0.7,
        "mid": (conf >= 0.3) & (conf <= 0.7),
        "low": conf < 0.3,
    }
    for bin_name, mask in masks.items():
        n = int(mask.sum())
        noisy_ade = float(np.mean(noisy_err[mask])) if n else np.nan
        sdedit_ade = float(np.mean(sdedit_err[mask])) if n else np.nan
        rows.append(
            {
                "condition": "gaussian_medium",
                "t_start": 2,
                "bin": bin_name,
                "N_frames": n,
                "noisy_ADE": noisy_ade,
                "sdedit_ADE": sdedit_ade,
                "ADE_delta": sdedit_ade - noisy_ade if n else np.nan,
                "note": "partial per-frame diagnosis using validated reconstructed Formal E1 confidence cache",
            }
        )
    return pd.DataFrame(rows)


def prior_sample_diagnostics() -> pd.DataFrame:
    rows = []
    clean = np.load(CLEAN_PATH).astype(np.float32) if CLEAN_PATH.is_file() else None
    candidates = [
        OLD_PRIOR_DIR / "generated_abs.npy",
        PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_prior_diagnostics" / "generated_abs_check.npy",
    ]
    for path in candidates:
        if not path.is_file():
            rows.append(
                {
                    "source": rel(path),
                    "exists": False,
                    "group": "prior_sample",
                    "n": 0,
                    "step_norm_mean": np.nan,
                    "step_norm_p95": np.nan,
                    "total_length_mean": np.nan,
                    "endpoint_displacement_mean": np.nan,
                    "acceleration_rms_mean": np.nan,
                    "off_room_ratio": np.nan,
                    "nearest_neighbor_l2_mean": np.nan,
                    "note": "prior sample output not found; no new sampling run",
                }
            )
            continue
        arr = np.load(path).astype(np.float32)
        stats = trajectory_stats(arr)
        nn = np.nan
        if clean is not None:
            nn = nearest_neighbor_distance(arr, clean)
        rows.append(
            {
                "source": rel(path),
                "exists": True,
                "group": "prior_sample",
                "n": int(arr.shape[0]),
                "step_norm_mean": stats["step_norm_mean"],
                "step_norm_p95": stats["step_norm_p95"],
                "total_length_mean": stats["total_length_mean"],
                "endpoint_displacement_mean": stats["endpoint_displacement_mean"],
                "acceleration_rms_mean": stats["acceleration_rms_mean"],
                "off_room_ratio": stats["off_room_ratio"],
                "nearest_neighbor_l2_mean": nn,
                "note": "existing old prior sample output; not newly sampled",
            }
        )
    if clean is not None:
        stats = trajectory_stats(clean)
        rows.append(
            {
                "source": rel(CLEAN_PATH),
                "exists": True,
                "group": "clean_reference",
                "n": int(clean.shape[0]),
                "step_norm_mean": stats["step_norm_mean"],
                "step_norm_p95": stats["step_norm_p95"],
                "total_length_mean": stats["total_length_mean"],
                "endpoint_displacement_mean": stats["endpoint_displacement_mean"],
                "acceleration_rms_mean": stats["acceleration_rms_mean"],
                "off_room_ratio": stats["off_room_ratio"],
                "nearest_neighbor_l2_mean": 0.0,
                "note": "clean indoor reference",
            }
        )
    return pd.DataFrame(rows)


def trajectory_stats(arr: np.ndarray) -> dict[str, float]:
    steps = arr[:, 1:, :] - arr[:, :-1, :]
    step_norm = np.linalg.norm(steps, axis=-1)
    acc = arr[:, 2:, :] - 2.0 * arr[:, 1:-1, :] + arr[:, :-2, :]
    acc_norm = np.linalg.norm(acc, axis=-1)
    endpoint = np.linalg.norm(arr[:, -1, :] - arr[:, 0, :], axis=-1)
    off_room = (arr[..., 0] < 0.0) | (arr[..., 0] > 3.0) | (arr[..., 1] < 0.0) | (arr[..., 1] > 3.0)
    return {
        "step_norm_mean": float(step_norm.mean()),
        "step_norm_p95": float(np.percentile(step_norm, 95)),
        "total_length_mean": float(step_norm.sum(axis=1).mean()),
        "endpoint_displacement_mean": float(endpoint.mean()),
        "acceleration_rms_mean": float(np.sqrt(np.mean(acc_norm**2))),
        "off_room_ratio": float(off_room.mean()),
    }


def nearest_neighbor_distance(samples: np.ndarray, clean: np.ndarray) -> float:
    n_samples = min(samples.shape[0], 300)
    n_clean = min(clean.shape[0], 2000)
    s = samples[:n_samples].reshape(n_samples, -1).astype(np.float64)
    c = clean[:n_clean].reshape(n_clean, -1).astype(np.float64)
    mins = []
    chunk = 50
    for start in range(0, n_samples, chunk):
        sub = s[start : start + chunk]
        d2 = ((sub[:, None, :] - c[None, :, :]) ** 2).mean(axis=2)
        mins.append(np.sqrt(np.min(d2, axis=1)))
    return float(np.concatenate(mins).mean())


def plot_tstart_sweeps(tstart_df: pd.DataFrame) -> None:
    if tstart_df.empty:
        return
    for source, source_df in tstart_df.groupby("source"):
        conditions = list(source_df["condition"].unique())
        fig, axes = plt.subplots(1, len(conditions), figsize=(5.0 * len(conditions), 4.0), squeeze=False)
        for ax, condition in zip(axes[0], conditions):
            sub = source_df[source_df["condition"] == condition].sort_values("t_start")
            ax.plot(sub["t_start"], sub["ADE"], marker="o", linewidth=2, label="SDEdit")
            ax.axhline(float(sub["noisy_ADE"].iloc[0]), color="gray", linestyle="--", label="noisy")
            ax.set_title(f"{condition}\n{sub['classification'].iloc[0]}")
            ax.set_xlabel("t_start")
            ax.set_ylabel("ADE")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        slug = Path(source).stem.replace("/", "_")
        fig.savefig(FIG_DIR / f"tstart_sweep_{slug}.png", dpi=180)
        plt.close(fig)


def plot_gaussian_partial_case(found_refined: dict[str, Path]) -> bool:
    key = "gaussian_medium:t2"
    if key not in found_refined:
        return False
    conf_path = FORMAL_E1_CONF_DIR / "gaussian_medium_confidence.npy"
    degraded_path = DDPM_V2_DIR / "gaussian_full_degraded.npy"
    if not (CLEAN_PATH.is_file() and conf_path.is_file() and degraded_path.is_file()):
        return False
    clean = np.load(CLEAN_PATH).astype(np.float32)[:200]
    degraded = np.load(degraded_path).astype(np.float32)[:200]
    refined = np.load(found_refined[key]).astype(np.float32)
    conf = np.load(conf_path).astype(np.float32)[:200]
    if clean.shape != degraded.shape or clean.shape != refined.shape or conf.shape != clean.shape[:2]:
        return False
    noisy_ade = np.linalg.norm(degraded - clean, axis=-1).mean(axis=1)
    sdedit_ade = np.linalg.norm(refined - clean, axis=-1).mean(axis=1)
    idx = int(np.argmax(noisy_ade - sdedit_ade))
    t = np.arange(clean.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(clean[idx, :, 0], clean[idx, :, 1], "k-", label="clean")
    axes[0].plot(degraded[idx, :, 0], degraded[idx, :, 1], color="tab:orange", label="degraded")
    axes[0].plot(refined[idx, :, 0], refined[idx, :, 1], color="tab:blue", label="SDEdit t=2")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(f"gaussian t=2 best improvement idx={idx}")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(t, np.linalg.norm(degraded[idx] - clean[idx], axis=-1), color="tab:orange", label="noisy error")
    axes[1].plot(t, np.linalg.norm(refined[idx] - clean[idx], axis=-1), color="tab:blue", label="SDEdit error")
    axes[1].set_title("Per-frame error")
    axes[1].set_xlabel("frame")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    axes[2].plot(t, conf[idx], color="tab:green")
    axes[2].axhline(0.7, color="gray", linestyle="--", linewidth=1)
    axes[2].axhline(0.3, color="gray", linestyle=":", linewidth=1)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Formal E1 confidence cache")
    axes[2].set_xlabel("frame")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "representative_gaussian_t2_partial_cache.png", dpi=180)
    plt.close(fig)
    return True


def write_inventory_md(artifact_df: pd.DataFrame, refined_df: pd.DataFrame) -> None:
    found_refined = refined_df[refined_df["exists"] == True]
    missing_refined = refined_df[refined_df["exists"] != True]
    unique_found = found_refined.drop_duplicates(subset=["path"])
    lines = [
        "# Stage 3 SDEdit Output Inventory",
        "",
        "Static inventory only. No SDEdit trajectory was regenerated.",
        "",
        "## Refined Trajectory Outputs",
        "",
        f"- expected refined-output entries checked: {len(refined_df)}",
        f"- found unique files: {len(unique_found)}",
        f"- missing entries: {len(missing_refined)}",
        "",
        "### Found",
        "",
    ]
    if unique_found.empty:
        lines.append("- None")
    else:
        for _, row in unique_found.iterrows():
            lines.append(f"- `{row['path']}` shape `{row['shape']}`")
    lines += ["", "### Missing", ""]
    for _, row in missing_refined.iterrows():
        lines.append(f"- `{row['scope']}` condition=`{row['condition']}` t_start=`{row['t_start']}` candidates=`{row['path']}`")
    lines += [
        "",
        "## Other Legacy Artifacts Found",
        "",
        f"- keyword-matched files: {len(artifact_df)}",
        "",
        "The detailed artifact search is reflected in `missing_output_inventory.md` and in the generated CSV diagnostics.",
    ]
    (OUT_DIR / "stage3_sdedit_output_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_missing_inventory_md(refined_df: pd.DataFrame) -> None:
    missing_refined = refined_df[refined_df["exists"] != True]
    lines = [
        "# Missing Output Inventory",
        "",
        "The old Stage 3 SDEdit refined trajectory archive is incomplete for full mechanism diagnosis.",
        "",
        "Only existing outputs were read. No old SDEdit script was rerun.",
        "",
        "## Found Per-Frame SDEdit Refined Outputs",
        "",
    ]
    found = refined_df[refined_df["exists"] == True]
    unique_found = found.drop_duplicates(subset=["path"])
    if unique_found.empty:
        lines.append("- None")
    else:
        for _, row in unique_found.iterrows():
            lines.append(f"- `{row['path']}` shape `{row['shape']}`")
    lines += [
        "",
        "## Missing Per-Frame Outputs Required For Full Diagnosis",
        "",
    ]
    for _, row in missing_refined.iterrows():
        lines.append(
            f"- scope=`{row['scope']}`, condition=`{row['condition']}`, t_start=`{row['t_start']}`; checked `{row['path']}`"
        )
    lines += [
        "",
        "## Consequence",
        "",
        "Full confidence-bin ADE decomposition and representative trajectory failure-mode plots cannot be computed for all six conditions or for every `t_start` without the per-frame SDEdit outputs.",
        "",
        "The script therefore reports:",
        "",
        "- summary-level condition and `t_start` metrics from saved CSV files;",
        "- a partial per-frame confidence-bin diagnosis only for `gaussian_medium`, `t_start=2`, because `outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t2_refined.npy` exists;",
        "- prior sample diagnostics only from old saved unconditional samples.",
        "",
        "## To Reproduce Missing Per-Frame Outputs Later",
        "",
        "Do not do this inside the current audit. If explicitly approved later, the closest old scripts are:",
        "",
        "- `tools/stage3_indoor/sdedit_gaussian_full.py` for gaussian `t_start=[1,2,3,5]`, checkpoint `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`, seeds `[42,43,44,45,46]`.",
        "- `tools/stage3_indoor/generalization_diagnostic.py` for six-condition `uncond_sdedit_t2`, same indoor-v2 prior and normalization.",
        "- `tools/stage3_indoor/diagnose_sdedit.py` for older diagnostic `t_start=[1,3,5,10,20]`, checkpoint `outputs/stage3_indoor/ddpm_prior/val_selected_model.pt`.",
    ]
    (OUT_DIR / "missing_output_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    table = df[cols].copy()

    def fmt(value) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "nan"
            return f"{float(value):.6f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if pd.isna(value):
            return "nan"
        return str(value)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in table.iterrows():
        body.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join([header, sep, *body])


def write_summary(
    refined_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    tstart_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    classifications: dict[str, str],
    gaussian_plot_written: bool,
) -> None:
    found_refined = refined_df[refined_df["exists"] == True]
    unique_found_refined = found_refined.drop_duplicates(subset=["path"])
    full_per_frame_available = bool(
        refined_df[refined_df["scope"].eq("six_condition_t2_expected")]["exists"].all()
        and refined_df[refined_df["scope"].eq("gaussian_tstart_sweep_expected")]["exists"].all()
    )
    condition_pivot = condition_df[
        (condition_df["source"].str.contains("generalization_summary", na=False))
        & (condition_df["method"] == "uncond_sdedit_t2")
    ].copy()
    improves = condition_pivot[condition_pivot["ADE_delta"] < 0]["condition"].tolist() if not condition_pivot.empty else []
    worsens = condition_pivot[condition_pivot["ADE_delta"] > 0]["condition"].tolist() if not condition_pivot.empty else []

    gaussian_conf_note = "unavailable"
    if not conf_df.empty and set(conf_df["bin"]) >= {"high", "mid", "low"}:
        high = conf_df[conf_df["bin"] == "high"].iloc[0]
        low = conf_df[conf_df["bin"] == "low"].iloc[0]
        gaussian_conf_note = (
            f"partial gaussian t=2: high delta={high['ADE_delta']:.6f}, "
            f"low delta={low['ADE_delta']:.6f}"
        )

    prior_reading = "unavailable"
    if not prior_df.empty and prior_df["exists"].fillna(False).any():
        sample = prior_df[(prior_df["exists"] == True) & (prior_df["group"] == "prior_sample")]
        clean = prior_df[prior_df["group"] == "clean_reference"]
        if not sample.empty and not clean.empty:
            s = sample.iloc[0]
            c = clean.iloc[0]
            step_ratio = float(s["step_norm_mean"]) / float(c["step_norm_mean"])
            acc_ratio = float(s["acceleration_rms_mean"]) / float(c["acceleration_rms_mean"])
            prior_reading = (
                f"old saved prior sample step mean ratio={step_ratio:.3f}, "
                f"acceleration RMS ratio={acc_ratio:.3f}, off-room={float(s['off_room_ratio']):.3f}"
            )

    lines = [
        "# Stage 3 SDEdit Mechanism Diagnosis",
        "",
        "Static mechanism diagnosis only. No model was trained, no checkpoint was modified, and no old SDEdit method was rerun.",
        "",
        "## 1. Do old SDEdit refined trajectories exist?",
        "",
        f"- full per-frame availability for all requested diagnostics: `{full_per_frame_available}`",
        f"- found unique per-frame SDEdit refined outputs: `{len(unique_found_refined)}`",
        "",
    ]
    if unique_found_refined.empty:
        lines.append("- Found: none")
    else:
        for _, row in unique_found_refined.iterrows():
            lines.append(f"- Found: `{row['path']}` shape `{row['shape']}`")
    lines += [
        "",
        "The archive is incomplete: only `gaussian_medium`, `t_start=2` has a per-frame SDEdit cache in the checked locations.",
        "",
        "## 2. Coverage",
        "",
        "Summary-level metrics cover six-condition `uncond_sdedit_t2` plus gaussian/burst legacy sweeps. Per-frame confidence-bin analysis only covers gaussian t=2.",
        "",
        "## 3. ADE Improvement By Condition",
        "",
    ]
    if not condition_pivot.empty:
        lines.append(
            format_table(
                condition_pivot,
                [
                    "condition",
                    "t_start",
                    "noisy_ADE",
                    "sdedit_ADE",
                    "ADE_delta",
                    "ADE_relative_improvement_pct",
                    "improved_fraction",
                    "noisy_acceleration",
                    "sdedit_acceleration",
                ],
            )
        )
    else:
        lines.append("_Condition-level rows unavailable._")
    lines += [
        "",
        f"Main improvements: `{', '.join(improves) if improves else 'none'}`.",
        f"Main degradations: `{', '.join(worsens) if worsens else 'none'}`.",
        "",
        "## 4. High / Mid / Low Confidence Bin Changes",
        "",
        gaussian_conf_note,
        "",
    ]
    if not conf_df.empty:
        lines.append(format_table(conf_df, ["condition", "t_start", "bin", "N_frames", "noisy_ADE", "sdedit_ADE", "ADE_delta"]))
    lines += [
        "",
        "Full six-condition confidence-bin diagnosis is unavailable because six-condition per-frame SDEdit outputs were not archived.",
        "",
        "## 5. t_start Curve Shape",
        "",
    ]
    for key, value in classifications.items():
        lines.append(f"- `{key}`: `{value}`")
    if not classifications:
        lines.append("- unavailable")
    lines += [
        "",
        "Interpretation: the final gaussian-v2 curve is a small U-shaped sweet spot with best `t_start=2`; scout/older burst curves are monotonic worse; the older prior diagnostic is monotonic worse for both gaussian and burst.",
        "",
        "## 6. Drift / Burst Failure Size",
        "",
    ]
    for condition in ["drift_medium", "burst_medium"]:
        sub = condition_pivot[condition_pivot["condition"] == condition]
        if sub.empty:
            lines.append(f"- `{condition}`: unavailable")
        else:
            row = sub.iloc[0]
            lines.append(
                f"- `{condition}`: ADE delta `{row['ADE_delta']:.6f}` "
                f"({row['ADE_relative_improvement_pct']:.3f}% relative improvement; negative means worse in the table convention)."
            )
    lines += [
        "",
        "Using the saved six-condition table, drift degradation is small in absolute ADE but meaningful relative to its low noisy baseline; burst degradation is larger and clearly destructive.",
        "",
        "## 7. Prior Sample Quality",
        "",
        prior_reading,
        "",
    ]
    if not prior_df.empty:
        lines.append(format_table(prior_df, ["source", "exists", "group", "n", "step_norm_mean", "step_norm_p95", "total_length_mean", "acceleration_rms_mean", "off_room_ratio", "nearest_neighbor_l2_mean"]))
    lines += [
        "",
        "The saved unconditional samples found here are from the older `ddpm_prior` line, not the final indoor-v2 EMA prior. The final v2 prior has `prior_check_v2.json`, but no saved generated trajectory array was found in the checked locations.",
        "",
        "## 8. Most Likely Mechanisms",
        "",
        "- `vanilla SDEdit lacks confidence awareness`: supported. It applies the prior uniformly after noising and has no per-frame reliability control.",
        "- `start_t not adaptive`: supported. Gaussian has a tiny sweet spot; burst worsens as intervention grows.",
        "- `relative representation cannot fix bias`: supported by no-change bias result.",
        "- `anchor preserves absolute offset`: supported by reconstruction with degraded `y[0]` anchor.",
        "- `prior too weak`: plausible but not fully decidable from archived final-v2 samples because final-v2 generated arrays were not saved; `prior_check_v2.json` says the prior passed its internal checks.",
        "- `domain / scale mismatch`: not primary for final indoor-v2 SDEdit because the prior is indoor-normalized; relevant for older Stage 2 prior blending.",
        "- `implementation issue`: no direct evidence from archived outputs; gaussian t=2 improvement and prior check suggest the pipeline ran coherently.",
        "",
        "## 9. Recommendation",
        "",
        "Recommendation: do not continue vanilla SDEdit as-is.",
        "",
        "A confidence-aware SDEdit follow-up is scientifically plausible only if it is explicitly framed as addressing the old failure mode: high-reliability frames should be protected while low-reliability frames receive prior intervention. But because the old archive lacks six-condition per-frame SDEdit outputs, first recreate the missing old outputs only if the goal is a strict apples-to-apples mechanism study.",
        "",
        "For reporting, the current old SDEdit result should be treated as a negative / limited prior-only result: small gaussian/jump gains, drift/burst harm, and no bias repair.",
        "",
        "## Figures",
        "",
        "- t_start sweep figures are under `outputs/stage4/stage3_sdedit_mechanism_diagnosis/figures/`.",
        f"- gaussian partial representative figure written: `{gaussian_plot_written}`.",
    ]
    (OUT_DIR / "stage3_sdedit_mechanism_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    artifact_df = search_artifacts()
    refined_df, found_refined = inventory_refined_outputs()

    write_inventory_md(artifact_df, refined_df)
    write_missing_inventory_md(refined_df)

    condition_df = build_condition_metrics()
    tstart_df, classifications = build_tstart_sweep()
    conf_df = confidence_bin_diagnosis(found_refined)
    prior_df = prior_sample_diagnostics()

    condition_df.to_csv(OUT_DIR / "stage3_sdedit_condition_metrics.csv", index=False)
    conf_df.to_csv(OUT_DIR / "stage3_sdedit_confidence_bin_metrics.csv", index=False)
    tstart_df.to_csv(OUT_DIR / "stage3_sdedit_tstart_sweep.csv", index=False)
    prior_df.to_csv(OUT_DIR / "stage3_sdedit_prior_sample_diagnostics.csv", index=False)

    plot_tstart_sweeps(tstart_df)
    gaussian_plot_written = plot_gaussian_partial_case(found_refined)

    write_summary(refined_df, condition_df, conf_df, tstart_df, prior_df, classifications, gaussian_plot_written)

    print("STAGE3_SDEDIT_MECHANISM_DIAGNOSIS_COMPLETE")
    print(f"output_dir={OUT_DIR}")
    print(f"found_per_frame_refined={len(found_refined)}")
    print(f"full_per_frame_available={bool(refined_df['exists'].all())}")
    print(f"condition_metrics={OUT_DIR / 'stage3_sdedit_condition_metrics.csv'}")
    print(f"summary={OUT_DIR / 'stage3_sdedit_mechanism_summary.md'}")


if __name__ == "__main__":
    main()
