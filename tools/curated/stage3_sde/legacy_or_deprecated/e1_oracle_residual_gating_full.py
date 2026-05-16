from __future__ import annotations

from pathlib import Path
import csv
import math
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / "stage4" / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
STAGE3_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor"
COND_DIR = STAGE3_DIR / "conditional_residual_ddpm_gaussian" / "seed42"
CACHE_DIR = STAGE3_DIR / "report" / "cache"

CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
STAGE3_GAUSS_SUMMARY_PATH = COND_DIR / "cond_residual_gaussian_summary.csv"
STAGE3_GENERALIZATION_SUMMARY_PATH = COND_DIR / "generalization_summary.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_oracle_residual_gating_full"
FIG_DIR = OUT_DIR / "figures"

FULL_METRICS_PATH = OUT_DIR / "e1_full_metrics.csv"
BIN_METRICS_PATH = OUT_DIR / "e1_confidence_bin_metrics.csv"
DELTA_SWEEP_PATH = OUT_DIR / "e1_delta0_sweep.csv"
PER_TRAJ_PATH = OUT_DIR / "e1_per_trajectory_metrics.csv"
PASS_FAIL_PATH = OUT_DIR / "e1_pass_fail_summary.csv"
SUMMARY_PATH = OUT_DIR / "e1_summary.md"

DEGRADED_PATHS = {
    "gaussian_medium": COND_DIR / "eval_degraded_gaussian.npy",
    "drift_medium": COND_DIR / "generalization_degraded_drift.npy",
    "jump_medium": COND_DIR / "generalization_degraded_jump.npy",
    "burst_medium": COND_DIR / "generalization_degraded_burst.npy",
    "bias_medium": COND_DIR / "generalization_degraded_bias.npy",
    "combined_medium": COND_DIR / "generalization_degraded_combined.npy",
}

COND_OUTPUT_PATHS = {
    "gaussian_medium": CACHE_DIR / "gaussian_cond_residual_t20_refined.npy",
    "drift_medium": CACHE_DIR / "drift_cond_residual_t20_refined.npy",
    "burst_medium": CACHE_DIR / "burst_cond_residual_t20_refined.npy",
    "bias_medium": CACHE_DIR / "bias_cond_residual_t20_refined.npy",
}

DEGRADATION_ORDER = [
    "gaussian_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
    "bias_medium",
    "combined_medium",
]
NON_FATAL_BIAS = "bias_medium"
FORMAL_REQUIRED_FOR_C1 = set(DEGRADATION_ORDER)
DELTA0_FIXED_VALUES = [0.02, 0.05, 0.10, 0.20]


def ensure_inputs() -> None:
    required = [CLEAN_PATH, STAGE3_GAUSS_SUMMARY_PATH, STAGE3_GENERALIZATION_SUMMARY_PATH]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Stage 3 files:\n" + "\n".join(str(p) for p in missing))


def find_cond_candidates(degradation: str) -> list[Path]:
    prefix = degradation.replace("_medium", "")
    return sorted((STAGE3_DIR).rglob(f"*{prefix}*cond*residual*t20*.npy"))


def load_clean_eval(n_eval: int) -> np.ndarray:
    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    if clean_all.ndim != 3 or clean_all.shape[1:] != (20, 2):
        raise ValueError(f"Unexpected clean trajectory shape: {clean_all.shape}")
    if clean_all.shape[0] < n_eval:
        raise ValueError(f"Need {n_eval} clean trajectories, found {clean_all.shape[0]}")
    return clean_all[:n_eval].astype(np.float32)


def load_stage3_arrays() -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, str]], list[dict[str, str]]]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    missing: list[dict[str, str]] = []
    inventory: list[dict[str, str]] = []

    for degradation in DEGRADATION_ORDER:
        degraded_path = DEGRADED_PATHS[degradation]
        cond_path = COND_OUTPUT_PATHS.get(degradation)
        degraded_exists = degraded_path.is_file()
        cond_exists = bool(cond_path and cond_path.is_file())
        inventory.append(
            {
                "degradation": degradation,
                "degraded_path": str(degraded_path),
                "degraded_exists": str(degraded_exists),
                "cond_path": str(cond_path) if cond_path else "MISSING",
                "cond_exists": str(cond_exists),
            }
        )

        if not degraded_exists or not cond_exists:
            candidates = find_cond_candidates(degradation)
            missing.append(
                {
                    "degradation": degradation,
                    "missing_degraded": str(not degraded_exists),
                    "missing_conditional_output": str(not cond_exists),
                    "candidate_conditional_paths": "; ".join(str(path) for path in candidates) or "NONE",
                }
            )
            continue

        degraded = np.load(degraded_path).astype(np.float32)
        cond_raw = np.load(cond_path).astype(np.float32)
        if cond_raw.ndim == 4:
            cond = cond_raw.mean(axis=0).astype(np.float32)
            cond_shape_type = "seed_stack"
        elif cond_raw.ndim == 3:
            cond = cond_raw.astype(np.float32)
            cond_shape_type = "aggregated_prediction"
        else:
            raise ValueError(f"Unexpected conditional output shape for {degradation}: {cond_raw.shape}")

        clean = load_clean_eval(degraded.shape[0])
        if degraded.shape != clean.shape or cond.shape != clean.shape:
            raise ValueError(
                f"Shape mismatch for {degradation}: clean={clean.shape}, degraded={degraded.shape}, cond={cond.shape}"
            )
        loaded[degradation] = {
            "clean": clean,
            "degraded": degraded,
            "cond": cond,
            "cond_raw": cond_raw,
            "cond_shape_type": np.array(cond_shape_type),
        }

    return loaded, missing, inventory


def load_stage3_metric_rows() -> pd.DataFrame:
    gauss_df = pd.read_csv(STAGE3_GAUSS_SUMMARY_PATH)
    gauss_df = gauss_df.assign(degradation="gaussian_medium")
    gen_df = pd.read_csv(STAGE3_GENERALIZATION_SUMMARY_PATH)
    common_cols = sorted(set(gauss_df.columns).intersection(gen_df.columns))
    return pd.concat([gauss_df[common_cols], gen_df[common_cols]], ignore_index=True)


def get_stage3_ade(metrics_df: pd.DataFrame, degradation: str, method: str) -> tuple[float, float]:
    rows = metrics_df[(metrics_df["degradation"] == degradation) & (metrics_df["method"] == method)]
    if rows.empty:
        raise RuntimeError(f"Missing Stage 3 summary row for {degradation}/{method}")
    row = rows.iloc[0]
    return float(row["ADE_mean"]), float(row.get("ADE_std", np.nan))


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def per_traj_rmse(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    err = frame_error(pred, clean)
    return np.sqrt(np.mean(err**2, axis=1)).astype(np.float32)


def per_traj_acc_rms(pred: np.ndarray) -> np.ndarray:
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    acc_sq = np.sum(acc**2, axis=-1)
    return np.sqrt(np.mean(acc_sq, axis=1)).astype(np.float32)


def stats(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "p25": math.nan,
            "p75": math.nan,
        }
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def confidence_from_delta(degraded: np.ndarray, clean: np.ndarray, delta0: float) -> tuple[np.ndarray, np.ndarray]:
    delta = frame_error(degraded, clean)
    if not np.isfinite(delta0) or delta0 <= 0.0:
        raise ValueError(f"Invalid delta_0: {delta0}")
    confidence = np.exp(-delta / delta0).astype(np.float32)
    return confidence, delta


def apply_e1(degraded: np.ndarray, cond: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    return (degraded + (1.0 - confidence[..., None]) * (cond - degraded)).astype(np.float32)


def automatic_delta0(degraded: np.ndarray, clean: np.ndarray) -> float:
    return float(np.median(frame_error(degraded, clean).reshape(-1)))


def verify_stage3_reproduction(loaded: dict[str, dict[str, np.ndarray]], metrics_df: pd.DataFrame) -> dict[str, float | bool]:
    if "gaussian_medium" not in loaded:
        raise RuntimeError("Stage 3 conditional output mismatch. Do not run formal E1.")
    arrays = loaded["gaussian_medium"]
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    cond = arrays["cond"]
    cond_ade = frame_error(cond, clean).mean(axis=1)
    noisy_ade = frame_error(degraded, clean).mean(axis=1)
    stage3_cond_mean, stage3_cond_std = get_stage3_ade(metrics_df, "gaussian_medium", "cond_residual_t20")
    stage3_noisy_mean, _ = get_stage3_ade(metrics_df, "gaussian_medium", "noisy_input")
    result = {
        "computed_cond_ade_mean": float(cond_ade.mean()),
        "computed_cond_ade_std": float(cond_ade.std()),
        "stage3_cond_ade_mean": stage3_cond_mean,
        "stage3_cond_ade_std": stage3_cond_std,
        "computed_noisy_ade_mean": float(noisy_ade.mean()),
        "stage3_noisy_ade_mean": stage3_noisy_mean,
        "cond_mean_abs_diff": abs(float(cond_ade.mean()) - stage3_cond_mean),
        "cond_std_abs_diff": abs(float(cond_ade.std()) - stage3_cond_std),
        "noisy_mean_abs_diff": abs(float(noisy_ade.mean()) - stage3_noisy_mean),
    }
    passed = (
        result["cond_mean_abs_diff"] < 1e-6
        and result["cond_std_abs_diff"] < 1e-6
        and result["noisy_mean_abs_diff"] < 1e-6
    )
    result["passed"] = bool(passed)
    if not passed:
        raise RuntimeError("Stage 3 conditional output mismatch. Do not run formal E1.")
    return result


def method_arrays(clean: np.ndarray, degraded: np.ndarray, cond: np.ndarray, e1: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "noisy_input": degraded,
        "stage3_cond_residual_t20": cond,
        "e1_oracle_gated_residual": e1,
    }


def build_evaluation_tables(
    loaded: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    full_rows: list[dict] = []
    bin_rows: list[dict] = []
    per_traj_rows: list[dict] = []
    pass_rows: list[dict] = []
    eval_cache: dict[str, dict] = {}

    for degradation, arrays in loaded.items():
        clean = arrays["clean"]
        degraded = arrays["degraded"]
        cond = arrays["cond"]
        delta0 = automatic_delta0(degraded, clean)
        confidence, delta = confidence_from_delta(degraded, clean, delta0)
        e1 = apply_e1(degraded, cond, confidence)
        methods = method_arrays(clean, degraded, cond, e1)
        errors = {name: frame_error(pred, clean) for name, pred in methods.items()}
        ade_traj = {name: err.mean(axis=1) for name, err in errors.items()}
        rmse_traj = {name: per_traj_rmse(pred, clean) for name, pred in methods.items()}
        smooth_traj = {name: per_traj_acc_rms(pred) for name, pred in methods.items()}
        high = confidence > 0.7
        mid = (confidence >= 0.3) & (confidence <= 0.7)
        low = confidence < 0.3
        masks = {"high": high, "mid": mid, "low": low}

        e1_win_vs_cond = float(np.mean(ade_traj["e1_oracle_gated_residual"] <= ade_traj["stage3_cond_residual_t20"]))
        e1_win_vs_noisy = float(np.mean(ade_traj["e1_oracle_gated_residual"] <= ade_traj["noisy_input"]))

        for method, pred in methods.items():
            ade_stats = stats(ade_traj[method])
            rmse_stats = stats(rmse_traj[method])
            smooth_stats = stats(smooth_traj[method])
            full_rows.append(
                {
                    "degradation": degradation,
                    "method": method,
                    "N_trajectories": clean.shape[0],
                    "delta0_auto": delta0,
                    "ADE_mean": ade_stats["mean"],
                    "ADE_std": ade_stats["std"],
                    "ADE_median": ade_stats["median"],
                    "ADE_min": ade_stats["min"],
                    "ADE_max": ade_stats["max"],
                    "ADE_p25": ade_stats["p25"],
                    "ADE_p75": ade_stats["p75"],
                    "RMSE_mean": rmse_stats["mean"],
                    "RMSE_std": rmse_stats["std"],
                    "smooth_acc_rms_mean": smooth_stats["mean"],
                    "smooth_acc_rms_std": smooth_stats["std"],
                    "e1_win_rate_vs_conditional": e1_win_vs_cond,
                    "e1_win_rate_vs_noisy": e1_win_vs_noisy,
                }
            )

        for idx in range(clean.shape[0]):
            per_traj_rows.append(
                {
                    "degradation": degradation,
                    "trajectory_id": idx,
                    "delta0_auto": delta0,
                    "ADE_noisy": float(ade_traj["noisy_input"][idx]),
                    "ADE_cond": float(ade_traj["stage3_cond_residual_t20"][idx]),
                    "ADE_e1": float(ade_traj["e1_oracle_gated_residual"][idx]),
                    "RMSE_noisy": float(rmse_traj["noisy_input"][idx]),
                    "RMSE_cond": float(rmse_traj["stage3_cond_residual_t20"][idx]),
                    "RMSE_e1": float(rmse_traj["e1_oracle_gated_residual"][idx]),
                    "smooth_noisy": float(smooth_traj["noisy_input"][idx]),
                    "smooth_cond": float(smooth_traj["stage3_cond_residual_t20"][idx]),
                    "smooth_e1": float(smooth_traj["e1_oracle_gated_residual"][idx]),
                    "e1_wins_vs_conditional": bool(ade_traj["e1_oracle_gated_residual"][idx] <= ade_traj["stage3_cond_residual_t20"][idx]),
                    "e1_wins_vs_noisy": bool(ade_traj["e1_oracle_gated_residual"][idx] <= ade_traj["noisy_input"][idx]),
                    "confidence_min": float(confidence[idx].min()),
                    "confidence_mean": float(confidence[idx].mean()),
                    "confidence_max": float(confidence[idx].max()),
                    "delta_min": float(delta[idx].min()),
                    "delta_mean": float(delta[idx].mean()),
                    "delta_max": float(delta[idx].max()),
                }
            )

        for bin_name, mask in masks.items():
            for method, err in errors.items():
                values = err[mask]
                s = stats(values)
                bin_rows.append(
                    {
                        "degradation": degradation,
                        "method": method,
                        "confidence_bin": bin_name,
                        "N_frames": int(mask.sum()),
                        "ADE_mean": s["mean"],
                        "ADE_std": s["std"],
                        "ADE_median": s["median"],
                        "ADE_min": s["min"],
                        "ADE_max": s["max"],
                        "ADE_p25": s["p25"],
                        "ADE_p75": s["p75"],
                        "delta0_auto": delta0,
                    }
                )

        high_noisy = float(errors["noisy_input"][high].mean()) if np.any(high) else math.nan
        high_cond = float(errors["stage3_cond_residual_t20"][high].mean()) if np.any(high) else math.nan
        high_e1 = float(errors["e1_oracle_gated_residual"][high].mean()) if np.any(high) else math.nan
        low_noisy = float(errors["noisy_input"][low].mean()) if np.any(low) else math.nan
        low_cond = float(errors["stage3_cond_residual_t20"][low].mean()) if np.any(low) else math.nan
        low_e1 = float(errors["e1_oracle_gated_residual"][low].mean()) if np.any(low) else math.nan
        no_harm_frame_ratio = (
            float(np.mean(errors["e1_oracle_gated_residual"][high] <= 1.10 * errors["noisy_input"][high]))
            if np.any(high)
            else math.nan
        )
        high_traj_ok = []
        for idx in range(clean.shape[0]):
            m = high[idx]
            if np.any(m):
                high_traj_ok.append(
                    bool(
                        errors["e1_oracle_gated_residual"][idx][m].mean()
                        <= 1.10 * errors["noisy_input"][idx][m].mean()
                    )
                )
        no_harm_traj_ratio = float(np.mean(high_traj_ok)) if high_traj_ok else math.nan
        c2_pass = bool(np.isfinite(high_noisy) and high_e1 <= 1.10 * high_noisy)
        low_useful = bool(np.isfinite(low_noisy) and np.isfinite(low_cond) and low_cond < low_noisy)
        if low_useful:
            c3_pass = bool(low_e1 <= 1.10 * low_cond)
            low_note = "conditional useful; E1 close-to-or-better-than conditional" if c3_pass else "conditional useful; E1 erased too much correction"
        else:
            c3_pass = True
            low_note = "no useful conditional correction existed to preserve"

        pass_rows.append(
            {
                "degradation": degradation,
                "is_bias_boundary": degradation == NON_FATAL_BIAS,
                "delta0_auto": delta0,
                "ADE_noisy": float(ade_traj["noisy_input"].mean()),
                "ADE_cond": float(ade_traj["stage3_cond_residual_t20"].mean()),
                "ADE_e1": float(ade_traj["e1_oracle_gated_residual"].mean()),
                "E1_beats_conditional": bool(ade_traj["e1_oracle_gated_residual"].mean() <= ade_traj["stage3_cond_residual_t20"].mean()),
                "E1_beats_noisy": bool(ade_traj["e1_oracle_gated_residual"].mean() <= ade_traj["noisy_input"].mean()),
                "win_rate_vs_conditional": e1_win_vs_cond,
                "win_rate_vs_noisy": e1_win_vs_noisy,
                "N_high_frames": int(high.sum()),
                "ADE_noisy_high": high_noisy,
                "ADE_cond_high": high_cond,
                "ADE_e1_high": high_e1,
                "C2_high_no_harm_pass": c2_pass,
                "no_harm_frame_ratio": no_harm_frame_ratio,
                "no_harm_trajectory_ratio": no_harm_traj_ratio,
                "N_low_frames": int(low.sum()),
                "ADE_noisy_low": low_noisy,
                "ADE_cond_low": low_cond,
                "ADE_e1_low": low_e1,
                "conditional_useful_low": low_useful,
                "C3_low_preservation_pass": c3_pass,
                "low_confidence_note": low_note,
            }
        )

        eval_cache[degradation] = {
            "clean": clean,
            "degraded": degraded,
            "cond": cond,
            "e1": e1,
            "confidence": confidence,
            "delta": delta,
            "errors": errors,
            "ade_traj": ade_traj,
            "delta0": delta0,
        }

    return (
        pd.DataFrame(full_rows),
        pd.DataFrame(bin_rows),
        pd.DataFrame(per_traj_rows),
        pd.DataFrame(pass_rows),
        eval_cache,
    )


def build_delta0_sweep(loaded: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict] = []
    for degradation, arrays in loaded.items():
        clean = arrays["clean"]
        degraded = arrays["degraded"]
        cond = arrays["cond"]
        auto = automatic_delta0(degraded, clean)
        sweep = [(f"fixed_{value:.2f}", value) for value in DELTA0_FIXED_VALUES] + [("auto_median_degraded_error", auto)]
        noisy_ade = frame_error(degraded, clean).mean(axis=1)
        cond_ade = frame_error(cond, clean).mean(axis=1)
        for label, delta0 in sweep:
            confidence, _ = confidence_from_delta(degraded, clean, delta0)
            e1 = apply_e1(degraded, cond, confidence)
            e1_err = frame_error(e1, clean)
            e1_ade = e1_err.mean(axis=1)
            high = confidence > 0.7
            mid = (confidence >= 0.3) & (confidence <= 0.7)
            low = confidence < 0.3
            rows.append(
                {
                    "degradation": degradation,
                    "delta0_label": label,
                    "delta0": delta0,
                    "ADE_noisy": float(noisy_ade.mean()),
                    "ADE_cond": float(cond_ade.mean()),
                    "ADE_e1": float(e1_ade.mean()),
                    "ADE_e1_high": float(e1_err[high].mean()) if np.any(high) else math.nan,
                    "ADE_e1_low": float(e1_err[low].mean()) if np.any(low) else math.nan,
                    "win_rate_vs_conditional": float(np.mean(e1_ade <= cond_ade)),
                    "N_high_frames": int(high.sum()),
                    "N_mid_frames": int(mid.sum()),
                    "N_low_frames": int(low.sum()),
                }
            )
    return pd.DataFrame(rows)


def choose_case(eval_cache: dict[str, dict], kind: str) -> tuple[str, int] | None:
    best: tuple[float, str, int] | None = None
    for degradation, data in eval_cache.items():
        errors = data["errors"]
        confidence = data["confidence"]
        if kind == "high_overcorrection":
            mask = confidence > 0.7
            for idx in range(confidence.shape[0]):
                if np.any(mask[idx]):
                    score = float((errors["stage3_cond_residual_t20"][idx][mask[idx]] - errors["noisy_input"][idx][mask[idx]]).mean())
                    if best is None or score > best[0]:
                        best = (score, degradation, idx)
        elif kind == "low_preserved":
            mask = confidence < 0.3
            for idx in range(confidence.shape[0]):
                if np.any(mask[idx]):
                    noisy = float(errors["noisy_input"][idx][mask[idx]].mean())
                    cond = float(errors["stage3_cond_residual_t20"][idx][mask[idx]].mean())
                    e1 = float(errors["e1_oracle_gated_residual"][idx][mask[idx]].mean())
                    if cond < noisy and e1 <= 1.10 * cond:
                        score = noisy - e1
                        if best is None or score > best[0]:
                            best = (score, degradation, idx)
        elif kind == "failure":
            ade = data["ade_traj"]
            for idx in range(confidence.shape[0]):
                score = float(ade["e1_oracle_gated_residual"][idx] - min(ade["noisy_input"][idx], ade["stage3_cond_residual_t20"][idx]))
                if best is None or score > best[0]:
                    best = (score, degradation, idx)
    if best is None:
        return None
    return best[1], best[2]


def plot_case(eval_cache: dict[str, dict], degradation: str, idx: int, title: str, path: Path) -> None:
    data = eval_cache[degradation]
    clean = data["clean"][idx]
    degraded = data["degraded"][idx]
    cond = data["cond"][idx]
    e1 = data["e1"][idx]
    confidence = data["confidence"][idx]
    ade_noisy = data["ade_traj"]["noisy_input"][idx]
    ade_cond = data["ade_traj"]["stage3_cond_residual_t20"][idx]
    ade_e1 = data["ade_traj"]["e1_oracle_gated_residual"][idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(clean[:, 0], clean[:, 1], "k-o", ms=3, lw=1.4, label="clean target")
    ax.plot(degraded[:, 0], degraded[:, 1], "C1-o", ms=3, lw=1.2, label="degraded input")
    ax.plot(cond[:, 0], cond[:, 1], "C3-o", ms=3, lw=1.2, label="Stage 3 cond_residual_t20")
    ax.plot(e1[:, 0], e1[:, 1], "C0-o", ms=3, lw=1.4, label="E1 oracle gated")
    ax.set_title(f"{title}\n{degradation}, trajectory {idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.02,
        f"ADE noisy={ade_noisy:.4f}\nADE cond={ade_cond:.4f}\nADE E1={ade_e1:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    axc = axes[1]
    axc.plot(np.arange(confidence.shape[0]), confidence, "C0-o", ms=3, lw=1.3)
    axc.axhline(0.7, color="0.35", ls="--", lw=1)
    axc.axhline(0.3, color="0.35", ls=":", lw=1)
    axc.set_ylim(0.0, 1.05)
    axc.set_title("oracle confidence c_t")
    axc.set_xlabel("t")
    axc.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_figures(eval_cache: dict[str, dict]) -> list[dict[str, str]]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_rows: list[dict[str, str]] = []
    requests = [
        ("high_confidence_overcorrection", choose_case(eval_cache, "high_overcorrection"), "High-confidence over-correction case"),
        ("low_confidence_preserved", choose_case(eval_cache, "low_preserved"), "Low-confidence correction-preserved case"),
        ("drift_case", ("drift_medium", int(np.argmax(eval_cache["drift_medium"]["ade_traj"]["stage3_cond_residual_t20"] - eval_cache["drift_medium"]["ade_traj"]["e1_oracle_gated_residual"]))) if "drift_medium" in eval_cache else None, "Drift case"),
        ("burst_case", ("burst_medium", int(np.argmax(eval_cache["burst_medium"]["ade_traj"]["stage3_cond_residual_t20"] - eval_cache["burst_medium"]["ade_traj"]["e1_oracle_gated_residual"]))) if "burst_medium" in eval_cache else None, "Burst case"),
        ("combined_case", ("combined_medium", 0) if "combined_medium" in eval_cache else None, "Combined case"),
        ("failure_or_ambiguous_case", choose_case(eval_cache, "failure"), "Failure or ambiguous case"),
    ]
    for stem, case, title in requests:
        if case is None:
            figure_rows.append({"figure": stem, "status": "MISSING", "path": "", "note": "required arrays unavailable"})
            continue
        degradation, idx = case
        path = FIG_DIR / f"{stem}_{degradation}_traj{idx}.png"
        plot_case(eval_cache, degradation, idx, title, path)
        figure_rows.append({"figure": stem, "status": "FOUND", "path": str(path), "note": f"{degradation}, trajectory {idx}"})
    return figure_rows


def assess_pass_fail(pass_df: pd.DataFrame, missing: list[dict[str, str]]) -> dict[str, object]:
    evaluated = set(pass_df["degradation"].tolist())
    missing_required = sorted(FORMAL_REQUIRED_FOR_C1 - evaluated)
    wins = pass_df[pass_df["E1_beats_conditional"] == True]["degradation"].tolist()
    c1_pass = len([deg for deg in DEGRADATION_ORDER if deg in wins]) >= 4 and not missing_required

    target_df = pass_df[pass_df["degradation"] != NON_FATAL_BIAS].copy()
    c2_failures = target_df[target_df["C2_high_no_harm_pass"] != True]["degradation"].tolist()
    c3_failures = target_df[target_df["C3_low_preservation_pass"] != True]["degradation"].tolist()
    c2_pass = len(c2_failures) == 0 and not target_df.empty
    c3_pass = len(c3_failures) == 0 and not target_df.empty
    overall_pass = bool(c1_pass and c2_pass and c3_pass)
    return {
        "evaluated_conditions": sorted(evaluated, key=DEGRADATION_ORDER.index),
        "missing_required_conditions": missing_required,
        "e1_beats_conditional_conditions": wins,
        "C1_pass": bool(c1_pass),
        "C1_note": (
            f"E1 beats conditional in {len(wins)}/{len(DEGRADATION_ORDER)} formal conditions; missing required per-frame outputs: {missing_required}"
        ),
        "C2_pass": bool(c2_pass),
        "C2_failures": c2_failures,
        "C3_pass": bool(c3_pass),
        "C3_failures": c3_failures,
        "overall_pass": overall_pass,
        "overall_label": "PASS" if overall_pass else "NO-PASS",
        "missing_artifacts": missing,
    }


def write_pass_fail_csv(pass_df: pd.DataFrame, pass_fail: dict[str, object]) -> pd.DataFrame:
    rows = pass_df.copy()
    missing_rows = []
    for missing in pass_fail["missing_artifacts"]:
        missing_rows.append(
            {
                "degradation": missing["degradation"],
                "is_bias_boundary": missing["degradation"] == NON_FATAL_BIAS,
                "status": "MISSING_PER_FRAME_COND_OUTPUT",
                "missing_degraded": missing["missing_degraded"],
                "missing_conditional_output": missing["missing_conditional_output"],
                "candidate_conditional_paths": missing["candidate_conditional_paths"],
            }
        )
    rows["status"] = "EVALUATED"
    rows["missing_degraded"] = ""
    rows["missing_conditional_output"] = ""
    rows["candidate_conditional_paths"] = ""
    if missing_rows:
        rows = pd.concat([rows, pd.DataFrame(missing_rows)], ignore_index=True, sort=False)
    rows["C1_global_pass"] = bool(pass_fail["C1_pass"])
    rows["C2_global_pass"] = bool(pass_fail["C2_pass"])
    rows["C3_global_pass"] = bool(pass_fail["C3_pass"])
    rows["overall_PASS"] = bool(pass_fail["overall_pass"])
    rows["overall_label"] = pass_fail["overall_label"]
    rows.to_csv(PASS_FAIL_PATH, index=False)
    return rows


def fmt(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "NaN"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    for col in table.columns:
        table[col] = table[col].map(
            lambda value: fmt(float(value), 6)
            if isinstance(value, (float, np.floating))
            else str(value)
        )
    headers = [str(col) for col in table.columns]
    rows = table.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(
    loaded: dict[str, dict[str, np.ndarray]],
    missing: list[dict[str, str]],
    inventory: list[dict[str, str]],
    reproduction: dict[str, float | bool],
    full_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    pass_fail: dict[str, object],
    figure_rows: list[dict[str, str]],
) -> None:
    full_pivot = full_df.pivot(index="degradation", columns="method", values="ADE_mean")
    lines: list[str] = [
        "# Stage 4 E1 Full Experiment: Oracle Confidence Residual Gating",
        "",
        "## Objective and hypothesis",
        "E1 tests whether oracle per-frame confidence can control the Stage 3 conditional residual correction and reduce over-refinement. It does not redesign the model, retrain, use SDEdit, or sample a new conditional output.",
        "",
        "## Data sources",
        f"- clean trajectories: `{CLEAN_PATH}`",
        f"- Stage 3 gaussian summary: `{STAGE3_GAUSS_SUMMARY_PATH}`",
        f"- Stage 3 generalization summary: `{STAGE3_GENERALIZATION_SUMMARY_PATH}`",
        "",
        "Loaded array inventory:",
    ]
    for row in inventory:
        lines.append(
            f"- {row['degradation']}: degraded_exists={row['degraded_exists']} `{row['degraded_path']}`; "
            f"cond_exists={row['cond_exists']} `{row['cond_path']}`"
        )
    if missing:
        lines.extend(["", "Missing per-frame inputs:"])
        for row in missing:
            lines.append(
                f"- {row['degradation']}: missing_conditional_output={row['missing_conditional_output']}; "
                f"candidates={row['candidate_conditional_paths']}"
            )

    lines.extend(["", "Loaded shapes:"])
    for degradation, arrays in loaded.items():
        lines.append(
            f"- {degradation}: clean={tuple(arrays['clean'].shape)}, degraded={tuple(arrays['degraded'].shape)}, "
            f"cond={tuple(arrays['cond'].shape)}"
        )

    lines.extend(
        [
            "",
            "## Stage 3 ADE reproduction check",
            f"- recomputed gaussian cond_residual_t20 ADE mean: {fmt(float(reproduction['computed_cond_ade_mean']))}",
            f"- official Stage 3 gaussian cond_residual_t20 ADE mean: {fmt(float(reproduction['stage3_cond_ade_mean']))}",
            f"- recomputed gaussian cond_residual_t20 ADE std: {fmt(float(reproduction['computed_cond_ade_std']))}",
            f"- official Stage 3 gaussian cond_residual_t20 ADE std: {fmt(float(reproduction['stage3_cond_ade_std']))}",
            f"- reproduction passed: {str(reproduction['passed']).upper()}",
            "",
            "## Method formula",
            "- r_hat = x_cond - y",
            "- c_t = exp(- ||y_t - x*_t|| / delta_0)",
            "- lambda_t = 1 - c_t",
            "- x_E1,t = y_t + lambda_t * r_hat_t",
            "",
            "## Delta_0 setting",
            "Main E1 metrics use an automatic per-degradation delta_0 equal to the median degraded-input frame error over the full evaluation set. Fixed values [0.02, 0.05, 0.10, 0.20] are reported separately in the sensitivity sweep.",
            "",
            "## Full method x degradation metric table",
            markdown_table(full_pivot.reset_index()),
            "",
            "## Confidence-bin analysis",
        ]
    )
    bin_compact = bin_df[["degradation", "method", "confidence_bin", "N_frames", "ADE_mean", "ADE_std", "ADE_median"]]
    lines.append(markdown_table(bin_compact))

    lines.extend(["", "## High-confidence no-harm analysis"])
    high_cols = [
        "degradation",
        "N_high_frames",
        "ADE_noisy_high",
        "ADE_cond_high",
        "ADE_e1_high",
        "C2_high_no_harm_pass",
        "no_harm_frame_ratio",
    ]
    lines.append(markdown_table(pass_df[high_cols]))

    lines.extend(["", "## Low-confidence correction-preservation analysis"])
    low_cols = [
        "degradation",
        "N_low_frames",
        "ADE_noisy_low",
        "ADE_cond_low",
        "ADE_e1_low",
        "conditional_useful_low",
        "C3_low_preservation_pass",
        "low_confidence_note",
    ]
    lines.append(markdown_table(pass_df[low_cols]))

    lines.extend(["", "## Delta_0 sensitivity"])
    lines.append(
        markdown_table(
            sweep_df[
                [
                    "degradation",
                    "delta0_label",
                    "delta0",
                    "ADE_noisy",
                    "ADE_cond",
                    "ADE_e1",
                    "ADE_e1_high",
                    "ADE_e1_low",
                    "win_rate_vs_conditional",
                    "N_high_frames",
                    "N_mid_frames",
                    "N_low_frames",
                ]
            ]
        )
    )

    lines.extend(["", "## Representative figures"])
    for row in figure_rows:
        lines.append(f"- {row['figure']}: {row['status']} {row['path']} {row['note']}")

    lines.extend(
        [
            "",
            "## PASS / NO-PASS judgment",
            f"- C1: {str(pass_fail['C1_pass']).upper()}",
            f"- C1 note: {pass_fail['C1_note']}",
            f"- C2: {str(pass_fail['C2_pass']).upper()}",
            f"- C2 failures: {pass_fail['C2_failures']}",
            f"- C3: {str(pass_fail['C3_pass']).upper()}",
            f"- C3 failures: {pass_fail['C3_failures']}",
            f"- Overall: {pass_fail['overall_label']}",
            "",
            "Bias boundary: bias is not the main E1 target. Bias failures are not fatal for E1 and motivate E2 absolute-space likelihood.",
            "",
            "Boundary statement: E1 tests confidence control of residual correction. E2 is needed for absolute-space bias anchoring.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    loaded, missing, inventory = load_stage3_arrays()
    if missing:
        print("Missing per-frame Stage 3 conditional outputs detected:")
        for row in missing:
            print(f"- {row['degradation']}: candidates={row['candidate_conditional_paths']}")
    if not loaded:
        raise RuntimeError("No usable Stage 3 conditional per-frame outputs found.")

    metrics_df = load_stage3_metric_rows()
    reproduction = verify_stage3_reproduction(loaded, metrics_df)
    full_df, bin_df, per_traj_df, pass_df, eval_cache = build_evaluation_tables(loaded)
    sweep_df = build_delta0_sweep(loaded)
    pass_fail = assess_pass_fail(pass_df, missing)
    pass_df_out = write_pass_fail_csv(pass_df, pass_fail)
    figure_rows = make_figures(eval_cache)

    full_df.to_csv(FULL_METRICS_PATH, index=False)
    bin_df.to_csv(BIN_METRICS_PATH, index=False)
    sweep_df.to_csv(DELTA_SWEEP_PATH, index=False)
    per_traj_df.to_csv(PER_TRAJ_PATH, index=False)
    write_summary(
        loaded=loaded,
        missing=missing,
        inventory=inventory,
        reproduction=reproduction,
        full_df=full_df,
        bin_df=bin_df,
        sweep_df=sweep_df,
        pass_df=pass_df_out,
        pass_fail=pass_fail,
        figure_rows=figure_rows,
    )

    print("STAGE4_E1_ORACLE_GATING_FULL_COMPLETE")
    print(f"Stage 3 ADE reproduction check passed: {reproduction['passed']}")
    print(f"number of degradation conditions evaluated: {len(loaded)}")
    print(f"C1: {pass_fail['C1_pass']}")
    print(f"C2: {pass_fail['C2_pass']}")
    print(f"C3: {pass_fail['C3_pass']}")
    print(f"overall PASS/NO-PASS: {pass_fail['overall_label']}")
    print(f"outputs: {OUT_DIR}")
    print(f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
