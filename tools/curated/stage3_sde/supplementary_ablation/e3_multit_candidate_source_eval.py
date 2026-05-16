from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3_multit_candidate_source_eval_mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
T_STARTS = [1, 2, 3]
VARIANTS = ["V1_single_t1", "V2_single_t2", "V3_single_t3", "V4_multit_confidence_bin"]
EXPECTED_SHAPE = (1000, 20, 2)
EXPECTED_CONF_SHAPE = (1000, 20)
TAU_HIGH = 0.7
GAMMA = 2.0
EPS = 1e-12
BASELINE_ANCHOR_ADE = 0.136567

DATASET_PATH = PROJECT_ROOT / "data" / "stage4" / "e3_holdout_1000" / "clean_trajs.npy"
CACHE_ROOT = PROJECT_ROOT / "outputs" / "stage4" / "e3_holdout1000_confidence_aware_sdedit"
ARRAY_ROOT = CACHE_ROOT / "arrays"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_multit_candidate_source_ablation"
FIG_DIR = OUT_DIR / "figures"
INVENTORY_CSV = OUT_DIR / "cache_inventory.csv"

FULL_METRICS_PATH = OUT_DIR / "e3_multit_candidate_source_full_metrics.csv"
CONDITION_SUMMARY_PATH = OUT_DIR / "e3_multit_candidate_source_condition_summary.csv"
BIN_METRICS_PATH = OUT_DIR / "e3_multit_candidate_source_bin_metrics.csv"
PASS_FAIL_PATH = OUT_DIR / "e3_multit_candidate_source_pass_fail_summary.csv"
DECISION_MD_PATH = OUT_DIR / "e3_multit_candidate_source_decision_summary.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_npy(path: Path, expected_shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.shape != expected_shape:
        raise ValueError(f"{label}: expected shape {expected_shape}, got {arr.shape} at {path}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{label}: non-numeric dtype {arr.dtype} at {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label}: non-finite values at {path}")
    return arr.astype(np.float32, copy=False)


def locate_unique_companion(condition: str, kind: str) -> Path:
    if kind == "degraded":
        patterns = [f"{condition}_degraded.npy", f"{condition}_noisy.npy"]
        keywords = ["degraded", "noisy"]
    elif kind == "confidence":
        patterns = [f"{condition}_confidence.npy"]
        keywords = ["confidence"]
    else:
        raise ValueError(f"Unsupported companion kind: {kind}")

    exact_matches = [ARRAY_ROOT / pattern for pattern in patterns if (ARRAY_ROOT / pattern).is_file()]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Ambiguous {kind} companion for {condition}: "
            + "; ".join(rel(path) for path in exact_matches)
        )

    matches: list[Path] = []
    for path in CACHE_ROOT.rglob("*.npy"):
        name = path.name.lower()
        if condition.lower() not in str(path).lower():
            continue
        if any(keyword in name for keyword in keywords):
            if "fused" in name or "sdedit" in name:
                continue
            matches.append(path)
    matches = sorted(set(matches))
    if len(matches) != 1:
        status = "missing" if not matches else "ambiguous"
        raise RuntimeError(
            f"{status.upper()}_{kind.upper()}_COMPANION for {condition}: "
            + ("none" if not matches else "; ".join(rel(path) for path in matches))
        )
    return matches[0]


def load_candidate_paths_from_inventory() -> dict[tuple[str, int], Path]:
    if not INVENTORY_CSV.is_file():
        raise FileNotFoundError(f"Missing inventory CSV: {INVENTORY_CSV}")
    df = pd.read_csv(INVENTORY_CSV)
    paths: dict[tuple[str, int], Path] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for condition in CONDITIONS:
        for t_start in T_STARTS:
            row = df[(df["condition"] == condition) & (df["t_start"] == t_start)]
            if len(row) != 1:
                missing.append(f"{condition} t{t_start}: row_count={len(row)}")
                continue
            item = row.iloc[0]
            if not bool(item["found"]) or bool(item["ambiguous"]):
                invalid.append(f"{condition} t{t_start}: found={item['found']} ambiguous={item['ambiguous']}")
                continue
            if not bool(item["shape_ok"]) or not bool(item["finite_ok"]):
                invalid.append(f"{condition} t{t_start}: shape_ok={item['shape_ok']} finite_ok={item['finite_ok']}")
                continue
            path = PROJECT_ROOT / str(item["path"])
            if not path.is_file():
                invalid.append(f"{condition} t{t_start}: file missing at {path}")
                continue
            paths[(condition, t_start)] = path
    if missing or invalid:
        raise RuntimeError(
            "Inventory is not ready for evaluation.\n"
            f"Missing rows: {missing or 'none'}\n"
            f"Invalid rows: {invalid or 'none'}"
        )
    return paths


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def per_traj_ade(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return frame_error(pred, clean).mean(axis=1)


def per_traj_rmse(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    err = frame_error(pred, clean)
    return np.sqrt(np.mean(err**2, axis=1)).astype(np.float32)


def acceleration_rms(traj: np.ndarray) -> np.ndarray:
    acc = traj[:, 2:, :] - 2.0 * traj[:, 1:-1, :] + traj[:, :-2, :]
    acc_norm = np.linalg.norm(acc, axis=-1)
    return np.sqrt(np.mean(acc_norm**2, axis=1)).astype(np.float32)


def confidence_masks(confidence: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "high": confidence >= 0.7,
        "mid_high": (confidence >= 0.5) & (confidence < 0.7),
        "mid_low": (confidence >= 0.25) & (confidence < 0.5),
        "low": confidence < 0.25,
    }


def masked_mean_error(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    err = frame_error(pred, clean)
    return float(np.mean(err[mask])) if int(mask.sum()) else math.nan


def fuse_from_source(degraded: np.ndarray, source: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    lam = np.power(1.0 - confidence, GAMMA).astype(np.float32)
    lam = np.where(confidence >= TAU_HIGH, 0.0, lam).astype(np.float32)
    fused = degraded + lam[..., None] * (source - degraded)
    return fused.astype(np.float32)


def multit_source(
    degraded: np.ndarray,
    confidence: np.ndarray,
    sdedit_t1: np.ndarray,
    sdedit_t2: np.ndarray,
    sdedit_t3: np.ndarray,
) -> np.ndarray:
    source = np.empty_like(degraded, dtype=np.float32)
    high = confidence >= 0.7
    mid_high = (confidence >= 0.5) & (confidence < 0.7)
    mid_low = (confidence >= 0.25) & (confidence < 0.5)
    low = confidence < 0.25
    source[high] = degraded[high]
    source[mid_high] = sdedit_t1[mid_high]
    source[mid_low] = sdedit_t2[mid_low]
    source[low] = sdedit_t3[low]
    return source


def status_text(value: object) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "UNINTERPRETABLE"


def method_metrics(
    condition: str,
    variant: str,
    pred: np.ndarray,
    source: np.ndarray,
    clean: np.ndarray,
    degraded: np.ndarray,
    confidence: np.ndarray,
    clean_acc_mean: float,
    ade_t1_condition: float | None,
) -> dict:
    ade_traj = per_traj_ade(pred, clean)
    rmse_traj = per_traj_rmse(pred, clean)
    acc_traj = acceleration_rms(pred)
    masks = confidence_masks(confidence)

    high_y = masked_mean_error(degraded, clean, masks["high"])
    high_fused = masked_mean_error(pred, clean, masks["high"])
    low_y = masked_mean_error(degraded, clean, masks["low"])
    low_fused = masked_mean_error(pred, clean, masks["low"])

    if math.isnan(high_y) or high_y <= 1e-12:
        high_ratio = math.nan
        high_pass: bool | str = "UNINTERPRETABLE"
    else:
        high_ratio = high_fused / high_y
        high_pass = bool(high_fused <= 1.05 * high_y)

    if math.isnan(low_y) or math.isnan(low_fused):
        low_improvement = math.nan
        low_pass: bool | str = "UNINTERPRETABLE"
    else:
        low_improvement = low_y - low_fused
        low_pass = bool(low_fused < low_y)

    source_motion = float(np.mean(np.linalg.norm(source - degraded, axis=-1)))
    fused_motion = float(np.mean(np.linalg.norm(pred - degraded, axis=-1)))
    motion_usage = fused_motion / (source_motion + EPS)
    ade_mean = float(np.mean(ade_traj))
    improvement_vs_t1 = math.nan if ade_t1_condition is None else float(ade_t1_condition - ade_mean)
    rel_improvement_vs_t1 = (
        math.nan
        if ade_t1_condition is None or abs(ade_t1_condition) <= EPS
        else float((ade_t1_condition - ade_mean) / ade_t1_condition)
    )

    return {
        "condition": condition,
        "variant": variant,
        "N": int(pred.shape[0]),
        "tau_high": TAU_HIGH,
        "gamma": GAMMA,
        "ADE": ade_mean,
        "RMSE": float(np.mean(rmse_traj)),
        "per_traj_ADE_mean": ade_mean,
        "per_traj_ADE_std": float(np.std(ade_traj)),
        "per_traj_ADE_median": float(np.median(ade_traj)),
        "per_traj_ADE_min": float(np.min(ade_traj)),
        "per_traj_ADE_max": float(np.max(ade_traj)),
        "per_traj_ADE_p10": float(np.percentile(ade_traj, 10)),
        "per_traj_ADE_p90": float(np.percentile(ade_traj, 90)),
        "high_conf_ADE_y": high_y,
        "high_conf_ADE_fused": high_fused,
        "high_conf_ADE_ratio": high_ratio,
        "high_conf_no_harm_pass": status_text(high_pass),
        "low_conf_ADE_y": low_y,
        "low_conf_ADE_fused": low_fused,
        "low_conf_improvement": low_improvement,
        "low_conf_pass": status_text(low_pass),
        "acceleration_RMS_clean": clean_acc_mean,
        "acceleration_RMS_fused": float(np.mean(acc_traj)),
        "acceleration_RMS_ratio": float(np.mean(acc_traj) / (clean_acc_mean + EPS)),
        "motion_usage_ratio": motion_usage,
        "improvement_vs_E3_t1": improvement_vs_t1,
        "relative_improvement_vs_E3_t1": rel_improvement_vs_t1,
    }


def bin_metric_rows(
    condition: str,
    variant: str,
    pred: np.ndarray,
    clean: np.ndarray,
    degraded: np.ndarray,
    confidence: np.ndarray,
) -> list[dict]:
    masks = confidence_masks(confidence)
    err_fused = frame_error(pred, clean)
    err_y = frame_error(degraded, clean)
    total = confidence.size
    rows = []
    for bin_name, mask in masks.items():
        fused_vals = err_fused[mask]
        y_vals = err_y[mask]
        rows.append(
            {
                "condition": condition,
                "variant": variant,
                "confidence_bin": bin_name,
                "N_frames": int(mask.sum()),
                "frame_fraction": float(mask.sum() / total),
                "ADE_y": float(np.mean(y_vals)) if y_vals.size else math.nan,
                "ADE_fused": float(np.mean(fused_vals)) if fused_vals.size else math.nan,
                "ADE_improvement_y_minus_fused": (
                    float(np.mean(y_vals) - np.mean(fused_vals)) if y_vals.size else math.nan
                ),
                "ADE_fused_std": float(np.std(fused_vals)) if fused_vals.size else math.nan,
                "ADE_fused_median": float(np.median(fused_vals)) if fused_vals.size else math.nan,
                "ADE_fused_min": float(np.min(fused_vals)) if fused_vals.size else math.nan,
                "ADE_fused_max": float(np.max(fused_vals)) if fused_vals.size else math.nan,
                "ADE_fused_p10": float(np.percentile(fused_vals, 10)) if fused_vals.size else math.nan,
                "ADE_fused_p90": float(np.percentile(fused_vals, 90)) if fused_vals.size else math.nan,
            }
        )
    return rows


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 6) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float) or isinstance(value, np.floating):
                vals.append("nan" if math.isnan(float(value)) else f"{float(value):.{float_digits}f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def plot_representative(
    condition: str,
    idx: int,
    clean: np.ndarray,
    degraded: np.ndarray,
    confidence: np.ndarray,
    preds: dict[str, np.ndarray],
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {
        "clean": "black",
        "degraded": "#7f8c8d",
        "V1_single_t1": "#1f77b4",
        "V2_single_t2": "#ff7f0e",
        "V3_single_t3": "#2ca02c",
        "V4_multit_confidence_bin": "#d62728",
    }
    labels = {
        "V1_single_t1": "E3-t1 fused",
        "V2_single_t2": "E3-t2 fused",
        "V3_single_t3": "E3-t3 fused",
        "V4_multit_confidence_bin": "E3-multit-bin fused",
    }
    ax = axes[0]
    ax.plot(clean[idx, :, 0], clean[idx, :, 1], "-o", color=colors["clean"], linewidth=2, markersize=3, label="clean")
    ax.plot(
        degraded[idx, :, 0],
        degraded[idx, :, 1],
        "-o",
        color=colors["degraded"],
        linewidth=1.5,
        markersize=3,
        label="degraded",
    )
    for variant, pred in preds.items():
        ax.plot(
            pred[idx, :, 0],
            pred[idx, :, 1],
            "-o",
            linewidth=1.2,
            markersize=2,
            color=colors[variant],
            label=labels[variant],
        )
    ax.set_title(f"{condition} trajectory {idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    axes[1].plot(confidence[idx], color="#34495e", linewidth=2)
    axes[1].axhline(0.7, color="#e74c3c", linestyle="--", linewidth=1)
    axes[1].axhline(0.5, color="#f39c12", linestyle="--", linewidth=1)
    axes[1].axhline(0.25, color="#3498db", linestyle="--", linewidth=1)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("confidence")
    axes[1].set_xlabel("frame")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(frame_error(degraded[idx : idx + 1], clean[idx : idx + 1])[0], color=colors["degraded"], label="degraded")
    for variant, pred in preds.items():
        axes[2].plot(
            frame_error(pred[idx : idx + 1], clean[idx : idx + 1])[0],
            color=colors[variant],
            label=labels[variant],
        )
    axes[2].set_title("per-frame error")
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("ADE")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    out_path = FIG_DIR / f"{condition}_median_t1_candidate_source_comparison.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def make_condition_summary(full_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        sub = full_df[full_df["condition"] == condition].set_index("variant")
        ade_y = float(sub["high_conf_ADE_y"].iloc[0])  # compact table keeps noisy high in pass columns below too
        best_variant = str(sub["ADE"].idxmin())
        row = {
            "condition": condition,
            "ADE_y": math.nan,
            "ADE_t1": float(sub.loc["V1_single_t1", "ADE"]),
            "ADE_t2": float(sub.loc["V2_single_t2", "ADE"]),
            "ADE_t3": float(sub.loc["V3_single_t3", "ADE"]),
            "ADE_multit_bin": float(sub.loc["V4_multit_confidence_bin", "ADE"]),
            "best_variant": best_variant,
            "high_conf_pass_t1": sub.loc["V1_single_t1", "high_conf_no_harm_pass"],
            "high_conf_pass_t2": sub.loc["V2_single_t2", "high_conf_no_harm_pass"],
            "high_conf_pass_t3": sub.loc["V3_single_t3", "high_conf_no_harm_pass"],
            "high_conf_pass_multit_bin": sub.loc["V4_multit_confidence_bin", "high_conf_no_harm_pass"],
            "low_conf_pass_t1": sub.loc["V1_single_t1", "low_conf_pass"],
            "low_conf_pass_t2": sub.loc["V2_single_t2", "low_conf_pass"],
            "low_conf_pass_t3": sub.loc["V3_single_t3", "low_conf_pass"],
            "low_conf_pass_multit_bin": sub.loc["V4_multit_confidence_bin", "low_conf_pass"],
            "motion_usage_ratio_t1": float(sub.loc["V1_single_t1", "motion_usage_ratio"]),
            "motion_usage_ratio_t2": float(sub.loc["V2_single_t2", "motion_usage_ratio"]),
            "motion_usage_ratio_t3": float(sub.loc["V3_single_t3", "motion_usage_ratio"]),
            "motion_usage_ratio_multit_bin": float(sub.loc["V4_multit_confidence_bin", "motion_usage_ratio"]),
            "high_conf_ADE_y": ade_y,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_decision(full_df: pd.DataFrame, baseline_status: str, baseline_ade: float) -> tuple[pd.DataFrame, str, str]:
    mean_ade = full_df.groupby("variant")["ADE"].mean().to_dict()
    mean_motion = full_df.groupby("variant")["motion_usage_ratio"].mean().to_dict()
    t1_by_condition = full_df[full_df["variant"] == "V1_single_t1"].set_index("condition")["ADE"]

    pass_rows: list[dict] = []
    case_a_candidates: list[str] = []

    for variant in ["V2_single_t2", "V3_single_t3"]:
        sub = full_df[full_df["variant"] == variant].copy()
        lower_than_t1 = int(sum(float(row.ADE) < float(t1_by_condition.loc[row.condition]) for row in sub.itertuples()))
        high_pass = int((sub["high_conf_no_harm_pass"] == "PASS").sum())
        interpretable_high = int((sub["high_conf_no_harm_pass"] != "UNINTERPRETABLE").sum())
        motion_in_range = bool(0.2 <= mean_motion[variant] <= 0.7)
        mean_lower = bool(mean_ade[variant] < mean_ade["V1_single_t1"])
        passed = bool(mean_lower and lower_than_t1 >= 4 and high_pass >= 5 and motion_in_range)
        if passed:
            case_a_candidates.append(variant)
        pass_rows.append(
            {
                "case": "Case A",
                "variant": variant,
                "six_condition_mean_ADE": mean_ade[variant],
                "six_condition_mean_ADE_t1": mean_ade["V1_single_t1"],
                "mean_ADE_lower_than_t1": mean_lower,
                "conditions_lower_than_t1": lower_than_t1,
                "high_conf_no_harm_pass_count": high_pass,
                "high_conf_interpretable_count": interpretable_high,
                "low_conf_pass_count": int((sub["low_conf_pass"] == "PASS").sum()),
                "mean_motion_usage_ratio": mean_motion[variant],
                "motion_usage_in_range_0p2_0p7": motion_in_range,
                "passed": passed,
            }
        )

    sub_multi = full_df[full_df["variant"] == "V4_multit_confidence_bin"].copy()
    lower_than_t1_multi = int(sum(float(row.ADE) < float(t1_by_condition.loc[row.condition]) for row in sub_multi.itertuples()))
    high_pass_multi = int((sub_multi["high_conf_no_harm_pass"] == "PASS").sum())
    interpretable_high_multi = int((sub_multi["high_conf_no_harm_pass"] != "UNINTERPRETABLE").sum())
    low_pass_multi = int((sub_multi["low_conf_pass"] == "PASS").sum())
    motion_multi = mean_motion["V4_multit_confidence_bin"]
    multi_lower_all_single = bool(
        mean_ade["V4_multit_confidence_bin"]
        < min(mean_ade["V1_single_t1"], mean_ade["V2_single_t2"], mean_ade["V3_single_t3"])
    )
    multi_passed = bool(
        multi_lower_all_single
        and lower_than_t1_multi >= 4
        and high_pass_multi >= 5
        and low_pass_multi >= 4
        and 0.2 <= motion_multi <= 0.7
    )
    pass_rows.append(
        {
            "case": "Case B",
            "variant": "V4_multit_confidence_bin",
            "six_condition_mean_ADE": mean_ade["V4_multit_confidence_bin"],
            "six_condition_mean_ADE_t1": mean_ade["V1_single_t1"],
            "mean_ADE_lower_than_all_single_t": multi_lower_all_single,
            "conditions_lower_than_t1": lower_than_t1_multi,
            "high_conf_no_harm_pass_count": high_pass_multi,
            "high_conf_interpretable_count": interpretable_high_multi,
            "low_conf_pass_count": low_pass_multi,
            "mean_motion_usage_ratio": motion_multi,
            "motion_usage_in_range_0p2_0p7": bool(0.2 <= motion_multi <= 0.7),
            "passed": multi_passed,
        }
    )

    if multi_passed:
        final_case = "Case B"
        final_rule = "V4_multit_confidence_bin"
    elif case_a_candidates:
        final_case = "Case A"
        final_rule = min(case_a_candidates, key=lambda variant: mean_ade[variant])
    else:
        final_case = "Case C"
        final_rule = "V1_single_t1"
        pass_rows.append(
            {
                "case": "Case C",
                "variant": "V1_single_t1",
                "six_condition_mean_ADE": mean_ade["V1_single_t1"],
                "six_condition_mean_ADE_t1": mean_ade["V1_single_t1"],
                "baseline_reproduction_status": baseline_status,
                "baseline_reproduction_ADE": baseline_ade,
                "passed": True,
                "notes": "Keep E3-t1 as final SDE candidate source; ablation-confirmed, not assumed in advance.",
            }
        )
    return pd.DataFrame(pass_rows), final_case, final_rule


def write_decision_markdown(
    full_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    baseline_status: str,
    baseline_ade: float,
    baseline_diff: float,
    final_case: str,
    final_rule: str,
    figure_paths: list[Path],
) -> None:
    mean_table = (
        full_df.groupby("variant")
        .agg(
            six_condition_mean_ADE=("ADE", "mean"),
            mean_motion_usage_ratio=("motion_usage_ratio", "mean"),
            high_conf_pass_count=("high_conf_no_harm_pass", lambda s: int((s == "PASS").sum())),
            low_conf_pass_count=("low_conf_pass", lambda s: int((s == "PASS").sum())),
        )
        .reset_index()
    )
    per_condition = cond_df[
        ["condition", "ADE_t1", "ADE_t2", "ADE_t3", "ADE_multit_bin", "best_variant"]
    ].copy()
    high_table = cond_df[
        [
            "condition",
            "high_conf_pass_t1",
            "high_conf_pass_t2",
            "high_conf_pass_t3",
            "high_conf_pass_multit_bin",
        ]
    ].copy()
    low_table = cond_df[
        ["condition", "low_conf_pass_t1", "low_conf_pass_t2", "low_conf_pass_t3", "low_conf_pass_multit_bin"]
    ].copy()
    motion_table = cond_df[
        [
            "condition",
            "motion_usage_ratio_t1",
            "motion_usage_ratio_t2",
            "motion_usage_ratio_t3",
            "motion_usage_ratio_multit_bin",
        ]
    ].copy()

    t2_better = bool(mean_table.set_index("variant").loc["V2_single_t2", "six_condition_mean_ADE"] < mean_table.set_index("variant").loc["V1_single_t1", "six_condition_mean_ADE"])
    t3_better = bool(mean_table.set_index("variant").loc["V3_single_t3", "six_condition_mean_ADE"] < mean_table.set_index("variant").loc["V1_single_t1", "six_condition_mean_ADE"])
    multit_best = bool(final_rule == "V4_multit_confidence_bin")

    lines = [
        "# E3 Multi-t Candidate-Source Ablation Decision Summary",
        "",
        "This evaluation reused existing per-frame SDEdit t1/t2/t3 candidates only. It did not train, generate candidates, use DPS, or modify existing E3 hold-out outputs.",
        "",
        "## Baseline Reproduction",
        "",
        f"- Known six-condition E3 fused t1 ADE anchor: `{BASELINE_ANCHOR_ADE:.6f}`",
        f"- Recomputed V1_single_t1 six-condition mean ADE: `{baseline_ade:.9f}`",
        f"- Absolute difference: `{baseline_diff:.9f}`",
        f"- Result: `{baseline_status}`",
        "",
        "## Six-Condition Mean ADE Table",
        "",
        markdown_table(mean_table, ["variant", "six_condition_mean_ADE", "mean_motion_usage_ratio", "high_conf_pass_count", "low_conf_pass_count"], 9),
        "",
        "## Per-Condition ADE Table",
        "",
        markdown_table(per_condition, ["condition", "ADE_t1", "ADE_t2", "ADE_t3", "ADE_multit_bin", "best_variant"], 9),
        "",
        "## High-Confidence No-Harm Table",
        "",
        markdown_table(high_table, list(high_table.columns), 6),
        "",
        "## Low-Confidence Improvement Table",
        "",
        markdown_table(low_table, list(low_table.columns), 6),
        "",
        "## Motion Usage Ratio Table",
        "",
        markdown_table(motion_table, list(motion_table.columns), 6),
        "",
        "## Case A/B/C Decision",
        "",
        markdown_table(pass_df.fillna(""), list(pass_df.columns), 9),
        "",
        f"- Final decision: `{final_case}`",
        f"- Final recommended SDE candidate source: `{final_rule}`",
        "",
        "## Required Answers",
        "",
        f"1. Are t2 or t3 better than t1 under the same fusion rule? `t2={t2_better}`, `t3={t3_better}`.",
        f"2. Is confidence-binned multi-t source better than all single-t variants? `{multit_best}`.",
        f"3. Candidate-source rule to use as final SDE component: `{final_rule}`.",
    ]
    if final_rule == "V1_single_t1":
        lines.append(
            "4. E3-t1 remains final because this multi-t candidate-source ablation confirmed it, not because it was assumed in advance."
        )
    else:
        lines.append(f"4. E3-t1 is replaced by `{final_rule}` under the pre-specified decision rules.")
    lines.extend(
        [
            "",
            "## Representative Figures",
            "",
            *[f"- `{rel(path)}`" for path in figure_paths],
            "",
            "## Output Files",
            "",
            f"- `{rel(FULL_METRICS_PATH)}`",
            f"- `{rel(CONDITION_SUMMARY_PATH)}`",
            f"- `{rel(BIN_METRICS_PATH)}`",
            f"- `{rel(PASS_FAIL_PATH)}`",
            f"- `{rel(DECISION_MD_PATH)}`",
        ]
    )
    DECISION_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(DECISION_MD_PATH)} written")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    candidate_paths = load_candidate_paths_from_inventory()
    clean = load_npy(DATASET_PATH, EXPECTED_SHAPE, "clean target dataset")
    clean_acc_mean_by_condition = float(np.mean(acceleration_rms(clean)))

    all_rows: list[dict] = []
    bin_rows: list[dict] = []
    figure_paths: list[Path] = []
    preds_by_condition: dict[str, dict[str, np.ndarray]] = {}

    for condition in CONDITIONS:
        degraded_path = locate_unique_companion(condition, "degraded")
        confidence_path = locate_unique_companion(condition, "confidence")
        degraded = load_npy(degraded_path, EXPECTED_SHAPE, f"{condition} degraded")
        confidence = load_npy(confidence_path, EXPECTED_CONF_SHAPE, f"{condition} confidence")

        sdedit = {
            1: load_npy(candidate_paths[(condition, 1)], EXPECTED_SHAPE, f"{condition} sdedit_t1"),
            2: load_npy(candidate_paths[(condition, 2)], EXPECTED_SHAPE, f"{condition} sdedit_t2"),
            3: load_npy(candidate_paths[(condition, 3)], EXPECTED_SHAPE, f"{condition} sdedit_t3"),
        }
        sources = {
            "V1_single_t1": sdedit[1],
            "V2_single_t2": sdedit[2],
            "V3_single_t3": sdedit[3],
            "V4_multit_confidence_bin": multit_source(degraded, confidence, sdedit[1], sdedit[2], sdedit[3]),
        }
        preds = {variant: fuse_from_source(degraded, source, confidence) for variant, source in sources.items()}
        preds_by_condition[condition] = preds

        ade_t1_condition = float(np.mean(per_traj_ade(preds["V1_single_t1"], clean)))
        for variant in VARIANTS:
            all_rows.append(
                method_metrics(
                    condition=condition,
                    variant=variant,
                    pred=preds[variant],
                    source=sources[variant],
                    clean=clean,
                    degraded=degraded,
                    confidence=confidence,
                    clean_acc_mean=clean_acc_mean_by_condition,
                    ade_t1_condition=ade_t1_condition,
                )
            )
            bin_rows.extend(bin_metric_rows(condition, variant, preds[variant], clean, degraded, confidence))

        t1_ade_traj = per_traj_ade(preds["V1_single_t1"], clean)
        median_idx = int(np.argmin(np.abs(t1_ade_traj - np.median(t1_ade_traj))))
        figure_paths.append(plot_representative(condition, median_idx, clean, degraded, confidence, preds))

    full_df = pd.DataFrame(all_rows)
    t1_mean_ade = float(full_df[full_df["variant"] == "V1_single_t1"]["ADE"].mean())
    baseline_diff = abs(t1_mean_ade - BASELINE_ANCHOR_ADE)
    baseline_status = "BASELINE_REPRODUCTION_PASS" if baseline_diff <= 1e-4 else "WARNING_BASELINE_MISMATCH"
    if baseline_status != "BASELINE_REPRODUCTION_PASS":
        print(
            "WARNING_BASELINE_MISMATCH "
            f"computed={t1_mean_ade:.9f} anchor={BASELINE_ANCHOR_ADE:.9f} diff={baseline_diff:.9f}"
        )
    else:
        print(f"BASELINE_REPRODUCTION_PASS computed={t1_mean_ade:.9f} anchor={BASELINE_ANCHOR_ADE:.9f}")

    for condition in CONDITIONS:
        t1_ade = float(full_df[(full_df["condition"] == condition) & (full_df["variant"] == "V1_single_t1")]["ADE"].iloc[0])
        for idx in full_df.index[full_df["condition"] == condition]:
            ade = float(full_df.loc[idx, "ADE"])
            full_df.loc[idx, "improvement_vs_E3_t1"] = t1_ade - ade
            full_df.loc[idx, "relative_improvement_vs_E3_t1"] = (t1_ade - ade) / (t1_ade + EPS)

    full_df.to_csv(FULL_METRICS_PATH, index=False)
    print(f"[FILE] {rel(FULL_METRICS_PATH)} written")

    cond_df = make_condition_summary(full_df)
    # Fill noisy ADE from companion degraded arrays for compact readability.
    ade_y_values = []
    for condition in CONDITIONS:
        degraded = load_npy(locate_unique_companion(condition, "degraded"), EXPECTED_SHAPE, f"{condition} degraded")
        ade_y_values.append(float(np.mean(per_traj_ade(degraded, clean))))
    cond_df["ADE_y"] = ade_y_values
    cond_df.to_csv(CONDITION_SUMMARY_PATH, index=False)
    print(f"[FILE] {rel(CONDITION_SUMMARY_PATH)} written")

    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(BIN_METRICS_PATH, index=False)
    print(f"[FILE] {rel(BIN_METRICS_PATH)} written")

    pass_df, final_case, final_rule = evaluate_decision(full_df, baseline_status, t1_mean_ade)
    pass_df.to_csv(PASS_FAIL_PATH, index=False)
    print(f"[FILE] {rel(PASS_FAIL_PATH)} written")

    write_decision_markdown(
        full_df=full_df,
        cond_df=cond_df,
        pass_df=pass_df,
        baseline_status=baseline_status,
        baseline_ade=t1_mean_ade,
        baseline_diff=baseline_diff,
        final_case=final_case,
        final_rule=final_rule,
        figure_paths=figure_paths,
    )

    mean_ades = full_df.groupby("variant")["ADE"].mean()
    print("E3_MULTIT_CANDIDATE_SOURCE_EVAL_COMPLETE")
    print(f"COMMAND={sys.executable} {rel(Path(__file__))}")
    print(f"BASELINE_REPRODUCTION_ADE={t1_mean_ade:.9f}")
    for variant in VARIANTS:
        print(f"MEAN_ADE_{variant}={float(mean_ades.loc[variant]):.9f}")
    print(f"FINAL_CASE={final_case}")
    print(f"FINAL_RECOMMENDED_SOURCE={final_rule}")
    print(f"OUTPUT_FULL_METRICS={rel(FULL_METRICS_PATH)}")
    print(f"OUTPUT_CONDITION_SUMMARY={rel(CONDITION_SUMMARY_PATH)}")
    print(f"OUTPUT_BIN_METRICS={rel(BIN_METRICS_PATH)}")
    print(f"OUTPUT_PASS_FAIL={rel(PASS_FAIL_PATH)}")
    print(f"OUTPUT_DECISION_MD={rel(DECISION_MD_PATH)}")


if __name__ == "__main__":
    main()
