from __future__ import annotations

from pathlib import Path
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


FORMAL_E1_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_oracle_residual_gating_6conditions"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_min_absolute_posterior_anchoring"
FIG_DIR = OUT_DIR / "figures"

CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
PROTOCOL_SUMMARY_PATH = FORMAL_E1_DIR / "e1_protocol_validation_summary.csv"
FORMAL_E1_FULL_PATH = FORMAL_E1_DIR / "e1_full_metrics.csv"

FULL_METRICS_PATH = OUT_DIR / "e2_min_full_metrics.csv"
BIN_METRICS_PATH = OUT_DIR / "e2_min_confidence_bin_metrics.csv"
PER_TRAJ_PATH = OUT_DIR / "e2_min_per_trajectory_metrics.csv"
SWEEP_PATH = OUT_DIR / "e2_min_parameter_sweep.csv"
PASS_FAIL_PATH = OUT_DIR / "e2_min_pass_fail_summary.csv"
DIAGNOSTICS_PATH = OUT_DIR / "e2_min_diagnostics.csv"
SUMMARY_PATH = OUT_DIR / "e2_min_summary.md"
TRAJ_PATH = OUT_DIR / "e2_min_optimized_trajectories.npz"

DEGRADATION_ORDER = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]

VARIANTS = ["V1_abs_anchor_smooth", "V2_uniform_cond_motion", "V3_conf_mod_cond_motion"]
ALPHA0_GRID = [0.0, 0.05, 0.10, 0.20, 0.50]
BETA_GRID = [0.01, 0.05, 0.10, 0.20]
SIGMA0_GRID = [0.50, 1.00]
RIDGE = 1e-7


def require_inputs() -> None:
    required = [CLEAN_PATH, PROTOCOL_SUMMARY_PATH, FORMAL_E1_FULL_PATH]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def per_traj_rmse(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    err = frame_error(pred, clean)
    return np.sqrt(np.mean(err**2, axis=1)).astype(np.float32)


def per_traj_acc_rms(pred: np.ndarray) -> np.ndarray:
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    return np.sqrt(np.mean(np.sum(acc**2, axis=-1), axis=1)).astype(np.float32)


def stats(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan, "p25": math.nan, "p75": math.nan}
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda value: f"{float(value):.6f}" if isinstance(value, (float, np.floating)) else str(value))
    lines = [
        "| " + " | ".join(map(str, out.columns)) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for row in out.values.tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def load_formal_e1_metrics() -> pd.DataFrame:
    df = pd.read_csv(FORMAL_E1_FULL_PATH)
    required = {"degradation", "method", "ADE_mean", "delta0_auto"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected Formal E1 metrics columns: {list(df.columns)}")
    return df


def load_arrays() -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    require_inputs()
    protocol_df = pd.read_csv(PROTOCOL_SUMMARY_PATH)
    formal_df = load_formal_e1_metrics()
    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    loaded: dict[str, dict] = {}

    for _, row in protocol_df.iterrows():
        degradation = str(row["degradation"])
        if degradation not in DEGRADATION_ORDER:
            continue
        if not bool(row.get("protocol_validation_pass", True)):
            raise RuntimeError(f"Protocol validation is false for {degradation}; refusing to run E2-Min.")

        degraded = np.load(str(row["degraded_path"])).astype(np.float32)
        cond = np.load(str(row["conditional_path"])).astype(np.float32)
        if cond.ndim == 4:
            cond = cond.mean(axis=0).astype(np.float32)
        clean = clean_all[: degraded.shape[0]].astype(np.float32)
        if clean.shape != degraded.shape or clean.shape != cond.shape:
            raise ValueError(f"Shape mismatch for {degradation}: clean={clean.shape}, degraded={degraded.shape}, cond={cond.shape}")

        e1_row = formal_df[(formal_df["degradation"] == degradation) & (formal_df["method"] == "e1_oracle_gated_residual")]
        if e1_row.empty:
            raise RuntimeError(f"Missing Formal E1 row for {degradation}")
        delta0 = float(e1_row.iloc[0]["delta0_auto"])
        conf = confidence_from_delta(degraded, clean, delta0)
        e1 = apply_e1(degraded, cond, conf)

        loaded[degradation] = {
            "clean": clean,
            "degraded": degraded,
            "cond": cond,
            "e1": e1,
            "confidence": conf,
            "delta0": delta0,
            "source": str(row["conditional_source"]),
            "conditional_path": str(row["conditional_path"]),
            "degraded_path": str(row["degraded_path"]),
        }

    missing = sorted(set(DEGRADATION_ORDER) - set(loaded))
    if missing:
        raise RuntimeError(f"Missing protocol-validated arrays for: {missing}")
    return loaded, protocol_df, formal_df


def confidence_from_delta(degraded: np.ndarray, clean: np.ndarray, delta0: float) -> np.ndarray:
    if delta0 <= 0 or not np.isfinite(delta0):
        raise ValueError(f"Invalid delta0: {delta0}")
    return np.exp(-frame_error(degraded, clean) / delta0).astype(np.float32)


def apply_e1(degraded: np.ndarray, cond: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    return (degraded + (1.0 - confidence[..., None]) * (cond - degraded)).astype(np.float32)


def variant_alpha(variant: str, alpha0: float, confidence: np.ndarray) -> np.ndarray:
    if variant == "V1_abs_anchor_smooth":
        return np.zeros_like(confidence, dtype=np.float64)
    if variant == "V2_uniform_cond_motion":
        return np.full_like(confidence, alpha0, dtype=np.float64)
    if variant == "V3_conf_mod_cond_motion":
        return (alpha0 * (1.0 - confidence)).astype(np.float64)
    raise ValueError(f"Unknown variant: {variant}")


def solve_single_trajectory(
    y: np.ndarray,
    x_cond: np.ndarray,
    confidence: np.ndarray,
    variant: str,
    alpha0: float,
    beta: float,
    sigma0: float,
) -> np.ndarray:
    t_len = y.shape[0]
    w_obs = (confidence.astype(np.float64) / (sigma0**2)).clip(min=0.0)
    alpha_frame = variant_alpha(variant, alpha0, confidence)
    alpha_edge = alpha_frame[1:]
    dcond = np.diff(x_cond.astype(np.float64), axis=0)
    pred = np.zeros_like(y, dtype=np.float64)

    a_base = np.zeros((t_len, t_len), dtype=np.float64)
    np.fill_diagonal(a_base, w_obs + RIDGE)

    for edge_idx in range(1, t_len):
        weight = float(alpha_edge[edge_idx - 1])
        if weight <= 0:
            continue
        i = edge_idx - 1
        j = edge_idx
        a_base[i, i] += weight
        a_base[j, j] += weight
        a_base[i, j] -= weight
        a_base[j, i] -= weight

    if beta > 0:
        for center in range(1, t_len - 1):
            idx = [center - 1, center, center + 1]
            q = np.array([1.0, -2.0, 1.0], dtype=np.float64)
            a_base[np.ix_(idx, idx)] += beta * np.outer(q, q)

    for dim in range(y.shape[1]):
        b = w_obs * y[:, dim].astype(np.float64)
        for edge_idx in range(1, t_len):
            weight = float(alpha_edge[edge_idx - 1])
            if weight <= 0:
                continue
            d = float(dcond[edge_idx - 1, dim])
            b[edge_idx - 1] -= weight * d
            b[edge_idx] += weight * d
        pred[:, dim] = np.linalg.solve(a_base, b)
    return pred.astype(np.float32)


def solve_batch(
    degraded: np.ndarray,
    cond: np.ndarray,
    confidence: np.ndarray,
    variant: str,
    alpha0: float,
    beta: float,
    sigma0: float,
) -> np.ndarray:
    out = np.empty_like(degraded, dtype=np.float32)
    for idx in range(degraded.shape[0]):
        out[idx] = solve_single_trajectory(degraded[idx], cond[idx], confidence[idx], variant, alpha0, beta, sigma0)
    return out


def baseline_methods(data: dict) -> dict[str, np.ndarray]:
    return {
        "noisy_input": data["degraded"],
        "stage3_cond_residual_t20": data["cond"],
        "formal_e1_linear_gate": data["e1"],
    }


def condition_ade(pred: np.ndarray, clean: np.ndarray) -> float:
    return float(frame_error(pred, clean).mean(axis=1).mean())


def evaluate_parameter_sweep(loaded: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    preds_cache: dict[tuple[str, float, float, float], dict[str, np.ndarray]] = {}

    total = len(VARIANTS) * len(ALPHA0_GRID) * len(BETA_GRID) * len(SIGMA0_GRID)
    done = 0
    for variant in VARIANTS:
        for alpha0 in ALPHA0_GRID:
            for beta in BETA_GRID:
                for sigma0 in SIGMA0_GRID:
                    done += 1
                    key = (variant, alpha0, beta, sigma0)
                    condition_preds: dict[str, np.ndarray] = {}
                    condition_rows: list[dict] = []
                    for degradation in DEGRADATION_ORDER:
                        data = loaded[degradation]
                        pred = solve_batch(data["degraded"], data["cond"], data["confidence"], variant, alpha0, beta, sigma0)
                        condition_preds[degradation] = pred
                        ade_e2 = condition_ade(pred, data["clean"])
                        ade_e1 = condition_ade(data["e1"], data["clean"])
                        ade_noisy = condition_ade(data["degraded"], data["clean"])
                        ade_cond = condition_ade(data["cond"], data["clean"])
                        condition_rows.append(
                            {
                                "variant": variant,
                                "alpha0": alpha0,
                                "beta": beta,
                                "sigma0": sigma0,
                                "degradation": degradation,
                                "conditional_source": data["source"],
                                "ADE_noisy": ade_noisy,
                                "ADE_cond": ade_cond,
                                "ADE_e1": ade_e1,
                                "ADE_e2": ade_e2,
                                "improves_vs_e1": bool(ade_e2 <= ade_e1),
                                "ADE_gain_vs_e1": ade_e1 - ade_e2,
                            }
                        )
                    n_improved = int(sum(row["improves_vs_e1"] for row in condition_rows))
                    mean_gain = float(np.mean([row["ADE_gain_vs_e1"] for row in condition_rows]))
                    mean_ade = float(np.mean([row["ADE_e2"] for row in condition_rows]))
                    for row in condition_rows:
                        row["six_ADE_improved_count"] = n_improved
                        row["mean_ADE_gain_vs_e1_across_6"] = mean_gain
                        row["mean_ADE_e2_across_6"] = mean_ade
                        rows.append(row)
                    preds_cache[key] = condition_preds
                    if done % 20 == 0 or done == total:
                        print(f"E2 parameter sweep progress: {done}/{total}")

    sweep_df = pd.DataFrame(rows)
    grouped = (
        sweep_df.groupby(["variant", "alpha0", "beta", "sigma0"], as_index=False)
        .agg(
            six_ADE_improved_count=("improves_vs_e1", "sum"),
            mean_ADE_gain_vs_e1_across_6=("ADE_gain_vs_e1", "mean"),
            mean_ADE_e2_across_6=("ADE_e2", "mean"),
        )
        .sort_values(
            ["six_ADE_improved_count", "mean_ADE_gain_vs_e1_across_6", "mean_ADE_e2_across_6"],
            ascending=[False, False, True],
        )
    )
    best = grouped.iloc[0].to_dict()
    best_key = (str(best["variant"]), float(best["alpha0"]), float(best["beta"]), float(best["sigma0"]))
    best["selection_rule"] = "maximize six-condition ADE improvements vs Formal E1; tie-break by mean ADE gain; then mean ADE"
    best["predictions"] = preds_cache[best_key]
    return sweep_df, best


def compute_motion_usage(pred: np.ndarray, degraded: np.ndarray, cond: np.ndarray) -> float:
    dy = np.diff(degraded, axis=1)
    dp = np.diff(pred, axis=1)
    dc = np.diff(cond, axis=1)
    den = float(np.linalg.norm(dc - dy, axis=-1).mean())
    num = float(np.linalg.norm(dp - dy, axis=-1).mean())
    return num / den if den > 0 else math.nan


def compute_offset_error(pred: np.ndarray, clean: np.ndarray) -> float:
    return float(np.linalg.norm(np.mean(pred - clean, axis=1), axis=-1).mean())


def build_evaluation_tables(loaded: dict[str, dict], selected: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    alpha0 = float(selected["alpha0"])
    beta = float(selected["beta"])
    sigma0 = float(selected["sigma0"])
    variant_predictions = selected["variant_predictions"]

    full_rows: list[dict] = []
    bin_rows: list[dict] = []
    per_traj_rows: list[dict] = []
    diag_rows: list[dict] = []
    eval_cache: dict[str, dict] = {}

    for degradation in DEGRADATION_ORDER:
        data = loaded[degradation]
        clean = data["clean"]
        degraded = data["degraded"]
        cond = data["cond"]
        e1 = data["e1"]
        confidence = data["confidence"]
        methods = baseline_methods(data)
        for variant in VARIANTS:
            methods[variant] = variant_predictions[variant][degradation]

        errors = {name: frame_error(pred, clean) for name, pred in methods.items()}
        ade_traj = {name: err.mean(axis=1) for name, err in errors.items()}
        rmse_traj = {name: per_traj_rmse(pred, clean) for name, pred in methods.items()}
        smooth_traj = {name: per_traj_acc_rms(pred) for name, pred in methods.items()}
        high = confidence > 0.7
        mid = (confidence >= 0.3) & (confidence <= 0.7)
        low = confidence < 0.3
        masks = {"high": high, "mid": mid, "low": low}

        for method, pred in methods.items():
            ade_s = stats(ade_traj[method])
            rmse_s = stats(rmse_traj[method])
            smooth_s = stats(smooth_traj[method])
            full_rows.append(
                {
                    "degradation": degradation,
                    "method": method,
                    "conditional_source": data["source"],
                    "N_trajectories": clean.shape[0],
                    "delta0_auto": data["delta0"],
                    "selected_alpha0": alpha0,
                    "selected_beta": beta,
                    "selected_sigma0": sigma0,
                    "ADE_mean": ade_s["mean"],
                    "ADE_std": ade_s["std"],
                    "ADE_median": ade_s["median"],
                    "ADE_min": ade_s["min"],
                    "ADE_max": ade_s["max"],
                    "ADE_p25": ade_s["p25"],
                    "ADE_p75": ade_s["p75"],
                    "RMSE_mean": rmse_s["mean"],
                    "RMSE_std": rmse_s["std"],
                    "smooth_acc_rms_mean": smooth_s["mean"],
                    "smooth_acc_rms_std": smooth_s["std"],
                    "win_rate_vs_noisy": float(np.mean(ade_traj[method] <= ade_traj["noisy_input"])),
                    "win_rate_vs_stage3_cond": float(np.mean(ade_traj[method] <= ade_traj["stage3_cond_residual_t20"])),
                    "win_rate_vs_formal_e1": float(np.mean(ade_traj[method] <= ade_traj["formal_e1_linear_gate"])),
                    "motion_usage_ratio": compute_motion_usage(pred, degraded, cond) if method.startswith("V") else math.nan,
                    "noisy_reversion_gap": float(ade_traj[method].mean() - ade_traj["noisy_input"].mean()),
                    "bias_offset_error": compute_offset_error(pred, clean) if degradation == "bias_medium" else math.nan,
                }
            )

            for bin_name, mask in masks.items():
                s = stats(errors[method][mask])
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
                    }
                )

        for idx in range(clean.shape[0]):
            row = {
                "degradation": degradation,
                "trajectory_id": idx,
                "delta0_auto": data["delta0"],
                "confidence_min": float(confidence[idx].min()),
                "confidence_mean": float(confidence[idx].mean()),
                "confidence_max": float(confidence[idx].max()),
            }
            for method in methods:
                row[f"ADE_{method}"] = float(ade_traj[method][idx])
                row[f"RMSE_{method}"] = float(rmse_traj[method][idx])
                row[f"smooth_{method}"] = float(smooth_traj[method][idx])
            per_traj_rows.append(row)

        for method in VARIANTS:
            high_ade = float(errors[method][high].mean()) if np.any(high) else math.nan
            high_noisy = float(errors["noisy_input"][high].mean()) if np.any(high) else math.nan
            low_ade = float(errors[method][low].mean()) if np.any(low) else math.nan
            low_noisy = float(errors["noisy_input"][low].mean()) if np.any(low) else math.nan
            low_cond = float(errors["stage3_cond_residual_t20"][low].mean()) if np.any(low) else math.nan
            low_useful = bool(np.isfinite(low_noisy) and np.isfinite(low_cond) and low_cond < low_noisy)
            if low_useful:
                c3 = bool(low_ade <= 1.10 * low_cond)
            else:
                c3 = True
            no_harm_ratio = float(np.mean(errors[method][high] <= 1.05 * errors["noisy_input"][high])) if np.any(high) else math.nan
            diag_rows.append(
                {
                    "degradation": degradation,
                    "method": method,
                    "ADE_noisy": float(ade_traj["noisy_input"].mean()),
                    "ADE_cond": float(ade_traj["stage3_cond_residual_t20"].mean()),
                    "ADE_e1": float(ade_traj["formal_e1_linear_gate"].mean()),
                    "ADE_e2": float(ade_traj[method].mean()),
                    "improves_vs_e1": bool(ade_traj[method].mean() <= ade_traj["formal_e1_linear_gate"].mean()),
                    "ADE_gain_vs_e1": float(ade_traj["formal_e1_linear_gate"].mean() - ade_traj[method].mean()),
                    "N_high_frames": int(high.sum()),
                    "ADE_noisy_high": high_noisy,
                    "ADE_e1_high": float(errors["formal_e1_linear_gate"][high].mean()) if np.any(high) else math.nan,
                    "ADE_e2_high": high_ade,
                    "high_conf_no_harm_pass_1p05": bool(np.isfinite(high_noisy) and high_ade <= 1.05 * high_noisy),
                    "high_conf_no_harm_ratio": no_harm_ratio,
                    "N_low_frames": int(low.sum()),
                    "ADE_noisy_low": low_noisy,
                    "ADE_cond_low": low_cond,
                    "ADE_e1_low": float(errors["formal_e1_linear_gate"][low].mean()) if np.any(low) else math.nan,
                    "ADE_e2_low": low_ade,
                    "conditional_useful_low": low_useful,
                    "low_conf_preservation_pass": c3,
                    "motion_usage_ratio": compute_motion_usage(methods[method], degraded, cond),
                    "noisy_reversion_gap": float(ade_traj[method].mean() - ade_traj["noisy_input"].mean()),
                    "bias_offset_error": compute_offset_error(methods[method], clean) if degradation == "bias_medium" else math.nan,
                }
            )

        eval_cache[degradation] = {
            "clean": clean,
            "degraded": degraded,
            "cond": cond,
            "e1": e1,
            "confidence": confidence,
            "methods": methods,
            "errors": errors,
            "ade_traj": ade_traj,
        }

    return pd.DataFrame(full_rows), pd.DataFrame(bin_rows), pd.DataFrame(per_traj_rows), pd.DataFrame(diag_rows), eval_cache


def select_shared_predictions(loaded: dict[str, dict], selected: dict) -> dict[str, dict[str, np.ndarray]]:
    alpha0 = float(selected["alpha0"])
    beta = float(selected["beta"])
    sigma0 = float(selected["sigma0"])
    preds: dict[str, dict[str, np.ndarray]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for degradation in DEGRADATION_ORDER:
            data = loaded[degradation]
            preds[variant][degradation] = solve_batch(data["degraded"], data["cond"], data["confidence"], variant, alpha0, beta, sigma0)
    return preds


def build_pass_fail(diagnostics_df: pd.DataFrame, selected: dict) -> pd.DataFrame:
    selected_variant = str(selected["variant"])
    selected_rows = diagnostics_df[diagnostics_df["method"] == selected_variant].copy()
    six_ade_count = int(selected_rows["improves_vs_e1"].sum())
    six_ade_pass = bool(six_ade_count == len(DEGRADATION_ORDER))
    c2_failures = selected_rows[selected_rows["high_conf_no_harm_pass_1p05"] != True]["degradation"].tolist()
    c3_failures = selected_rows[selected_rows["low_conf_preservation_pass"] != True]["degradation"].tolist()
    repair_rows = selected_rows[selected_rows["degradation"].isin(["drift_medium", "burst_medium"])]
    direct_repair_pass = bool(
        (repair_rows["improves_vs_e1"].all())
        and (repair_rows["high_conf_no_harm_pass_1p05"].all())
    )

    bias_row = selected_rows[selected_rows["degradation"] == "bias_medium"]
    bias_offset_e1 = math.nan
    bias_offset_e2 = math.nan
    bias_offset_improves = False
    if not bias_row.empty:
        degradation = "bias_medium"
        # This is filled from diagnostics for E2; E1 baseline is computed in the caller cache and injected below if needed.
        bias_offset_e2 = float(bias_row.iloc[0]["bias_offset_error"])

    summary = selected_rows.copy()
    summary["selected_variant"] = selected_variant
    summary["selected_alpha0"] = float(selected["alpha0"])
    summary["selected_beta"] = float(selected["beta"])
    summary["selected_sigma0"] = float(selected["sigma0"])
    summary["E2_6ADE_improved_count"] = six_ade_count
    summary["E2_6ADE_PASS"] = six_ade_pass
    summary["E2_C1_6ADE_PASS"] = six_ade_pass
    summary["E2_C2_high_conf_PASS"] = len(c2_failures) == 0
    summary["E2_C2_failures"] = ";".join(c2_failures)
    summary["E2_C3_low_conf_PASS"] = len(c3_failures) == 0
    summary["E2_C3_failures"] = ";".join(c3_failures)
    summary["E2_C4_drift_burst_repair_PASS"] = direct_repair_pass
    summary["E2_C5_bias_offset_error_E1"] = bias_offset_e1
    summary["E2_C5_bias_offset_error_E2"] = bias_offset_e2
    summary["E2_C5_bias_offset_improves"] = bias_offset_improves
    summary["overall_label"] = "PASS" if six_ade_pass else "NO-PASS"
    return summary


def inject_bias_offset(pass_df: pd.DataFrame, eval_cache: dict) -> pd.DataFrame:
    if "bias_medium" not in eval_cache:
        return pass_df
    data = eval_cache["bias_medium"]
    e1_offset = compute_offset_error(data["e1"], data["clean"])
    selected_variant = str(pass_df["selected_variant"].iloc[0])
    e2_offset = compute_offset_error(data["methods"][selected_variant], data["clean"])
    pass_df["E2_C5_bias_offset_error_E1"] = e1_offset
    pass_df["E2_C5_bias_offset_error_E2"] = e2_offset
    pass_df["E2_C5_bias_offset_improves"] = bool(e2_offset <= e1_offset)
    return pass_df


def choose_case(eval_cache: dict, kind: str, selected_variant: str) -> tuple[str, int] | None:
    best: tuple[float, str, int] | None = None
    for degradation, data in eval_cache.items():
        conf = data["confidence"]
        errors = data["errors"]
        ade = data["ade_traj"]
        if kind == "drift_high_conf_failure" and degradation != "drift_medium":
            continue
        if kind == "burst_boundary_failure" and degradation != "burst_medium":
            continue
        if kind == "bias_offset" and degradation != "bias_medium":
            continue
        if kind == "combined" and degradation != "combined_medium":
            continue

        for idx in range(conf.shape[0]):
            if kind in {"drift_high_conf_failure", "burst_boundary_failure"}:
                mask = conf[idx] > 0.7
                if not np.any(mask):
                    continue
                score = float((errors["formal_e1_linear_gate"][idx][mask] - errors["noisy_input"][idx][mask]).mean())
            elif kind == "bias_offset":
                score = float(np.linalg.norm(np.mean(data["e1"][idx] - data["clean"][idx], axis=0)))
            elif kind == "combined":
                score = float(ade["formal_e1_linear_gate"][idx] - ade[selected_variant][idx])
            elif kind == "low_conf_preservation":
                mask = conf[idx] < 0.3
                if not np.any(mask):
                    continue
                score = float(errors["stage3_cond_residual_t20"][idx][mask].mean() - errors[selected_variant][idx][mask].mean())
            else:
                score = float(ade[selected_variant][idx] - ade["formal_e1_linear_gate"][idx])
            if best is None or score > best[0]:
                best = (score, degradation, idx)
    if best is None:
        return None
    return best[1], best[2]


def plot_case(eval_cache: dict, degradation: str, idx: int, selected_variant: str, title: str, path: Path) -> None:
    data = eval_cache[degradation]
    clean = data["clean"][idx]
    degraded = data["degraded"][idx]
    cond = data["cond"][idx]
    e1 = data["e1"][idx]
    v1 = data["methods"]["V1_abs_anchor_smooth"][idx]
    v3 = data["methods"]["V3_conf_mod_cond_motion"][idx]
    conf = data["confidence"][idx]
    errors = data["errors"]
    t = np.arange(conf.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    ax.plot(clean[:, 0], clean[:, 1], "k-o", lw=1.4, ms=3, label="clean target")
    ax.plot(degraded[:, 0], degraded[:, 1], "C1-o", lw=1.1, ms=3, label="degraded input")
    ax.plot(cond[:, 0], cond[:, 1], "C3-o", lw=1.1, ms=3, label="Stage 3 cond")
    ax.plot(e1[:, 0], e1[:, 1], "C0-o", lw=1.1, ms=3, label="Formal E1")
    ax.plot(v1[:, 0], v1[:, 1], "C2-o", lw=1.2, ms=3, label="E2 V1")
    ax.plot(v3[:, 0], v3[:, 1], "C4-o", lw=1.2, ms=3, label="E2 V3")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\n{degradation}, trajectory {idx}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    axe = axes[1]
    for name, color, label in [
        ("noisy_input", "C1", "noisy"),
        ("stage3_cond_residual_t20", "C3", "cond"),
        ("formal_e1_linear_gate", "C0", "E1"),
        ("V1_abs_anchor_smooth", "C2", "E2 V1"),
        ("V3_conf_mod_cond_motion", "C4", "E2 V3"),
    ]:
        axe.plot(t, errors[name][idx], color=color, marker="o", ms=3, lw=1.1, label=label)
    axe.set_title("per-frame error")
    axe.set_xlabel("t")
    axe.grid(alpha=0.25)
    axe.legend(fontsize=7)

    axc = axes[2]
    axc.plot(t, conf, "k-o", ms=3, lw=1.2, label="confidence c_t")
    axc.axhline(0.7, color="0.35", ls="--", lw=1)
    axc.axhline(0.3, color="0.35", ls=":", lw=1)
    axc.set_ylim(0, 1.05)
    axc.set_title(f"confidence; selected={selected_variant}")
    axc.set_xlabel("t")
    axc.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_figures(eval_cache: dict, selected_variant: str) -> list[dict[str, str]]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    requests = [
        ("drift_high_confidence_e1_failure", "drift_high_conf_failure", "Drift high-confidence E1 failure case"),
        ("burst_boundary_failure", "burst_boundary_failure", "Burst boundary failure case"),
        ("bias_absolute_offset", "bias_offset", "Bias absolute-offset case"),
        ("combined_degradation", "combined", "Combined degradation case"),
        ("low_confidence_preservation", "low_conf_preservation", "Low-confidence correction-preservation case"),
        ("failure_or_ambiguous", "failure", "Failure or ambiguous case"),
    ]
    rows: list[dict[str, str]] = []
    for stem, kind, title in requests:
        case = choose_case(eval_cache, kind, selected_variant)
        if case is None:
            rows.append({"figure": stem, "status": "MISSING", "path": "", "note": "no eligible case"})
            continue
        degradation, idx = case
        path = FIG_DIR / f"{stem}_{degradation}_traj{idx}.png"
        plot_case(eval_cache, degradation, idx, selected_variant, title, path)
        rows.append({"figure": stem, "status": "FOUND", "path": str(path), "note": f"{degradation}, trajectory {idx}"})
    return rows


def save_trajectories(eval_cache: dict) -> None:
    arrays = {}
    for degradation, data in eval_cache.items():
        arrays[f"{degradation}__clean"] = data["clean"]
        arrays[f"{degradation}__degraded"] = data["degraded"]
        arrays[f"{degradation}__stage3_cond"] = data["cond"]
        arrays[f"{degradation}__formal_e1"] = data["e1"]
        arrays[f"{degradation}__confidence"] = data["confidence"]
        for variant in VARIANTS:
            arrays[f"{degradation}__{variant}"] = data["methods"][variant]
    np.savez_compressed(TRAJ_PATH, **arrays)


def write_summary(
    protocol_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    selected: dict,
    full_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    figure_rows: list[dict[str, str]],
) -> None:
    selected_variant = str(selected["variant"])
    selected_metrics = full_df[full_df["method"].isin(["noisy_input", "stage3_cond_residual_t20", "formal_e1_linear_gate"] + VARIANTS)]
    ade_matrix = selected_metrics.pivot(index="degradation", columns="method", values="ADE_mean").reset_index()
    variant_table = full_df[full_df["method"].isin(VARIANTS)][
        ["degradation", "method", "ADE_mean", "RMSE_mean", "motion_usage_ratio", "noisy_reversion_gap", "bias_offset_error"]
    ]
    selected_diag = diagnostics_df[diagnostics_df["method"] == selected_variant]
    top_sweep = (
        sweep_df.groupby(["variant", "alpha0", "beta", "sigma0"], as_index=False)
        .agg(
            six_ADE_improved_count=("improves_vs_e1", "sum"),
            mean_ADE_gain_vs_e1_across_6=("ADE_gain_vs_e1", "mean"),
            mean_ADE_e2_across_6=("ADE_e2", "mean"),
        )
        .sort_values(["six_ADE_improved_count", "mean_ADE_gain_vs_e1_across_6", "mean_ADE_e2_across_6"], ascending=[False, False, True])
        .head(12)
    )
    bias_offsets = full_df[full_df["degradation"].eq("bias_medium")][["method", "bias_offset_error"]].dropna()

    lines = [
        "# Stage 4 E2-Min Absolute-Space Posterior Anchoring",
        "",
        "## Objective and Hypothesis",
        "E2-Min tests deterministic confidence-modulated absolute-space posterior anchoring after E1. E1 is complete: it showed confidence is useful, but scalar residual gating failed high-confidence no-harm for drift and burst.",
        "",
        "## Selection Rule",
        "Per the updated protocol, the selected shared setting is chosen only by six-condition ADE improvement versus Formal E1: maximize the number of degradations with ADE_E2 <= ADE_E1, with ties broken by mean ADE gain and then mean ADE.",
        "",
        "## Selected Setting",
        f"- selected variant: {selected_variant}",
        f"- alpha0: {float(selected['alpha0']):.6f}",
        f"- beta: {float(selected['beta']):.6f}",
        f"- sigma0: {float(selected['sigma0']):.6f}",
        f"- six-condition ADE improvements: {int(selected['six_ADE_improved_count'])}/6",
        f"- mean ADE gain vs Formal E1: {float(selected['mean_ADE_gain_vs_e1_across_6']):.6f}",
        "",
        "## Formula",
        "V1: alpha_t = 0. V2: alpha_t = alpha0. V3: alpha_t = alpha0 * (1 - c_t). Each trajectory is solved deterministically in absolute space from the quadratic objective with observation anchoring, optional conditional motion reference, and acceleration smoothness.",
        "",
        "## Data Provenance",
        markdown_table(protocol_df[["degradation", "conditional_source", "conditional_path", "degraded_path", "shape_clean", "shape_cond", "protocol_validation_pass"]]),
        "",
        "## Full ADE Matrix",
        markdown_table(ade_matrix),
        "",
        "## V1 / V2 / V3 Metrics",
        markdown_table(variant_table),
        "",
        "## Confidence-Bin Analysis",
        markdown_table(bin_df[bin_df["method"].eq(selected_variant)][["degradation", "confidence_bin", "N_frames", "ADE_mean", "ADE_std", "ADE_median", "ADE_p25", "ADE_p75"]]),
        "",
        "## Bias Offset Analysis",
        markdown_table(bias_offsets),
        "",
        "## Parameter Sweep Top Rows",
        markdown_table(top_sweep),
        "",
        "## PASS / NO-PASS",
        markdown_table(pass_df[["degradation", "selected_variant", "ADE_noisy", "ADE_cond", "ADE_e1", "ADE_e2", "improves_vs_e1", "E2_6ADE_improved_count", "E2_6ADE_PASS", "high_conf_no_harm_pass_1p05", "low_conf_preservation_pass", "E2_C5_bias_offset_improves", "overall_label"]]),
        "",
        "## Diagnostics",
        markdown_table(selected_diag[["degradation", "ADE_gain_vs_e1", "ADE_e2_high", "ADE_noisy_high", "high_conf_no_harm_pass_1p05", "ADE_e2_low", "ADE_cond_low", "low_conf_preservation_pass", "motion_usage_ratio", "noisy_reversion_gap"]]),
        "",
        "## Figures",
    ]
    for row in figure_rows:
        lines.append(f"- {row['figure']}: {row['status']} {row['path']} ({row['note']})")
    lines.extend(
        [
            "",
            "## E2-DPS Decision",
            "If six-condition ADE improves but drift/burst high-confidence no-harm or bias offset remain weak, E2-DPS remains motivated as the next probabilistic posterior anchoring step. If E2-Min satisfies six-condition ADE and repairs the E1 failures, E2-DPS can be scoped as confirmatory rather than rescue.",
            "",
            "## Boundary Statement",
            "E2-Min is deterministic posterior anchoring in absolute space. It does not train, resample conditional outputs, use SDEdit, or modify Stage 3 data.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded, protocol_df, formal_df = load_arrays()
    print(f"Loaded Stage 3 protocol-validated arrays: {len(loaded)}/6")

    sweep_df, selected = evaluate_parameter_sweep(loaded)
    variant_predictions = select_shared_predictions(loaded, selected)
    selected["variant_predictions"] = variant_predictions

    full_df, bin_df, per_traj_df, diagnostics_df, eval_cache = build_evaluation_tables(loaded, selected)
    pass_df = build_pass_fail(diagnostics_df, selected)
    pass_df = inject_bias_offset(pass_df, eval_cache)
    figure_rows = make_figures(eval_cache, str(selected["variant"]))
    save_trajectories(eval_cache)

    sweep_df.to_csv(SWEEP_PATH, index=False)
    full_df.to_csv(FULL_METRICS_PATH, index=False)
    bin_df.to_csv(BIN_METRICS_PATH, index=False)
    per_traj_df.to_csv(PER_TRAJ_PATH, index=False)
    diagnostics_df.to_csv(DIAGNOSTICS_PATH, index=False)
    pass_df.to_csv(PASS_FAIL_PATH, index=False)
    pd.DataFrame(figure_rows).to_csv(OUT_DIR / "e2_min_figures.csv", index=False)
    write_summary(protocol_df, sweep_df, selected, full_df, bin_df, diagnostics_df, pass_df, figure_rows)

    selected_variant = str(selected["variant"])
    selected_diag = diagnostics_df[diagnostics_df["method"] == selected_variant]
    drift_c2 = bool(selected_diag[selected_diag["degradation"].eq("drift_medium")]["high_conf_no_harm_pass_1p05"].iloc[0])
    burst_c2 = bool(selected_diag[selected_diag["degradation"].eq("burst_medium")]["high_conf_no_harm_pass_1p05"].iloc[0])
    bias_improves = bool(pass_df["E2_C5_bias_offset_improves"].iloc[0])

    print("STAGE4_E2_MIN_ABSOLUTE_POSTERIOR_ANCHORING_COMPLETE")
    print(f"output_dir: {OUT_DIR}")
    print(f"selected_variant: {selected_variant}")
    print(f"selected_alpha0: {float(selected['alpha0']):.6f}")
    print(f"selected_beta: {float(selected['beta']):.6f}")
    print(f"selected_sigma0: {float(selected['sigma0']):.6f}")
    print(f"six_ADE_improved_count: {int(selected['six_ADE_improved_count'])}/6")
    print(f"E2_6ADE_PASS: {bool(pass_df['E2_6ADE_PASS'].iloc[0])}")
    print(f"drift_high_conf_no_harm_fixed: {drift_c2}")
    print(f"burst_high_conf_no_harm_fixed: {burst_c2}")
    print(f"bias_offset_improves: {bias_improves}")
    print(f"full_metrics: {FULL_METRICS_PATH}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
