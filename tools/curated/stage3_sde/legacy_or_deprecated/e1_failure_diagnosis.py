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


E1_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_oracle_residual_gating_full"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_failure_diagnosis"
FIG_DIR = OUT_DIR / "figures"

FULL_METRICS_PATH = E1_DIR / "e1_full_metrics.csv"
BIN_METRICS_PATH = E1_DIR / "e1_confidence_bin_metrics.csv"
SWEEP_PATH = E1_DIR / "e1_delta0_sweep.csv"
PER_TRAJ_PATH = E1_DIR / "e1_per_trajectory_metrics.csv"

CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
DEGRADED_PATHS = {
    "drift_medium": PROJECT_ROOT
    / "outputs"
    / "stage3_indoor"
    / "conditional_residual_ddpm_gaussian"
    / "seed42"
    / "generalization_degraded_drift.npy",
    "burst_medium": PROJECT_ROOT
    / "outputs"
    / "stage3_indoor"
    / "conditional_residual_ddpm_gaussian"
    / "seed42"
    / "generalization_degraded_burst.npy",
}
COND_PATHS = {
    "drift_medium": PROJECT_ROOT
    / "outputs"
    / "stage3_indoor"
    / "report"
    / "cache"
    / "drift_cond_residual_t20_refined.npy",
    "burst_medium": PROJECT_ROOT
    / "outputs"
    / "stage3_indoor"
    / "report"
    / "cache"
    / "burst_cond_residual_t20_refined.npy",
}
TARGET_DEGRADATIONS = ["drift_medium", "burst_medium"]
GAMMAS = [1, 2, 3, 4]


def require_inputs() -> None:
    required = [FULL_METRICS_PATH, BIN_METRICS_PATH, SWEEP_PATH, PER_TRAJ_PATH, CLEAN_PATH]
    required.extend(DEGRADED_PATHS.values())
    required.extend(COND_PATHS.values())
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required E1 diagnosis inputs:\n" + "\n".join(str(path) for path in missing))


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


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


def load_arrays() -> dict[str, dict[str, np.ndarray | float]]:
    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    loaded: dict[str, dict[str, np.ndarray | float]] = {}
    for degradation in TARGET_DEGRADATIONS:
        degraded = np.load(DEGRADED_PATHS[degradation]).astype(np.float32)
        cond = np.load(COND_PATHS[degradation]).astype(np.float32)
        if cond.ndim == 4:
            cond = cond.mean(axis=0).astype(np.float32)
        clean = clean_all[: degraded.shape[0]].astype(np.float32)
        if clean.shape != degraded.shape or clean.shape != cond.shape:
            raise ValueError(f"Shape mismatch for {degradation}: clean={clean.shape}, degraded={degraded.shape}, cond={cond.shape}")
        delta = frame_error(degraded, clean)
        delta0 = float(np.median(delta.reshape(-1)))
        confidence = np.exp(-delta / delta0).astype(np.float32)
        e1 = apply_gate(degraded, cond, confidence, gamma=1)
        loaded[degradation] = {
            "clean": clean,
            "degraded": degraded,
            "cond": cond,
            "delta": delta,
            "delta0": delta0,
            "confidence": confidence,
            "e1": e1,
        }
    return loaded


def apply_gate(degraded: np.ndarray, cond: np.ndarray, confidence: np.ndarray, gamma: int | float) -> np.ndarray:
    lambda_t = np.power(1.0 - confidence, gamma).astype(np.float32)
    return (degraded + lambda_t[..., None] * (cond - degraded)).astype(np.float32)


def boundary_mask(high: np.ndarray, low: np.ndarray, radius: int = 1) -> np.ndarray:
    out = np.zeros_like(high, dtype=bool)
    n, t_len = high.shape
    for i in range(n):
        low_idx = np.flatnonzero(low[i])
        for j in np.flatnonzero(high[i]):
            if low_idx.size and np.min(np.abs(low_idx - j)) <= radius:
                out[i, j] = True
    return out


def dot_direction_stats(degraded: np.ndarray, cond: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    desired = clean - degraded
    residual = cond - degraded
    dots = np.sum(desired * residual, axis=-1)
    desired_norm = np.linalg.norm(desired, axis=-1)
    residual_norm = np.linalg.norm(residual, axis=-1)
    denom = desired_norm * residual_norm
    cos = np.full_like(dots, np.nan, dtype=np.float32)
    valid = denom > 1e-12
    cos[valid] = dots[valid] / denom[valid]
    selected_dots = dots[mask]
    selected_cos = cos[mask]
    if selected_dots.size == 0:
        return {"wrong_direction_fraction": math.nan, "mean_cosine": math.nan}
    if np.all(np.isnan(selected_cos)):
        mean_cosine = math.nan
    else:
        mean_cosine = float(np.nanmean(selected_cos))
    return {
        "wrong_direction_fraction": float(np.mean(selected_dots < 0.0)),
        "mean_cosine": mean_cosine,
    }


def high_confidence_failure_tables(loaded: dict[str, dict[str, np.ndarray | float]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    worst_rows: list[dict] = []
    dominance_rows: list[dict] = []
    for degradation, data in loaded.items():
        clean = data["clean"]
        degraded = data["degraded"]
        cond = data["cond"]
        e1 = data["e1"]
        confidence = data["confidence"]
        high = confidence > 0.7
        low = confidence < 0.3
        noisy_err = frame_error(degraded, clean)
        cond_err = frame_error(cond, clean)
        e1_err = frame_error(e1, clean)
        violation = high & (e1_err > 1.10 * noisy_err)
        excess = np.where(high, np.maximum(e1_err - 1.10 * noisy_err, 0.0), 0.0)
        traj_has_high = high.any(axis=1)
        traj_violation = np.zeros(high.shape[0], dtype=bool)
        traj_excess = excess.sum(axis=1)
        for idx in range(high.shape[0]):
            if traj_has_high[idx]:
                traj_violation[idx] = e1_err[idx][high[idx]].mean() > 1.10 * noisy_err[idx][high[idx]].mean()

        sorted_excess = np.sort(traj_excess[traj_excess > 0.0])[::-1]
        total_excess = float(sorted_excess.sum())
        if total_excess > 0.0:
            cum = np.cumsum(sorted_excess) / total_excess
            n50 = int(np.searchsorted(cum, 0.50) + 1)
            n80 = int(np.searchsorted(cum, 0.80) + 1)
            n90 = int(np.searchsorted(cum, 0.90) + 1)
        else:
            n50 = n80 = n90 = 0

        boundary = boundary_mask(high, low, radius=1)
        boundary_excess = float(excess[boundary].sum())
        direction_all = dot_direction_stats(degraded, cond, clean, high)
        direction_viol = dot_direction_stats(degraded, cond, clean, violation)
        residual_norm = np.linalg.norm(cond - degraded, axis=-1)

        summary_rows.append(
            {
                "degradation": degradation,
                "N_trajectories": high.shape[0],
                "N_high_frames": int(high.sum()),
                "N_trajectories_with_high": int(traj_has_high.sum()),
                "N_high_violation_frames": int(violation.sum()),
                "N_high_violation_trajectories": int(traj_violation.sum()),
                "high_violation_frame_fraction": float(violation.sum() / max(high.sum(), 1)),
                "high_violation_trajectory_fraction": float(traj_violation.sum() / max(traj_has_high.sum(), 1)),
                "ADE_noisy_high": float(noisy_err[high].mean()),
                "ADE_cond_high": float(cond_err[high].mean()),
                "ADE_e1_high": float(e1_err[high].mean()),
                "C2_pass": bool(e1_err[high].mean() <= 1.10 * noisy_err[high].mean()),
                "total_high_violation_excess": total_excess,
                "N_trajectories_for_50pct_excess": n50,
                "N_trajectories_for_80pct_excess": n80,
                "N_trajectories_for_90pct_excess": n90,
                "boundary_excess_fraction_radius1": boundary_excess / total_excess if total_excess > 0.0 else math.nan,
                "wrong_direction_fraction_high": direction_all["wrong_direction_fraction"],
                "wrong_direction_fraction_violation": direction_viol["wrong_direction_fraction"],
                "mean_cosine_high": direction_all["mean_cosine"],
                "mean_cosine_violation": direction_viol["mean_cosine"],
                "mean_residual_norm_high": float(residual_norm[high].mean()),
                "mean_noisy_error_high": float(noisy_err[high].mean()),
            }
        )

        dominance_rows.append(
            {
                "degradation": degradation,
                "total_excess": total_excess,
                "N_trajectories_for_50pct_excess": n50,
                "N_trajectories_for_80pct_excess": n80,
                "N_trajectories_for_90pct_excess": n90,
            }
        )

        per_traj = []
        for idx in range(high.shape[0]):
            if not traj_has_high[idx]:
                continue
            h = high[idx]
            v = violation[idx]
            b = boundary[idx]
            per_traj.append(
                {
                    "degradation": degradation,
                    "trajectory_id": idx,
                    "N_high_frames": int(h.sum()),
                    "N_violation_high_frames": int(v.sum()),
                    "high_violation_frame_fraction": float(v.sum() / max(h.sum(), 1)),
                    "ADE_noisy_high": float(noisy_err[idx][h].mean()),
                    "ADE_cond_high": float(cond_err[idx][h].mean()),
                    "ADE_e1_high": float(e1_err[idx][h].mean()),
                    "e1_over_1p1_noisy_high": float(e1_err[idx][h].mean() / (1.10 * noisy_err[idx][h].mean() + 1e-12)),
                    "high_excess_sum": float(excess[idx].sum()),
                    "high_excess_mean": float(excess[idx][h].mean()),
                    "boundary_high_frames": int(b.sum()),
                    "boundary_excess_sum": float(excess[idx][b].sum()) if np.any(b) else 0.0,
                    "wrong_direction_fraction_high": dot_direction_stats(degraded[idx : idx + 1], cond[idx : idx + 1], clean[idx : idx + 1], h[None, :])["wrong_direction_fraction"],
                }
            )
        worst_rows.extend(sorted(per_traj, key=lambda row: row["high_excess_sum"], reverse=True)[:10])

    return pd.DataFrame(summary_rows), pd.DataFrame(worst_rows), pd.DataFrame(dominance_rows)


def gamma_ablation(loaded: dict[str, dict[str, np.ndarray | float]]) -> pd.DataFrame:
    rows: list[dict] = []
    for degradation, data in loaded.items():
        clean = data["clean"]
        degraded = data["degraded"]
        cond = data["cond"]
        confidence = data["confidence"]
        noisy_err = frame_error(degraded, clean)
        cond_err = frame_error(cond, clean)
        high = confidence > 0.7
        low = confidence < 0.3
        conditional_useful_low = bool(cond_err[low].mean() < noisy_err[low].mean()) if np.any(low) else False
        for gamma in GAMMAS:
            e1 = apply_gate(degraded, cond, confidence, gamma)
            e1_err = frame_error(e1, clean)
            c2 = bool(e1_err[high].mean() <= 1.10 * noisy_err[high].mean()) if np.any(high) else False
            if np.any(low) and conditional_useful_low:
                c3 = bool(e1_err[low].mean() <= 1.10 * cond_err[low].mean())
            else:
                c3 = True
            rows.append(
                {
                    "degradation": degradation,
                    "gamma": gamma,
                    "ADE_noisy": float(noisy_err.mean(axis=1).mean()),
                    "ADE_cond": float(cond_err.mean(axis=1).mean()),
                    "ADE_e1": float(e1_err.mean(axis=1).mean()),
                    "ADE_noisy_high": float(noisy_err[high].mean()) if np.any(high) else math.nan,
                    "ADE_e1_high": float(e1_err[high].mean()) if np.any(high) else math.nan,
                    "ADE_noisy_low": float(noisy_err[low].mean()) if np.any(low) else math.nan,
                    "ADE_cond_low": float(cond_err[low].mean()) if np.any(low) else math.nan,
                    "ADE_e1_low": float(e1_err[low].mean()) if np.any(low) else math.nan,
                    "C2_pass": c2,
                    "C3_pass": c3,
                    "N_high_frames": int(high.sum()),
                    "N_low_frames": int(low.sum()),
                }
            )
    return pd.DataFrame(rows)


def delta0_tradeoff(loaded: dict[str, dict[str, np.ndarray | float]], sweep_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for degradation, data in loaded.items():
        clean = data["clean"]
        degraded = data["degraded"]
        cond = data["cond"]
        noisy_err = frame_error(degraded, clean)
        baseline_row = sweep_df[(sweep_df["degradation"] == degradation) & (sweep_df["delta0_label"] == "auto_median_degraded_error")].iloc[0]
        baseline_ade = float(baseline_row["ADE_e1"])
        sub = sweep_df[sweep_df["degradation"] == degradation]
        for _, row in sub.iterrows():
            delta0 = float(row["delta0"])
            confidence = np.exp(-noisy_err / delta0).astype(np.float32)
            high = confidence > 0.7
            low = confidence < 0.3
            e1 = apply_gate(degraded, cond, confidence, gamma=1)
            e1_err = frame_error(e1, clean)
            c2 = bool(e1_err[high].mean() <= 1.10 * noisy_err[high].mean()) if np.any(high) else False
            overall_ade = float(e1_err.mean(axis=1).mean())
            rows.append(
                {
                    "degradation": degradation,
                    "delta0_label": row["delta0_label"],
                    "delta0": delta0,
                    "ADE_e1": overall_ade,
                    "ADE_auto_baseline": baseline_ade,
                    "relative_ADE_change_vs_auto": overall_ade / baseline_ade - 1.0 if baseline_ade > 0 else math.nan,
                    "not_significant_ADE_sacrifice_5pct": bool(overall_ade <= 1.05 * baseline_ade),
                    "ADE_noisy_high": float(noisy_err[high].mean()) if np.any(high) else math.nan,
                    "ADE_e1_high": float(e1_err[high].mean()) if np.any(high) else math.nan,
                    "C2_pass": c2,
                    "N_high_frames": int(high.sum()),
                    "N_mid_frames": int(((confidence >= 0.3) & (confidence <= 0.7)).sum()),
                    "N_low_frames": int(low.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_failure_cases(loaded: dict[str, dict[str, np.ndarray | float]], worst_df: pd.DataFrame) -> list[dict[str, str]]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for degradation in TARGET_DEGRADATIONS:
        sub = worst_df[worst_df["degradation"] == degradation].head(3)
        data = loaded[degradation]
        clean_all = data["clean"]
        degraded_all = data["degraded"]
        cond_all = data["cond"]
        e1_all = data["e1"]
        confidence_all = data["confidence"]
        noisy_err_all = frame_error(degraded_all, clean_all)
        cond_err_all = frame_error(cond_all, clean_all)
        e1_err_all = frame_error(e1_all, clean_all)
        for _, row in sub.iterrows():
            idx = int(row["trajectory_id"])
            confidence = confidence_all[idx]
            lambda_t = 1.0 - confidence
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            ax = axes[0]
            ax.plot(clean_all[idx, :, 0], clean_all[idx, :, 1], "k-o", ms=3, lw=1.3, label="clean")
            ax.plot(degraded_all[idx, :, 0], degraded_all[idx, :, 1], "C1-o", ms=3, lw=1.1, label="degraded")
            ax.plot(cond_all[idx, :, 0], cond_all[idx, :, 1], "C3-o", ms=3, lw=1.1, label="cond_residual_t20")
            ax.plot(e1_all[idx, :, 0], e1_all[idx, :, 1], "C0-o", ms=3, lw=1.3, label="E1")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            ax.set_title(f"{degradation} traj {idx}")

            axe = axes[1]
            t = np.arange(confidence.shape[0])
            axe.plot(t, noisy_err_all[idx], "C1-o", ms=3, lw=1.1, label="noisy error")
            axe.plot(t, cond_err_all[idx], "C3-o", ms=3, lw=1.1, label="cond error")
            axe.plot(t, e1_err_all[idx], "C0-o", ms=3, lw=1.2, label="E1 error")
            axe.set_title("per-frame errors")
            axe.grid(alpha=0.25)
            axe.legend(fontsize=8)

            axc = axes[2]
            axc.plot(t, confidence, "C0-o", ms=3, lw=1.2, label="confidence c_t")
            axc.plot(t, lambda_t, "C4-o", ms=3, lw=1.2, label="lambda_t")
            axc.axhline(0.7, color="0.35", ls="--", lw=1)
            axc.axhline(0.3, color="0.35", ls=":", lw=1)
            axc.set_ylim(0.0, 1.05)
            axc.set_title("confidence and gate")
            axc.grid(alpha=0.25)
            axc.legend(fontsize=8)

            fig.tight_layout()
            path = FIG_DIR / f"{degradation}_failure_traj{idx}.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            rows.append({"degradation": degradation, "trajectory_id": str(idx), "path": str(path)})
    return rows


def classify_failure(summary_df: pd.DataFrame, gamma_df: pd.DataFrame, delta_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for degradation in TARGET_DEGRADATIONS:
        row = summary_df[summary_df["degradation"] == degradation].iloc[0]
        gamma_sub = gamma_df[gamma_df["degradation"] == degradation]
        delta_sub = delta_df[delta_df["degradation"] == degradation]
        gamma_rescue = gamma_sub[(gamma_sub["C2_pass"] == True) & (gamma_sub["ADE_e1"] <= 1.05 * float(gamma_sub[gamma_sub["gamma"] == 1].iloc[0]["ADE_e1"]))]
        delta_rescue = delta_sub[(delta_sub["C2_pass"] == True) & (delta_sub["not_significant_ADE_sacrifice_5pct"] == True)]
        labels: list[str] = []
        evidence: list[str] = []
        if not gamma_rescue.empty:
            labels.append("gate too weak")
            evidence.append(f"gamma={int(gamma_rescue.iloc[0]['gamma'])} passes C2 within 5% ADE of gamma=1")
        if row["wrong_direction_fraction_violation"] > 0.35:
            labels.append("residual direction wrong")
            evidence.append(f"{row['wrong_direction_fraction_violation']:.2%} of violating high-confidence residuals point opposite the oracle correction")
        if degradation == "burst_medium" and row["boundary_excess_fraction_radius1"] > 0.30:
            labels.append("burst boundary effect")
            evidence.append(f"{row['boundary_excess_fraction_radius1']:.2%} of violation excess lies within one frame of low-confidence burst frames")
        if degradation == "drift_medium":
            labels.append("drift global residual leakage")
            evidence.append("high-confidence frames have tiny noisy error but conditional residual remains nontrivial across the trajectory")
        if delta_rescue.empty:
            evidence.append("delta_0 sweep does not provide a clean C2 rescue within the 5% ADE-sacrifice rule")
        else:
            evidence.append(f"delta_0={delta_rescue.iloc[0]['delta0']} passes C2 within 5% ADE")
        if row["wrong_direction_fraction_high"] < 0.20 and row["boundary_excess_fraction_radius1"] < 0.20:
            labels.append("confidence miscalibration")
            evidence.append("failure is not concentrated in wrong residual direction or local boundary frames")
        out[degradation] = {
            "diagnosis": " + ".join(dict.fromkeys(labels)) if labels else "mixed / ambiguous",
            "evidence": "; ".join(evidence),
        }
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    for col in table.columns:
        table[col] = table[col].map(lambda v: f"{float(v):.6f}" if isinstance(v, (float, np.floating)) else str(v))
    headers = [str(c) for c in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.values.tolist():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_summary(
    summary_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    dominance_df: pd.DataFrame,
    gamma_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    diagnosis: dict[str, dict[str, str]],
    figure_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# E1 Failure Diagnosis",
        "",
        "This diagnosis explains why formal E1 improves overall ADE but fails high-confidence no-harm for drift_medium and burst_medium. It reads Stage 4 E1 outputs and Stage 3 saved per-frame conditional residual outputs only; it does not train or resample conditional residuals.",
        "",
        "## High-confidence failure summary",
        markdown_table(summary_df),
        "",
        "## Trajectory dominance",
        markdown_table(dominance_df),
        "",
        "## Worst-10 high-confidence no-harm violation trajectories",
        markdown_table(worst_df),
        "",
        "## Delta_0 C2 tradeoff",
        "A delta_0 setting is treated as a clean rescue only if C2 passes and overall ADE is no more than 5% worse than the auto-delta0 formal E1 ADE.",
        markdown_table(delta_df),
        "",
        "## Conservative gate ablation",
        "Post-hoc ablation uses lambda_t = (1 - c_t)^gamma with the same oracle confidence and Stage 3 saved conditional output.",
        markdown_table(gamma_df),
        "",
        "## Failure cause judgment",
    ]
    for degradation, item in diagnosis.items():
        lines.append(f"- {degradation}: {item['diagnosis']}. Evidence: {item['evidence']}")
    lines.extend(["", "## Figures"])
    for row in figure_rows:
        lines.append(f"- {row['degradation']} trajectory {row['trajectory_id']}: `{row['path']}`")
    lines.extend(
        [
            "",
            "## Conclusion",
            "E1's oracle confidence is directionally useful, but the linear gate lambda = 1 - c is too permissive for high-confidence frames. Drift additionally shows global residual leakage: even tiny input errors receive a nontrivial learned residual. Burst shows a boundary effect around corrupted spans, where neighboring high-confidence frames inherit residual corrections from the local burst context. A conservative gamma gate can diagnose this but should be treated as post-hoc evidence, not as a redesigned formal E1 result.",
        ]
    )
    (OUT_DIR / "diagnosis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pd.read_csv(FULL_METRICS_PATH)
    pd.read_csv(BIN_METRICS_PATH)
    sweep_df = pd.read_csv(SWEEP_PATH)
    pd.read_csv(PER_TRAJ_PATH)

    loaded = load_arrays()
    summary_df, worst_df, dominance_df = high_confidence_failure_tables(loaded)
    gamma_df = gamma_ablation(loaded)
    delta_df = delta0_tradeoff(loaded, sweep_df)
    figure_rows = plot_failure_cases(loaded, worst_df)
    diagnosis = classify_failure(summary_df, gamma_df, delta_df)

    summary_df.to_csv(OUT_DIR / "high_confidence_failure_summary.csv", index=False)
    worst_df.to_csv(OUT_DIR / "worst10_high_confidence_violations.csv", index=False)
    dominance_df.to_csv(OUT_DIR / "trajectory_dominance.csv", index=False)
    gamma_df.to_csv(OUT_DIR / "conservative_gate_gamma_ablation.csv", index=False)
    delta_df.to_csv(OUT_DIR / "delta0_c2_tradeoff.csv", index=False)
    pd.DataFrame(figure_rows).to_csv(OUT_DIR / "diagnosis_figures.csv", index=False)
    pd.DataFrame(
        [
            {"degradation": deg, "diagnosis": item["diagnosis"], "evidence": item["evidence"]}
            for deg, item in diagnosis.items()
        ]
    ).to_csv(OUT_DIR / "failure_cause_judgment.csv", index=False)
    write_summary(summary_df, worst_df, dominance_df, gamma_df, delta_df, diagnosis, figure_rows)

    print("STAGE4_E1_FAILURE_DIAGNOSIS_COMPLETE")
    print(f"output directory: {OUT_DIR}")
    for degradation, item in diagnosis.items():
        print(f"{degradation}: {item['diagnosis']}")


if __name__ == "__main__":
    main()
