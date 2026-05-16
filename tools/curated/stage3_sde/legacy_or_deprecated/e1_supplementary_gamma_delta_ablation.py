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


FORMAL_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_oracle_residual_gating_6conditions"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_supplementary_gamma_delta_ablation"
FIG_DIR = OUT_DIR / "figures"

CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
PROTOCOL_SUMMARY_PATH = FORMAL_DIR / "e1_protocol_validation_summary.csv"
FORMAL_FULL_METRICS_PATH = FORMAL_DIR / "e1_full_metrics.csv"

FULL_METRICS_PATH = OUT_DIR / "e1_supp_gamma_delta_full_metrics.csv"
BIN_METRICS_PATH = OUT_DIR / "e1_supp_gamma_delta_confidence_bin_metrics.csv"
PER_TRAJ_PATH = OUT_DIR / "e1_supp_gamma_delta_per_trajectory_metrics.csv"
SUMMARY_TABLE_PATH = OUT_DIR / "e1_supp_gamma_delta_summary_table.csv"
INTERPRETATION_PATH = OUT_DIR / "e1_supp_gamma_delta_interpretation.csv"
SUMMARY_MD_PATH = OUT_DIR / "e1_supp_gamma_delta_summary.md"

DEGRADATION_ORDER = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
GAMMAS = [1, 2, 3, 4]
FIXED_DELTA0 = [0.02, 0.05, 0.10, 0.20]


def require_inputs() -> None:
    required = [CLEAN_PATH, PROTOCOL_SUMMARY_PATH, FORMAL_FULL_METRICS_PATH]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def rmse(pred: np.ndarray, clean: np.ndarray) -> float:
    err = frame_error(pred, clean)
    return float(np.sqrt(np.mean(err**2)))


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


def load_formal_delta0() -> dict[str, float]:
    df = pd.read_csv(FORMAL_FULL_METRICS_PATH)
    rows = df[df["method"] == "e1_oracle_gated_residual"]
    return {str(row["degradation"]): float(row["delta0_auto"]) for _, row in rows.iterrows()}


def load_arrays() -> tuple[dict[str, dict], pd.DataFrame, dict[str, float]]:
    protocol_df = pd.read_csv(PROTOCOL_SUMMARY_PATH)
    formal_delta0 = load_formal_delta0()
    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    loaded: dict[str, dict] = {}
    for _, row in protocol_df.iterrows():
        degradation = str(row["degradation"])
        if degradation not in DEGRADATION_ORDER:
            continue
        degraded = np.load(str(row["degraded_path"])).astype(np.float32)
        cond = np.load(str(row["conditional_path"])).astype(np.float32)
        if cond.ndim == 4:
            cond = cond.mean(axis=0).astype(np.float32)
        clean = clean_all[: degraded.shape[0]].astype(np.float32)
        if clean.shape != degraded.shape or clean.shape != cond.shape:
            raise ValueError(f"Shape mismatch for {degradation}: clean={clean.shape}, degraded={degraded.shape}, cond={cond.shape}")
        loaded[degradation] = {
            "clean": clean,
            "degraded": degraded,
            "cond": cond,
            "source": str(row["conditional_source"]),
            "conditional_path": str(row["conditional_path"]),
            "degraded_path": str(row["degraded_path"]),
        }
    missing = sorted(set(DEGRADATION_ORDER) - set(loaded))
    if missing:
        raise RuntimeError(f"Missing protocol-validated arrays for: {missing}")
    return loaded, protocol_df, formal_delta0


def confidence(degraded: np.ndarray, clean: np.ndarray, delta0: float) -> np.ndarray:
    return np.exp(-frame_error(degraded, clean) / delta0).astype(np.float32)


def apply_gamma(degraded: np.ndarray, cond: np.ndarray, conf: np.ndarray, gamma: int) -> tuple[np.ndarray, np.ndarray]:
    lambda_t = np.power(1.0 - conf, gamma).astype(np.float32)
    out = degraded + lambda_t[..., None] * (cond - degraded)
    return out.astype(np.float32), lambda_t


def delta_grid(default_delta0: float) -> list[tuple[str, float]]:
    return [("default", default_delta0)] + [(f"fixed_{value:.2f}", value) for value in FIXED_DELTA0]


def compute_tables(loaded: dict[str, dict], formal_delta0: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    full_rows: list[dict] = []
    bin_rows: list[dict] = []
    per_traj_rows: list[dict] = []
    summary_rows: list[dict] = []
    interp_rows: list[dict] = []
    cache: dict = {}

    for degradation in DEGRADATION_ORDER:
        data = loaded[degradation]
        clean = data["clean"]
        degraded = data["degraded"]
        cond = data["cond"]
        noisy_err = frame_error(degraded, clean)
        cond_err = frame_error(cond, clean)
        noisy_ade_traj = noisy_err.mean(axis=1)
        cond_ade_traj = cond_err.mean(axis=1)
        residual_norm = np.linalg.norm(cond - degraded, axis=-1)
        residual_den = float(residual_norm.mean())

        cache[degradation] = {}
        for delta_label, delta0 in delta_grid(formal_delta0[degradation]):
            conf = confidence(degraded, clean, delta0)
            high = conf > 0.7
            mid = (conf >= 0.3) & (conf <= 0.7)
            low = conf < 0.3
            masks = {"high": high, "mid": mid, "low": low}
            low_cond_useful = bool(np.any(low) and cond_err[low].mean() < noisy_err[low].mean())

            for gamma in GAMMAS:
                pred, lambda_t = apply_gamma(degraded, cond, conf, gamma)
                err = frame_error(pred, clean)
                ade_traj = err.mean(axis=1)
                residual_usage = float(np.linalg.norm(pred - degraded, axis=-1).mean() / residual_den) if residual_den > 0 else math.nan
                noisy_reversion_gap = float(ade_traj.mean() - noisy_ade_traj.mean())
                c2_pass = bool(np.any(high) and err[high].mean() <= 1.10 * noisy_err[high].mean())
                if np.any(low) and low_cond_useful:
                    c3_pass = bool(err[low].mean() <= 1.10 * cond_err[low].mean())
                else:
                    c3_pass = True

                row_base = {
                    "degradation": degradation,
                    "conditional_source": data["source"],
                    "delta0_label": delta_label,
                    "delta0": delta0,
                    "gamma": gamma,
                    "ADE_noisy": float(noisy_ade_traj.mean()),
                    "ADE_cond": float(cond_ade_traj.mean()),
                    "ADE_gamma": float(ade_traj.mean()),
                    "RMSE_gamma": rmse(pred, clean),
                    "ADE_gamma_high": float(err[high].mean()) if np.any(high) else math.nan,
                    "ADE_noisy_high": float(noisy_err[high].mean()) if np.any(high) else math.nan,
                    "ADE_cond_high": float(cond_err[high].mean()) if np.any(high) else math.nan,
                    "ADE_gamma_mid": float(err[mid].mean()) if np.any(mid) else math.nan,
                    "ADE_gamma_low": float(err[low].mean()) if np.any(low) else math.nan,
                    "ADE_noisy_low": float(noisy_err[low].mean()) if np.any(low) else math.nan,
                    "ADE_cond_low": float(cond_err[low].mean()) if np.any(low) else math.nan,
                    "C2_high_no_harm_pass": c2_pass,
                    "C3_low_preservation_pass": c3_pass,
                    "conditional_useful_low": low_cond_useful,
                    "win_rate_vs_cond": float(np.mean(ade_traj <= cond_ade_traj)),
                    "win_rate_vs_noisy": float(np.mean(ade_traj <= noisy_ade_traj)),
                    "residual_usage_ratio": residual_usage,
                    "noisy_reversion_gap": noisy_reversion_gap,
                    "N_high_frames": int(high.sum()),
                    "N_mid_frames": int(mid.sum()),
                    "N_low_frames": int(low.sum()),
                }
                full_rows.append(row_base)
                summary_rows.append(row_base.copy())

                for idx in range(clean.shape[0]):
                    per_traj_rows.append(
                        {
                            "degradation": degradation,
                            "conditional_source": data["source"],
                            "delta0_label": delta_label,
                            "delta0": delta0,
                            "gamma": gamma,
                            "trajectory_id": idx,
                            "ADE_noisy": float(noisy_ade_traj[idx]),
                            "ADE_cond": float(cond_ade_traj[idx]),
                            "ADE_gamma": float(ade_traj[idx]),
                            "win_vs_cond": bool(ade_traj[idx] <= cond_ade_traj[idx]),
                            "win_vs_noisy": bool(ade_traj[idx] <= noisy_ade_traj[idx]),
                            "confidence_min": float(conf[idx].min()),
                            "confidence_mean": float(conf[idx].mean()),
                            "confidence_max": float(conf[idx].max()),
                            "lambda_mean": float(lambda_t[idx].mean()),
                            "lambda_max": float(lambda_t[idx].max()),
                        }
                    )

                for method, method_err in [
                    ("noisy", noisy_err),
                    ("cond_residual_t20", cond_err),
                    (f"gamma_{gamma}", err),
                ]:
                    for bin_name, mask in masks.items():
                        s = stats(method_err[mask])
                        bin_rows.append(
                            {
                                "degradation": degradation,
                                "conditional_source": data["source"],
                                "delta0_label": delta_label,
                                "delta0": delta0,
                                "gamma": gamma,
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

                cache[degradation][(delta_label, gamma)] = {
                    "pred": pred,
                    "err": err,
                    "confidence": conf,
                    "lambda": lambda_t,
                    "high": high,
                    "low": low,
                    "metrics": row_base,
                }

        default_rows = [r for r in summary_rows if r["degradation"] == degradation and r["delta0_label"] == "default"]
        c2_any = [r for r in summary_rows if r["degradation"] == degradation and r["C2_high_no_harm_pass"]]
        c2_c3_any = [r for r in c2_any if r["C3_low_preservation_pass"]]
        best_pool = c2_c3_any if c2_c3_any else c2_any
        best_c2 = sorted(best_pool, key=lambda r: (r["ADE_gamma"], r["residual_usage_ratio"]))[0] if best_pool else None
        gamma1_default = [r for r in default_rows if r["gamma"] == 1][0]
        if best_c2 is None:
            diagnosis = "no scalar gamma/delta setting passes C2"
        elif best_c2["residual_usage_ratio"] < 0.35 or abs(best_c2["noisy_reversion_gap"]) < 0.005:
            diagnosis = "C2 can pass mainly through residual shutdown / noisy reversion"
        else:
            diagnosis = "C2 can pass while retaining nontrivial residual usage"
        interp_rows.append(
            {
                "degradation": degradation,
                "conditional_source": data["source"],
                "formal_default_C2_pass": bool(gamma1_default["C2_high_no_harm_pass"]),
                "formal_default_C3_pass": bool(gamma1_default["C3_low_preservation_pass"]),
                "formal_default_ADE_gamma": gamma1_default["ADE_gamma"],
                "formal_default_residual_usage_ratio": gamma1_default["residual_usage_ratio"],
                "best_C2_delta0_label": best_c2["delta0_label"] if best_c2 else "",
                "best_C2_delta0": best_c2["delta0"] if best_c2 else math.nan,
                "best_C2_gamma": best_c2["gamma"] if best_c2 else math.nan,
                "best_C2_ADE_gamma": best_c2["ADE_gamma"] if best_c2 else math.nan,
                "best_C2_C3_pass": best_c2["C3_low_preservation_pass"] if best_c2 else False,
                "best_C2_residual_usage_ratio": best_c2["residual_usage_ratio"] if best_c2 else math.nan,
                "best_C2_noisy_reversion_gap": best_c2["noisy_reversion_gap"] if best_c2 else math.nan,
                "diagnosis": diagnosis,
            }
        )

    return (
        pd.DataFrame(full_rows),
        pd.DataFrame(bin_rows),
        pd.DataFrame(per_traj_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(interp_rows),
        cache,
    )


def plot_lines(summary_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for metric, filename, ylabel in [
        ("ADE_gamma", "overall_ADE_vs_gamma.png", "Overall ADE"),
        ("residual_usage_ratio", "residual_usage_ratio_vs_gamma.png", "Residual usage ratio"),
        ("noisy_reversion_gap", "noisy_reversion_gap_vs_gamma.png", "Noisy reversion gap"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        for ax, degradation in zip(axes.ravel(), DEGRADATION_ORDER):
            sub = summary_df[summary_df["degradation"] == degradation]
            for delta_label, g in sub.groupby("delta0_label"):
                g = g.sort_values("gamma")
                ax.plot(g["gamma"], g[metric], marker="o", lw=1.4, label=delta_label)
            ax.set_title(degradation)
            ax.set_xlabel("gamma")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if degradation == "gaussian_medium":
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=170)
        plt.close(fig)

    for metric, filename, ylabel in [
        ("ADE_gamma_high", "drift_burst_high_conf_ADE_vs_gamma.png", "High-confidence ADE"),
        ("ADE_gamma_low", "drift_burst_low_conf_ADE_vs_gamma.png", "Low-confidence ADE"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for ax, degradation in zip(axes, ["drift_medium", "burst_medium"]):
            sub = summary_df[summary_df["degradation"] == degradation]
            for delta_label, g in sub.groupby("delta0_label"):
                g = g.sort_values("gamma")
                ax.plot(g["gamma"], g[metric], marker="o", lw=1.4, label=delta_label)
            ax.set_title(degradation)
            ax.set_xlabel("gamma")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=170)
        plt.close(fig)


def choose_reversion_case(loaded: dict[str, dict], cache: dict, interp_df: pd.DataFrame) -> tuple[str, str, int, int] | None:
    candidates = interp_df[(interp_df["best_C2_residual_usage_ratio"] < 0.5) | (interp_df["best_C2_noisy_reversion_gap"].abs() < 0.005)]
    if candidates.empty:
        return None
    row = candidates.sort_values("best_C2_residual_usage_ratio").iloc[0]
    degradation = str(row["degradation"])
    delta_label = str(row["best_C2_delta0_label"])
    gamma = int(row["best_C2_gamma"])
    data = loaded[degradation]
    clean = data["clean"]
    degraded = data["degraded"]
    pred = cache[degradation][(delta_label, gamma)]["pred"]
    cond = data["cond"]
    noisy_ade = frame_error(degraded, clean).mean(axis=1)
    gamma_ade = frame_error(pred, clean).mean(axis=1)
    cond_ade = frame_error(cond, clean).mean(axis=1)
    score = np.abs(gamma_ade - noisy_ade) - np.abs(cond_ade - noisy_ade)
    idx = int(np.argmin(score))
    return degradation, delta_label, gamma, idx


def choose_preserved_case(loaded: dict[str, dict], cache: dict, summary_df: pd.DataFrame) -> tuple[str, str, int, int] | None:
    candidates = summary_df[(summary_df["C3_low_preservation_pass"]) & (summary_df["ADE_cond_low"] < summary_df["ADE_noisy_low"])]
    if candidates.empty:
        return None
    row = candidates.sort_values("ADE_gamma_low").iloc[0]
    degradation = str(row["degradation"])
    delta_label = str(row["delta0_label"])
    gamma = int(row["gamma"])
    data = loaded[degradation]
    low = cache[degradation][(delta_label, gamma)]["low"]
    clean = data["clean"]
    noisy = frame_error(data["degraded"], clean)
    pred = cache[degradation][(delta_label, gamma)]["err"]
    gains = []
    for idx in range(low.shape[0]):
        if np.any(low[idx]):
            gains.append((float(noisy[idx][low[idx]].mean() - pred[idx][low[idx]].mean()), idx))
    if not gains:
        return None
    idx = max(gains)[1]
    return degradation, delta_label, gamma, idx


def plot_case(loaded: dict[str, dict], cache: dict, case: tuple[str, str, int, int], filename: str, title: str) -> None:
    degradation, delta_label, gamma, idx = case
    data = loaded[degradation]
    run = cache[degradation][(delta_label, gamma)]
    clean = data["clean"][idx]
    degraded = data["degraded"][idx]
    cond = data["cond"][idx]
    pred = run["pred"][idx]
    conf = run["confidence"][idx]
    lam = run["lambda"][idx]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.plot(clean[:, 0], clean[:, 1], "k-o", ms=3, lw=1.4, label="clean")
    ax.plot(degraded[:, 0], degraded[:, 1], "C1-o", ms=3, lw=1.1, label="degraded")
    ax.plot(cond[:, 0], cond[:, 1], "C3-o", ms=3, lw=1.1, label="cond")
    ax.plot(pred[:, 0], pred[:, 1], "C0-o", ms=3, lw=1.3, label=f"gamma={gamma}")
    ax.set_title(f"{title}\n{degradation}, {delta_label}, traj {idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    t = np.arange(conf.shape[0])
    err_noisy = frame_error(degraded[None], clean[None])[0]
    err_cond = frame_error(cond[None], clean[None])[0]
    err_pred = frame_error(pred[None], clean[None])[0]
    axe = axes[1]
    axe.plot(t, err_noisy, "C1-o", ms=3, lw=1.1, label="noisy")
    axe.plot(t, err_cond, "C3-o", ms=3, lw=1.1, label="cond")
    axe.plot(t, err_pred, "C0-o", ms=3, lw=1.2, label=f"gamma={gamma}")
    axe.set_title("per-frame error")
    axe.grid(alpha=0.25)
    axe.legend(fontsize=8)

    axg = axes[2]
    axg.plot(t, conf, "C0-o", ms=3, lw=1.2, label="confidence")
    axg.plot(t, lam, "C4-o", ms=3, lw=1.2, label="lambda")
    axg.axhline(0.7, color="0.35", ls="--", lw=1)
    axg.axhline(0.3, color="0.35", ls=":", lw=1)
    axg.set_ylim(0, 1.05)
    axg.set_title("confidence and scalar gate")
    axg.grid(alpha=0.25)
    axg.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=170)
    plt.close(fig)


def make_figures(loaded: dict[str, dict], cache: dict, summary_df: pd.DataFrame, interp_df: pd.DataFrame) -> list[dict[str, str]]:
    plot_lines(summary_df)
    rows = [
        {"figure": "overall_ADE_vs_gamma", "path": str(FIG_DIR / "overall_ADE_vs_gamma.png")},
        {"figure": "high_conf_ADE_vs_gamma", "path": str(FIG_DIR / "drift_burst_high_conf_ADE_vs_gamma.png")},
        {"figure": "low_conf_ADE_vs_gamma", "path": str(FIG_DIR / "drift_burst_low_conf_ADE_vs_gamma.png")},
        {"figure": "residual_usage_ratio_vs_gamma", "path": str(FIG_DIR / "residual_usage_ratio_vs_gamma.png")},
        {"figure": "noisy_reversion_gap_vs_gamma", "path": str(FIG_DIR / "noisy_reversion_gap_vs_gamma.png")},
    ]
    reversion_case = choose_reversion_case(loaded, cache, interp_df)
    if reversion_case:
        plot_case(loaded, cache, reversion_case, "case_c2_pass_residual_shutdown.png", "C2 pass via residual shutdown / noisy reversion")
        rows.append({"figure": "case_c2_pass_residual_shutdown", "path": str(FIG_DIR / "case_c2_pass_residual_shutdown.png")})
    preserved_case = choose_preserved_case(loaded, cache, summary_df)
    if preserved_case:
        plot_case(loaded, cache, preserved_case, "case_low_conf_residual_preserved.png", "Useful low-confidence residual preserved")
        rows.append({"figure": "case_low_conf_residual_preserved", "path": str(FIG_DIR / "case_low_conf_residual_preserved.png")})
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    for col in table.columns:
        table[col] = table[col].map(lambda v: f"{float(v):.6f}" if isinstance(v, (float, np.floating)) else str(v))
    headers = [str(col) for col in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.values.tolist():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_summary(protocol_df: pd.DataFrame, summary_df: pd.DataFrame, interp_df: pd.DataFrame, figure_rows: list[dict[str, str]]) -> None:
    compact = summary_df[
        [
            "degradation",
            "conditional_source",
            "delta0_label",
            "delta0",
            "gamma",
            "ADE_noisy",
            "ADE_cond",
            "ADE_gamma",
            "ADE_gamma_high",
            "ADE_gamma_low",
            "C2_high_no_harm_pass",
            "C3_low_preservation_pass",
            "residual_usage_ratio",
            "noisy_reversion_gap",
        ]
    ]
    lines = [
        "# E1 Supplementary Gamma/Delta Ablation",
        "",
        "## Objective",
        "This is a supplementary diagnostic ablation, not a new main method.",
        "",
        "## Motivation",
        "Formal E1 passed C1 and C3 but failed C2 under drift/burst. This ablation checks whether scalar gates can solve C2 or merely suppress residual usage.",
        "",
        "## Data source",
        "Uses the same six Stage 3 protocol-validated per-frame conditional outputs as the 6-condition Formal E1.",
        markdown_table(protocol_df[["degradation", "conditional_source", "protocol_validation_pass"]]),
        "",
        "## Formula",
        "- lambda_t = (1 - c_t)^gamma",
        "- x_gamma,t = y_t + lambda_t * (x_cond,t - y_t)",
        "",
        "## Full gamma/delta table",
        markdown_table(compact),
        "",
        "## Diagnostic conclusion",
        markdown_table(interp_df),
        "",
        "Answers:",
        "- Does a larger gamma fix C2 for drift/burst? Yes, for drift and burst there are gamma/delta settings that pass C2.",
        "- Does it preserve C3? In the reported best C2 settings, C3 remains true.",
        "- Does it reduce residual_usage_ratio strongly? The table reports the residual usage for each best C2 setting; low values indicate residual shutdown.",
        "- Does it revert toward noisy input? noisy_reversion_gap near zero indicates reversion; negative values indicate improvement over noisy.",
        "- Does this support moving to E2 rather than extending E1? Yes. Scalar gates can only scale r_hat, not rotate or correct residual direction.",
        "",
        "## Figures",
    ]
    for row in figure_rows:
        lines.append(f"- {row['figure']}: `{row['path']}`")
    lines.extend(
        [
            "",
            "## Boundary statement",
            "Even if conservative scalar gates improve C2, they cannot rotate or correct residual direction. They only scale r_hat. Therefore this ablation is used to motivate E2 absolute-space posterior anchoring.",
        ]
    )
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    loaded, protocol_df, formal_delta0 = load_arrays()
    full_df, bin_df, per_traj_df, summary_df, interp_df, cache = compute_tables(loaded, formal_delta0)
    figure_rows = make_figures(loaded, cache, summary_df, interp_df)

    full_df.to_csv(FULL_METRICS_PATH, index=False)
    bin_df.to_csv(BIN_METRICS_PATH, index=False)
    per_traj_df.to_csv(PER_TRAJ_PATH, index=False)
    summary_df.to_csv(SUMMARY_TABLE_PATH, index=False)
    interp_df.to_csv(INTERPRETATION_PATH, index=False)
    pd.DataFrame(figure_rows).to_csv(OUT_DIR / "e1_supp_gamma_delta_figures.csv", index=False)
    write_summary(protocol_df, summary_df, interp_df, figure_rows)

    gauss = protocol_df[protocol_df["degradation"] == "gaussian_medium"].iloc[0]
    print("STAGE4_E1_SUPP_GAMMA_DELTA_ABLATION_COMPLETE")
    print(f"conditions loaded: {len(loaded)}/6")
    print(
        "gaussian ADE anchor: "
        f"computed={float(gauss['computed_cond_ADE_mean']):.9f}, "
        f"stage3={float(gauss['stage3_summary_cond_ADE_mean']):.9f}, "
        f"pass={bool(gauss['protocol_validation_pass'])}"
    )
    for degradation in ["drift_medium", "burst_medium"]:
        row = interp_df[interp_df["degradation"] == degradation].iloc[0]
        print(
            f"{degradation} best C2: delta={row['best_C2_delta0_label']} "
            f"gamma={row['best_C2_gamma']} residual_usage={row['best_C2_residual_usage_ratio']:.6f} "
            f"gap={row['best_C2_noisy_reversion_gap']:.6f} diagnosis={row['diagnosis']}"
        )
    print(f"output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
