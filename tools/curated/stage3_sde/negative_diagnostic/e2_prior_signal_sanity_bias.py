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


DEGRADATION = "bias_medium"
EPS = 1e-12

CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
PROTOCOL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "stage4"
    / "e1_oracle_residual_gating_6conditions"
    / "e1_protocol_validation_summary.csv"
)
E2_TRAJ_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "stage4"
    / "e2_min_absolute_posterior_anchoring"
    / "e2_min_optimized_trajectories.npz"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_prior_signal_sanity_bias"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "bias_prior_signal_summary.md"
METRICS_PATH = OUT_DIR / "bias_prior_signal_metrics.csv"
PER_TRAJ_PATH = OUT_DIR / "bias_prior_signal_per_trajectory.csv"


def require_inputs() -> None:
    missing = [path for path in [CLEAN_PATH, PROTOCOL_SUMMARY_PATH] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str], dict[str, np.ndarray]]:
    require_inputs()
    protocol = pd.read_csv(PROTOCOL_SUMMARY_PATH)
    rows = protocol[protocol["degradation"] == DEGRADATION]
    if rows.empty:
        raise RuntimeError(f"Missing protocol row for {DEGRADATION}")
    row = rows.iloc[0]
    if not bool(row.get("protocol_validation_pass", True)):
        raise RuntimeError(f"Protocol validation is false for {DEGRADATION}")

    degraded = np.load(str(row["degraded_path"])).astype(np.float32)
    cond = np.load(str(row["conditional_path"])).astype(np.float32)
    if cond.ndim == 4:
        cond = cond.mean(axis=0).astype(np.float32)
    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    clean = clean_all[: degraded.shape[0]].astype(np.float32)
    if clean.shape != degraded.shape or clean.shape != cond.shape:
        raise ValueError(f"Shape mismatch: clean={clean.shape}, degraded={degraded.shape}, cond={cond.shape}")

    optional: dict[str, np.ndarray] = {}
    if E2_TRAJ_PATH.is_file():
        z = np.load(E2_TRAJ_PATH)
        for key in ["formal_e1", "V1_abs_anchor_smooth", "V2_uniform_cond_motion", "V3_conf_mod_cond_motion"]:
            zkey = f"{DEGRADATION}__{key}"
            if zkey in z.files:
                optional[key] = z[zkey].astype(np.float32)

    provenance = {
        "degradation": DEGRADATION,
        "conditional_source": str(row["conditional_source"]),
        "conditional_path": str(row["conditional_path"]),
        "degraded_path": str(row["degraded_path"]),
        "clean_path": str(CLEAN_PATH),
        "shape_clean": str(tuple(clean.shape)),
        "shape_degraded": str(tuple(degraded.shape)),
        "shape_cond": str(tuple(cond.shape)),
    }
    return clean, degraded, cond, provenance, optional


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def stats(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    names = ["mean", "std", "median", "min", "max", "p25", "p75"]
    if vals.size == 0:
        return {f"{prefix}{name}": math.nan for name in names}
    return {
        f"{prefix}mean": float(vals.mean()),
        f"{prefix}std": float(vals.std()),
        f"{prefix}median": float(np.median(vals)),
        f"{prefix}min": float(vals.min()),
        f"{prefix}max": float(vals.max()),
        f"{prefix}p25": float(np.percentile(vals, 25)),
        f"{prefix}p75": float(np.percentile(vals, 75)),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    valid = (an > EPS) & (bn > EPS)
    out = np.full(an.shape, np.nan, dtype=np.float64)
    out[valid] = np.sum(a[valid] * b[valid], axis=-1) / (an[valid] * bn[valid])
    return out, valid


def centered_shape_ade(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    err = pred - clean
    offset = err.mean(axis=1, keepdims=True)
    centered = err - offset
    return np.linalg.norm(centered, axis=-1).mean(axis=1)


def compute_metrics(clean: np.ndarray, degraded: np.ndarray, cond: np.ndarray, optional: dict[str, np.ndarray]):
    r = cond - degraded
    oracle = clean - degraded
    noisy_err = frame_error(degraded, clean)
    cond_err = frame_error(cond, clean)
    ade_noisy_traj = noisy_err.mean(axis=1)
    ade_cond_traj = cond_err.mean(axis=1)

    offset_noisy = (degraded - clean).mean(axis=1)
    offset_cond = (cond - clean).mean(axis=1)
    offset_noisy_norm = np.linalg.norm(offset_noisy, axis=-1)
    offset_cond_norm = np.linalg.norm(offset_cond, axis=-1)

    frame_cos, frame_valid = cosine(r.reshape(-1, 2), oracle.reshape(-1, 2))
    mean_r = r.mean(axis=1)
    mean_oracle = oracle.mean(axis=1)
    global_cos, global_valid = cosine(mean_r, mean_oracle)
    global_ratio = np.linalg.norm(mean_r, axis=-1) / np.maximum(np.linalg.norm(mean_oracle, axis=-1), EPS)

    shape_noisy = centered_shape_ade(degraded, clean)
    shape_cond = centered_shape_ade(cond, clean)

    per_rows = []
    for idx in range(clean.shape[0]):
        per_rows.append(
            {
                "trajectory_id": idx,
                "ADE_noisy": float(ade_noisy_traj[idx]),
                "ADE_cond": float(ade_cond_traj[idx]),
                "ADE_improvement_noisy_minus_cond": float(ade_noisy_traj[idx] - ade_cond_traj[idx]),
                "offset_noisy_x": float(offset_noisy[idx, 0]),
                "offset_noisy_y": float(offset_noisy[idx, 1]),
                "offset_cond_x": float(offset_cond[idx, 0]),
                "offset_cond_y": float(offset_cond[idx, 1]),
                "offset_norm_noisy": float(offset_noisy_norm[idx]),
                "offset_norm_cond": float(offset_cond_norm[idx]),
                "offset_norm_improvement": float(offset_noisy_norm[idx] - offset_cond_norm[idx]),
                "offset_norm_rel_improvement": float((offset_noisy_norm[idx] - offset_cond_norm[idx]) / max(offset_noisy_norm[idx], EPS)),
                "global_correction_cosine": float(global_cos[idx]),
                "global_correction_ratio": float(global_ratio[idx]),
                "shape_centered_ADE_noisy": float(shape_noisy[idx]),
                "shape_centered_ADE_cond": float(shape_cond[idx]),
                "shape_centered_ADE_improvement": float(shape_noisy[idx] - shape_cond[idx]),
            }
        )
    per_df = pd.DataFrame(per_rows)

    metrics = {
        "degradation": DEGRADATION,
        "N_trajectories": int(clean.shape[0]),
        "T": int(clean.shape[1]),
        "ADE_noisy": float(ade_noisy_traj.mean()),
        "ADE_cond": float(ade_cond_traj.mean()),
        "ADE_improvement_noisy_minus_cond": float(ade_noisy_traj.mean() - ade_cond_traj.mean()),
        "ADE_relative_improvement": float((ade_noisy_traj.mean() - ade_cond_traj.mean()) / ade_noisy_traj.mean()),
        "offset_norm_noisy_mean": float(offset_noisy_norm.mean()),
        "offset_norm_cond_mean": float(offset_cond_norm.mean()),
        "offset_norm_improvement": float(offset_noisy_norm.mean() - offset_cond_norm.mean()),
        "offset_norm_relative_improvement": float((offset_noisy_norm.mean() - offset_cond_norm.mean()) / offset_noisy_norm.mean()),
        "offset_reduced_fraction_trajectories": float(np.mean(offset_cond_norm < offset_noisy_norm)),
        "shape_centered_ADE_noisy_mean": float(shape_noisy.mean()),
        "shape_centered_ADE_cond_mean": float(shape_cond.mean()),
        "shape_centered_ADE_improvement": float(shape_noisy.mean() - shape_cond.mean()),
        "frame_cosine_valid_frames": int(frame_valid.sum()),
        "frame_cosine_total_frames": int(frame_valid.size),
        "frame_cosine_mean": float(np.nanmean(frame_cos)),
        "frame_cosine_median": float(np.nanmedian(frame_cos)),
        "frame_cosine_p25": float(np.nanpercentile(frame_cos, 25)),
        "frame_cosine_p75": float(np.nanpercentile(frame_cos, 75)),
        "frame_cosine_fraction_gt_0": float(np.nanmean(frame_cos > 0.0)),
        "frame_cosine_fraction_gt_0p5": float(np.nanmean(frame_cos > 0.5)),
        "frame_cosine_fraction_lt_0": float(np.nanmean(frame_cos < 0.0)),
        "global_cosine_valid_trajectories": int(global_valid.sum()),
        "global_cosine_mean": float(np.nanmean(global_cos)),
        "global_cosine_median": float(np.nanmedian(global_cos)),
        "global_cosine_p25": float(np.nanpercentile(global_cos, 25)),
        "global_cosine_p75": float(np.nanpercentile(global_cos, 75)),
        "global_ratio_mean": float(np.nanmean(global_ratio)),
        "global_ratio_median": float(np.nanmedian(global_ratio)),
        "global_ratio_p25": float(np.nanpercentile(global_ratio, 25)),
        "global_ratio_p75": float(np.nanpercentile(global_ratio, 75)),
    }

    for name, pred in optional.items():
        err = frame_error(pred, clean).mean(axis=1)
        off = np.linalg.norm((pred - clean).mean(axis=1), axis=-1)
        metrics[f"{name}_ADE"] = float(err.mean())
        metrics[f"{name}_offset_norm_mean"] = float(off.mean())

    return metrics, per_df, {
        "r": r,
        "oracle": oracle,
        "noisy_err": noisy_err,
        "cond_err": cond_err,
        "frame_cos": frame_cos.reshape(clean.shape[:2]),
        "global_cos": global_cos,
        "global_ratio": global_ratio,
        "offset_noisy_norm": offset_noisy_norm,
        "offset_cond_norm": offset_cond_norm,
    }


def choose_representatives(per_df: pd.DataFrame) -> list[tuple[str, int]]:
    candidates = [
        ("best_ADE_improvement", int(per_df["ADE_improvement_noisy_minus_cond"].idxmax())),
        ("largest_offset_reduction", int(per_df["offset_norm_improvement"].idxmax())),
        ("worst_offset_regression", int(per_df["offset_norm_improvement"].idxmin())),
        ("highest_global_alignment", int(per_df["global_correction_cosine"].idxmax())),
        ("median_global_alignment", int((per_df["global_correction_cosine"] - per_df["global_correction_cosine"].median()).abs().idxmin())),
    ]
    seen: set[int] = set()
    reps: list[tuple[str, int]] = []
    for label, idx in candidates:
        if idx in seen:
            continue
        seen.add(idx)
        reps.append((label, idx))
        if len(reps) >= 5:
            break
    return reps


def plot_case(
    clean: np.ndarray,
    degraded: np.ndarray,
    cond: np.ndarray,
    aux: dict[str, np.ndarray],
    per_df: pd.DataFrame,
    label: str,
    idx: int,
    path: Path,
) -> None:
    x_star = clean[idx]
    y = degraded[idx]
    x_cond = cond[idx]
    r = aux["r"][idx]
    oracle = aux["oracle"][idx]
    t = np.arange(x_star.shape[0])
    step = 3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    ax = axes[0]
    ax.plot(x_star[:, 0], x_star[:, 1], "k-o", lw=1.4, ms=3, label="clean target")
    ax.plot(y[:, 0], y[:, 1], "C1-o", lw=1.1, ms=3, label="degraded input")
    ax.plot(x_cond[:, 0], x_cond[:, 1], "C3-o", lw=1.1, ms=3, label="Stage 3 cond")
    ax.quiver(
        y[::step, 0],
        y[::step, 1],
        r[::step, 0],
        r[::step, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="C3",
        width=0.004,
        alpha=0.75,
        label="residual r",
    )
    ax.quiver(
        y[::step, 0],
        y[::step, 1],
        oracle[::step, 0],
        oracle[::step, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="C0",
        width=0.003,
        alpha=0.65,
        label="oracle correction",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{label}: trajectory {idx}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    axe = axes[1]
    axe.plot(t, aux["noisy_err"][idx], "C1-o", lw=1.2, ms=3, label="noisy error")
    axe.plot(t, aux["cond_err"][idx], "C3-o", lw=1.2, ms=3, label="cond error")
    axe2 = axe.twinx()
    axe2.plot(t, aux["frame_cos"][idx], "C0--", lw=1.0, label="frame cosine")
    axe.set_xlabel("t")
    axe.set_ylabel("error")
    axe2.set_ylabel("cosine")
    axe2.set_ylim(-1.05, 1.05)
    axe.grid(alpha=0.25)
    lines, labels = axe.get_legend_handles_labels()
    lines2, labels2 = axe2.get_legend_handles_labels()
    axe.legend(lines + lines2, labels + labels2, fontsize=7)
    row = per_df.iloc[idx]
    axe.set_title(
        f"ADE noisy={row['ADE_noisy']:.4f}, cond={row['ADE_cond']:.4f}\n"
        f"offset noisy={row['offset_norm_noisy']:.4f}, cond={row['offset_norm_cond']:.4f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_figures(clean: np.ndarray, degraded: np.ndarray, cond: np.ndarray, aux: dict[str, np.ndarray], per_df: pd.DataFrame) -> list[dict[str, str]]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, idx in choose_representatives(per_df):
        path = FIG_DIR / f"{label}_traj{idx}.png"
        plot_case(clean, degraded, cond, aux, per_df, label, idx, path)
        rows.append({"case": label, "trajectory_id": idx, "path": str(path)})
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    for col in table.columns:
        table[col] = table[col].map(lambda v: f"{float(v):.6f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for row in table.values.tolist():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def classify_signal(metrics: dict[str, float]) -> dict[str, str]:
    ade_rel = metrics["ADE_relative_improvement"]
    offset_rel = metrics["offset_norm_relative_improvement"]
    frame_cos = metrics["frame_cosine_mean"]
    global_cos = metrics["global_cosine_mean"]
    if ade_rel > 0.01 and offset_rel > 0.05 and global_cos > 0.5:
        signal = "clear bias-correcting signal"
        h5 = "keep H5 as a core E2-DPS target"
    elif offset_rel > 0.01 and global_cos > 0.25:
        signal = "weak bias-correcting signal"
        h5 = "keep H5, but treat it as high risk"
    else:
        signal = "very weak or absent bias-correcting signal"
        h5 = "downgrade E2-DPS H5 to known limitation / future work unless DPS adds a stronger absolute-position likelihood"
    if metrics["shape_centered_ADE_improvement"] > metrics["offset_norm_improvement"]:
        source = "mostly local shape correction"
    elif metrics["offset_norm_improvement"] > 0:
        source = "some global offset correction"
    else:
        source = "neither reliable shape nor global offset correction"
    return {"signal": signal, "h5_recommendation": h5, "correction_source": source}


def write_summary(metrics: dict[str, float], per_df: pd.DataFrame, provenance: dict[str, str], figure_rows: list[dict[str, str]], signal: dict[str, str]) -> None:
    key_metrics = pd.DataFrame([metrics]).T.reset_index()
    key_metrics.columns = ["metric", "value"]
    key_metrics = key_metrics[key_metrics["metric"].isin(
        [
            "ADE_noisy",
            "ADE_cond",
            "ADE_improvement_noisy_minus_cond",
            "ADE_relative_improvement",
            "offset_norm_noisy_mean",
            "offset_norm_cond_mean",
            "offset_norm_improvement",
            "offset_norm_relative_improvement",
            "offset_reduced_fraction_trajectories",
            "shape_centered_ADE_noisy_mean",
            "shape_centered_ADE_cond_mean",
            "shape_centered_ADE_improvement",
            "frame_cosine_mean",
            "frame_cosine_median",
            "frame_cosine_p25",
            "frame_cosine_p75",
            "frame_cosine_fraction_gt_0",
            "frame_cosine_fraction_gt_0p5",
            "frame_cosine_fraction_lt_0",
            "global_cosine_mean",
            "global_cosine_median",
            "global_ratio_mean",
            "global_ratio_median",
        ]
    )]
    provenance_df = pd.DataFrame([provenance])
    fig_df = pd.DataFrame(figure_rows)
    top_cases = per_df.sort_values("offset_norm_improvement", ascending=False).head(5)[
        [
            "trajectory_id",
            "ADE_noisy",
            "ADE_cond",
            "offset_norm_noisy",
            "offset_norm_cond",
            "offset_norm_improvement",
            "global_correction_cosine",
            "global_correction_ratio",
        ]
    ]

    lines = [
        "# Bias Prior-Signal Sanity Check",
        "",
        "## Data",
        "This diagnostic reads existing Stage 3 protocol-validated per-frame conditional outputs only. It does not train, resample, run SDEdit, or modify Stage 3 files.",
        "",
        markdown_table(provenance_df),
        "",
        "## Key Metrics",
        markdown_table(key_metrics),
        "",
        "## Direct Answers",
        f"1. Stage 3 cond_residual_t20 improves ADE on bias_medium: {'yes' if metrics['ADE_cond'] < metrics['ADE_noisy'] else 'no'}; ADE_noisy={metrics['ADE_noisy']:.6f}, ADE_cond={metrics['ADE_cond']:.6f}.",
        f"2. It reduces absolute offset error: {'yes, but weakly' if metrics['offset_norm_cond_mean'] < metrics['offset_norm_noisy_mean'] else 'no'}; offset_noisy={metrics['offset_norm_noisy_mean']:.6f}, offset_cond={metrics['offset_norm_cond_mean']:.6f}.",
        f"3. Residual direction alignment is {signal['signal']}: frame cosine mean={metrics['frame_cosine_mean']:.6f}, median={metrics['frame_cosine_median']:.6f}; global cosine mean={metrics['global_cosine_mean']:.6f}, median={metrics['global_cosine_median']:.6f}.",
        f"4. Correction source: {signal['correction_source']}.",
        f"5. E2-DPS prior-signal judgment: {signal['signal']}.",
        f"6. H5 recommendation: {signal['h5_recommendation']}.",
        "",
        "## Offset-Reduction Cases",
        markdown_table(top_cases),
        "",
        "## Figures",
    ]
    for row in figure_rows:
        lines.append(f"- {row['case']} trajectory {row['trajectory_id']}: {row['path']}")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean, degraded, cond, provenance, optional = load_arrays()
    metrics, per_df, aux = compute_metrics(clean, degraded, cond, optional)
    signal = classify_signal(metrics)
    figure_rows = make_figures(clean, degraded, cond, aux, per_df)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
    per_df.to_csv(PER_TRAJ_PATH, index=False)
    pd.DataFrame(figure_rows).to_csv(OUT_DIR / "bias_prior_signal_figures.csv", index=False)
    write_summary(metrics, per_df, provenance, figure_rows, signal)

    print("STAGE4_E2_PRIOR_SIGNAL_SANITY_BIAS_COMPLETE")
    print(f"ADE_noisy: {metrics['ADE_noisy']:.9f}")
    print(f"ADE_cond: {metrics['ADE_cond']:.9f}")
    print(f"offset_norm_noisy: {metrics['offset_norm_noisy_mean']:.9f}")
    print(f"offset_norm_cond: {metrics['offset_norm_cond_mean']:.9f}")
    print(f"frame_cosine_mean: {metrics['frame_cosine_mean']:.9f}")
    print(f"frame_cosine_median: {metrics['frame_cosine_median']:.9f}")
    print(f"global_correction_cosine_mean: {metrics['global_cosine_mean']:.9f}")
    print(f"global_correction_cosine_median: {metrics['global_cosine_median']:.9f}")
    print(f"bias_correcting_signal: {signal['signal']}")
    print(f"H5_recommendation: {signal['h5_recommendation']}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"metrics: {METRICS_PATH}")
    print(f"per_trajectory: {PER_TRAJ_PATH}")
    print(f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
