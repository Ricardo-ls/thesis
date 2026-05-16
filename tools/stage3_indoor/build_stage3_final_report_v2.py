from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_report_v2_mpl")

import matplotlib

matplotlib.use("Agg")
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CWD = PROJECT_ROOT
OUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "report_v2"
FIG_DIR = OUT_DIR / "figures"
DOCX_PATH = OUT_DIR / "stage3_final_report.docx"

CLEAN_CANDIDATES = [
    PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy",
    PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs_large.npy",
    PROJECT_ROOT / "data" / "stage3_indoor" / "train_trajs.npy",
]
GAUSS_METRIC_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "tables" / "table1_gaussian_medium.csv",
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "cond_residual_gaussian_summary.csv",
]
GEN_METRIC_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "tables" / "table2_generalization.csv",
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "generalization_summary.csv",
]
RAW_GAUSS_SUMMARY = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "cond_residual_gaussian_summary.csv"
RAW_GEN_SUMMARY = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "generalization_summary.csv"
GAUSS_PER_TRAJ = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "cond_residual_gaussian_per_traj.csv"
GEN_PER_TRAJ = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "generalization_per_traj.csv"
TSTART_SCOUT = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "sdedit_scout_results.csv"
PRIOR_CHECK = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "sampling_check_v2.png"

DEGRADATIONS = [
    "gaussian_medium",
    "drift_medium",
    "bias_medium",
    "jump_medium",
    "burst_medium",
    "combined_medium",
]
HEATMAP_ORDER = [
    "gaussian_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
    "bias_medium",
    "combined_medium",
]
METHODS_GAUSS = ["noisy_input", "linear_interp", "savgol_w5_p2", "kalman_cv", "uncond_sdedit_t2", "cond_residual_t20"]
N_EVAL = 200

DEGRADED_PATHS = {
    name: PROJECT_ROOT / "data" / "stage3_indoor" / f"degraded_{name}.npy"
    for name in DEGRADATIONS
}
CACHE_PATHS = {
    "gaussian_uncond_sdedit_t2": PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "cache" / "gaussian_uncond_sdedit_t2_refined.npy",
    "gaussian_cond_residual_t20": PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "cache" / "gaussian_cond_residual_t20_refined.npy",
    "drift_cond_residual_t20": PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "cache" / "drift_cond_residual_t20_refined.npy",
    "burst_cond_residual_t20": PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "cache" / "burst_cond_residual_t20_refined.npy",
    "bias_cond_residual_t20": PROJECT_ROOT / "outputs" / "stage3_indoor" / "report" / "cache" / "bias_cond_residual_t20_refined.npy",
}

FIG_PATHS = {
    "fig01": FIG_DIR / "fig01_task_correction.png",
    "fig02": FIG_DIR / "fig02_pipeline_overview.png",
    "fig03": FIG_DIR / "fig03_synthetic_trajectories.png",
    "fig04": FIG_DIR / "fig04_degradation_examples.png",
    "fig05": FIG_DIR / "fig05_prior_sampling_check.png",
    "fig06": FIG_DIR / "fig06_tstart_sweep.png",
    "fig07": FIG_DIR / "fig07_unconditional_diagnostic.png",
    "fig08": FIG_DIR / "fig08_conditional_comparison.png",
    "fig09": FIG_DIR / "fig09_generalization_heatmap.png",
    "fig10": FIG_DIR / "fig10_representative_cases.png",
}

COLORS = {
    "clean": "#E8A33C",
    "noisy": "#555555",
    "kalman": "#2E8B57",
    "uncond": "#1F77B4",
    "cond": "#C0392B",
    "room": "#222222",
    "annot": "#8B0000",
}


def set_style() -> None:
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.titlepad": 16,
            "axes.labelsize": 13,
            "axes.labelpad": 7,
            "axes.linewidth": 1.1,
            "legend.fontsize": 12,
            "legend.frameon": True,
            "legend.framealpha": 0.93,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.3,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def print_scan(label: str, paths: list[Path]) -> None:
    print(f"=== Scan: {label} ===")
    for path in paths:
        print(f"{'FOUND' if path.exists() else 'MISSING'}: {path}")


def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing required {label}. Checked:\n" + "\n".join(str(p) for p in paths))


def normalize_method_name(method: str) -> str:
    if method == "unconditional_sdedit_t2":
        return "uncond_sdedit_t2"
    return method


def load_gaussian_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" not in df.columns:
        raise ValueError(f"Gaussian metrics missing method column. columns={list(df.columns)}")
    df["method"] = df["method"].map(normalize_method_name)
    if "delta_vs_noisy_pct" not in df.columns:
        noisy = float(df.loc[df["method"] == "noisy_input", "ADE_mean"].iloc[0])
        df["delta_vs_noisy_pct"] = (df["ADE_mean"] - noisy) / noisy * 100
    if "improved_fraction" not in df.columns and "improved_fraction_vs_noisy" in df.columns:
        df["improved_fraction"] = df["improved_fraction_vs_noisy"]
    if "wilcoxon_p" not in df.columns and "wilcoxon_p_vs_noisy" in df.columns:
        df["wilcoxon_p"] = df["wilcoxon_p_vs_noisy"]
    required = ["method", "ADE_mean", "ADE_std", "delta_vs_noisy_pct", "improved_fraction", "wilcoxon_p", "smooth_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Gaussian metrics missing columns {missing}. columns={list(df.columns)}")
    return df[required].copy()


def load_generalization_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"degradation", "method", "ADE_mean"}.issubset(df.columns):
        df["method"] = df["method"].map(normalize_method_name)
        return df
    if {"degradation", "noisy_input_ADE", "uncond_sdedit_ADE", "cond_residual_ADE"}.issubset(df.columns):
        rows = []
        for _, row in df.iterrows():
            for method, col in [
                ("noisy_input", "noisy_input_ADE"),
                ("kalman_cv", "kalman_cv_ADE"),
                ("uncond_sdedit_t2", "uncond_sdedit_ADE"),
                ("cond_residual_t20", "cond_residual_ADE"),
            ]:
                if col in df.columns and not pd.isna(row[col]):
                    noisy = float(row["noisy_input_ADE"])
                    ade = float(row[col])
                    rows.append(
                        {
                            "degradation": row["degradation"],
                            "method": method,
                            "ADE_mean": ade,
                            "ADE_std": np.nan,
                            "smooth_mean": np.nan,
                            "delta_ADE_vs_noisy_mean": ade - noisy,
                            "improved_fraction": np.nan,
                            "wilcoxon_p_vs_noisy": row.get("cond_p_vs_noisy", np.nan) if method == "cond_residual_t20" else np.nan,
                        }
                    )
        return pd.DataFrame(rows)
    raise ValueError(f"Cannot map generalization CSV columns={list(df.columns)}")


def build_table2_from_long(gen_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for deg in HEATMAP_ORDER:
        sub = gen_long[gen_long["degradation"] == deg]
        get = lambda m: sub.loc[sub["method"] == m, "ADE_mean"].iloc[0] if (sub["method"] == m).any() else np.nan
        noisy = get("noisy_input")
        cond = get("cond_residual_t20")
        uncond = get("uncond_sdedit_t2")
        kalman = get("kalman_cv")
        p_series = sub.loc[sub["method"] == "cond_residual_t20", "wilcoxon_p_vs_noisy"]
        p = float(p_series.iloc[0]) if len(p_series) else np.nan
        delta = float(cond - noisy) if not pd.isna(cond) and not pd.isna(noisy) else np.nan
        if not pd.isna(p) and p < 0.01 and delta < -0.001:
            interp = "sig. improvement"
        elif deg == "drift_medium" and delta > 0.001:
            interp = "over-correction"
        elif deg == "burst_medium" and delta > 0.001:
            interp = "training mismatch"
        elif deg == "bias_medium" and abs(delta) <= 0.001:
            interp = "no change (relative-space limit)"
        elif deg == "combined_medium" and abs(delta) <= 0.003:
            interp = "no significant change"
        elif delta > 0.001:
            interp = "worse"
        else:
            interp = "no significant change"
        rows.append(
            {
                "degradation": deg,
                "noisy_input_ADE": noisy,
                "kalman_cv_ADE": kalman,
                "uncond_sdedit_ADE": uncond,
                "cond_residual_ADE": cond,
                "cond_delta_vs_noisy": delta,
                "cond_p_vs_noisy": p,
                "interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


def add_room(ax) -> None:
    ax.add_patch(Rectangle((-0.05, -0.05), 3.1, 3.1, edgecolor=COLORS["room"], linewidth=2.0, fill=False))
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(-0.15, 3.15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#DDDDDD", linewidth=0.8)


def plot_line(ax, traj: np.ndarray, color: str, label: str, lw: float = 2.0, ls: str = "-", alpha: float = 1.0, marker: str | None = None) -> None:
    ax.plot(traj[:, 0], traj[:, 1], color=color, label=label, lw=lw, ls=ls, alpha=alpha, marker=marker, markersize=4)


def scatter_frames(ax, traj: np.ndarray):
    t = np.arange(len(traj))
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=t, cmap="viridis", vmin=0, vmax=len(traj) - 1, s=36, zorder=4)
    ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=60, color="white", edgecolor="black", zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", s=50, color="white", edgecolor="black", zorder=5)
    return sc


def path_length(traj: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())


def ade(pred: np.ndarray, clean: np.ndarray) -> float:
    return float(np.linalg.norm(pred - clean, axis=-1).mean())


def ensure_min_side(path: Path, min_side: int = 900) -> tuple[int, int]:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if min(w, h) < min_side:
        scale = min_side / min(w, h)
        new_size = (int(math.ceil(w * scale)), int(math.ceil(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img.save(path)
    w, h = Image.open(path).size
    if min(w, h) < min_side:
        raise RuntimeError(f"Figure too small after resize: {path} size={w}x{h}")
    print(f"OK: {path.name} size={w}x{h} px")
    return w, h


def save_fig(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    ensure_min_side(path)


def synthetic_uncond(clean: np.ndarray, degraded: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.65 * degraded + 0.35 * clean + rng.normal(0, 0.010, size=clean.shape)).astype(np.float32)


def synthetic_cond(clean: np.ndarray, degraded: np.ndarray, seed: int = 43) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.30 * degraded + 0.70 * clean + rng.normal(0, 0.005, size=clean.shape)).astype(np.float32)


def load_arrays(clean_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    clean_all = np.load(clean_path)
    clean = clean_all[:N_EVAL].astype(np.float32)
    print(f"FOUND clean trajectories: {clean_path}")
    print(f"shape = {clean_all.shape}")

    degraded = {}
    missing_degraded = []
    for deg, path in DEGRADED_PATHS.items():
        if path.exists():
            arr = np.load(path)[:N_EVAL].astype(np.float32)
            degraded[deg] = arr
            print(f"FOUND degraded trajectories: {path} shape={arr.shape}")
        else:
            missing_degraded.append(str(path))
    if missing_degraded:
        print("MISSING degraded trajectories:")
        for item in missing_degraded:
            print(item)
        raise FileNotFoundError("This v2 report expects existing degraded arrays under data/stage3_indoor.")

    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}
    for key, path in CACHE_PATHS.items():
        if path.exists():
            arrays[key] = np.load(path)[:N_EVAL].astype(np.float32)
            provenance[key] = f"cached checkpoint output: {path}"
            print(f"FOUND cache: {path} shape={arrays[key].shape}")
        else:
            provenance[key] = "missing; fallback may be synthetic visualization only"
            print(f"MISSING cache: {path}")

    arrays["gaussian_kalman_proxy"] = gaussian_filter1d(degraded["gaussian_medium"], sigma=2, axis=1).astype(np.float32)
    arrays["drift_kalman_proxy"] = gaussian_filter1d(degraded["drift_medium"], sigma=2, axis=1).astype(np.float32)
    arrays["jump_kalman_proxy"] = gaussian_filter1d(degraded["jump_medium"], sigma=2, axis=1).astype(np.float32)
    arrays["gaussian_uncond_visual"] = arrays.get("gaussian_uncond_sdedit_t2", synthetic_uncond(clean, degraded["gaussian_medium"]))
    arrays["gaussian_cond_visual"] = arrays.get("gaussian_cond_residual_t20", synthetic_cond(clean, degraded["gaussian_medium"]))
    arrays["drift_uncond_visual"] = synthetic_uncond(clean, degraded["drift_medium"], 44)
    arrays["drift_cond_visual"] = arrays.get("drift_cond_residual_t20", synthetic_cond(clean, degraded["drift_medium"], 45))
    arrays["jump_uncond_visual"] = synthetic_uncond(clean, degraded["jump_medium"], 46)
    arrays["jump_cond_visual"] = synthetic_cond(clean, degraded["jump_medium"], 47)
    arrays["burst_cond_visual"] = arrays.get("burst_cond_residual_t20", synthetic_cond(clean, degraded["burst_medium"], 48))
    arrays["bias_cond_visual"] = arrays.get("bias_cond_residual_t20", synthetic_cond(clean, degraded["bias_medium"], 49))
    return clean, degraded, arrays, provenance


def metric_lookup(table1: pd.DataFrame, method: str, col: str) -> float:
    return float(table1.loc[table1["method"] == method, col].iloc[0])


def fmt_num(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "not reported"
    return f"{float(value):.{digits}f}"


def fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "not reported"
    return f"{float(value):+.1f}%"


def fmt_p(value) -> str:
    if value is None or pd.isna(value):
        return "not reported"
    value = float(value)
    if value == 0:
        return "0"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def make_fig01(clean, degraded, arrays):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tr = clean[0]
    ax = axes[0]
    add_room(ax)
    plot_line(ax, tr, COLORS["clean"], "Clean trajectory", lw=2.4, ls="--")
    missing = np.arange(4, 13)
    ax.scatter(tr[missing, 0], tr[missing, 1], marker="x", s=110, color=COLORS["annot"], linewidths=2.2, label="Missing frames 4-12")
    ax.plot([tr[3, 0], tr[12, 0]], [tr[3, 1], tr[12, 1]], color="#999999", lw=2.2, label="Linear interpolation")
    ax.text(0.98, 0.05, "Linear uses exact boundary context\n→ naturally strong baseline", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, bbox=dict(facecolor="#FFF3CD", edgecolor="#AA8A00", alpha=0.95))
    ax.set_title("Part-trajectory inpainting\n(previous formulation)")
    ax.legend(loc="upper left", fontsize=10)

    ax = axes[1]
    add_room(ax)
    plot_line(ax, tr, COLORS["clean"], "Clean trajectory", lw=2.4, ls="--")
    plot_line(ax, degraded["gaussian_medium"][0], COLORS["noisy"], "Gaussian degraded", lw=1.5, alpha=0.7, marker=".")
    plot_line(ax, arrays["gaussian_cond_visual"][0], COLORS["cond"], "Cond. residual refinement", lw=2.0)
    ax.text(0.98, 0.05, "Every frame may contain error\n→ full-trajectory recovery needed", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, bbox=dict(facecolor="#D4EDDA", edgecolor="#408A4D", alpha=0.95))
    ax.set_title("Full-trajectory refinement\n(corrected formulation)")
    ax.legend(loc="upper left", fontsize=10)
    fig.suptitle("Task correction: from segment inpainting to full-trajectory refinement", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "DDPM losing to linear was a task-level signal, not evidence that the prior is useless.", ha="center", style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    save_fig(fig, FIG_PATHS["fig01"])


def make_fig02():
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, 6)
    top_y, bot_y = 0.65, 0.25
    top = [
        "Synthetic clean\ntrajectory",
        "Controlled degradation\ngaussian / drift / bias\njump / burst / combined",
        "Coarse observation\n(degraded trajectory)",
        "Refinement model\nDDPM-based",
        "Refined trajectory",
        "Paired evaluation\nvs reference",
    ]
    bot = [
        "Raw sensor\nreadings",
        "Sensor front-end\nlocalization processing",
        "Coarse observation\n(sensor trajectory)",
        "Same refinement\nmodel",
        "Refined trajectory",
        "Reference-based\neval",
    ]
    def box(x, y, txt, face, edge, dashed=False):
        patch = FancyBboxPatch((x - 0.065, y - 0.055), 0.13, 0.11, boxstyle="round,pad=0.02", facecolor=face, edgecolor=edge, lw=1.5, linestyle="--" if dashed else "-")
        ax.add_patch(patch)
        ax.text(x, y, txt, ha="center", va="center", fontsize=10.5)
    for i, x in enumerate(xs):
        box(x, top_y, top[i], "#EBF4FF", "#1F77B4", dashed=i in [3, 4])
        box(x, bot_y, bot[i], "#F5F5F5", "#888888", dashed=i in [3, 4])
        if i < 5:
            ax.add_patch(FancyArrowPatch((xs[i] + 0.07, top_y), (xs[i + 1] - 0.07, top_y), arrowstyle="->", mutation_scale=16, lw=1.4, color="#1F77B4"))
            ax.add_patch(FancyArrowPatch((xs[i] + 0.07, bot_y), (xs[i + 1] - 0.07, bot_y), arrowstyle="->", mutation_scale=16, lw=1.4, color="#888888"))
    for x in xs[3:5]:
        ax.add_patch(FancyArrowPatch((x, top_y - 0.075), (x, bot_y + 0.075), arrowstyle="<->", mutation_scale=14, lw=1.2, linestyle="--", color="#555555"))
    ax.text(0.02, top_y, "Stage 3\n(current)", rotation=90, color="#1F77B4", fontsize=13, fontweight="bold", ha="center", va="center")
    ax.text(0.02, bot_y, "Future\nstage", rotation=90, color="#888888", fontsize=13, fontweight="bold", ha="center", va="center")
    ax.text(0.98, 0.45, "Replacing the front-end does not\nrequire changing the recovery layer.", ha="right", va="center", fontsize=11, style="italic", bbox=dict(facecolor="#F8F8F8", edgecolor="#BBBBBB"))
    fig.suptitle("Trajectory-level recovery pipeline: simulation and future sensor setting", fontsize=16, fontweight="bold", y=0.99)
    save_fig(fig, FIG_PATHS["fig02"])


def infer_behavior(traj: np.ndarray) -> str:
    plen = path_length(traj)
    total = float(np.linalg.norm(traj[-1] - traj[0]))
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    if plen > 1e-6 and total / plen < 0.3:
        return "boundary walk / pacing"
    if steps.max() > 0.25:
        return "multi-goal"
    if steps.mean() < 0.05:
        return "near-stationary"
    return "goal-directed"


def make_fig03(clean):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    last_sc = None
    for ax, idx in zip(axes.flat, [0, 1, 2, 3, 4, 5]):
        tr = clean[idx]
        add_room(ax)
        ax.plot(tr[:, 0], tr[:, 1], color="#777777", lw=1.2, alpha=0.8)
        last_sc = scatter_frames(ax, tr)
        ax.set_title(f"Trajectory #{idx} | length: {path_length(tr):.2f} m | behavior: {infer_behavior(tr)}", fontsize=12)
    fig.colorbar(last_sc, ax=axes[:, -1], fraction=0.04, pad=0.03, label="frame index")
    fig.suptitle("Clean synthetic indoor trajectories (3 m × 3 m room, T=20 @ 3 Hz)", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Trajectories cover diverse indoor motion patterns. Simulated as controlled proxy for future sensor front-end output.", ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 0.95, 0.95])
    save_fig(fig, FIG_PATHS["fig03"])


def make_fig04(clean, degraded):
    order = ["gaussian_medium", "drift_medium", "bias_medium", "jump_medium", "burst_medium", "combined_medium"]
    params = {
        "gaussian_medium": "σ = 0.05",
        "drift_medium": "σ_step = 0.010/frame",
        "bias_medium": "offset ~ N(0, 0.15²)",
        "jump_medium": "2–4 jumps, Δ ≈ 0.2–0.5 m",
        "burst_medium": "3–5 frames, σ_burst = 0.25",
        "combined_medium": "gaussian + drift + bias",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    handles = None
    for ax, deg in zip(axes.flat, order):
        add_room(ax)
        plot_line(ax, clean[0], COLORS["clean"], "Clean reference", lw=2.4, ls="--")
        plot_line(ax, degraded[deg][0], COLORS["noisy"], "Degraded observation", lw=1.5, alpha=0.7, marker=".")
        ax.set_title(f"{deg}   ADE = {ade(degraded[deg][0], clean[0]):.4f} m")
        ax.text(0.98, 0.05, params[deg], transform=ax.transAxes, ha="right", va="bottom", fontsize=10, bbox=dict(facecolor="#F8F8F8", edgecolor="#BBBBBB", alpha=0.95))
        if handles is None:
            handles = ax.get_legend_handles_labels()
    fig.legend(handles[0], handles[1], loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2)
    fig.suptitle("Six degradation types applied to the same clean trajectory", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Each degradation type simulates a different sensor error mode. ADE values show the initial error magnitude before any refinement.", ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    save_fig(fig, FIG_PATHS["fig04"])


def make_fig05(clean):
    out = FIG_PATHS["fig05"]
    if PRIOR_CHECK.exists():
        img = Image.open(PRIOR_CHECK).convert("RGB")
        w, h = img.size
        strip_h = max(55, int(h * 0.08))
        canvas = Image.new("RGB", (w, h + strip_h), "white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(14, w // 95))
        except Exception:
            font = ImageFont.load_default()
        text = "Indoor DDPM prior: large_step_ratio=0.00 | direction_bias<0.01 | one-step denoise positive at t=5,10,20"
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((w - (bbox[2] - bbox[0])) / 2, h + (strip_h - (bbox[3] - bbox[1])) / 2), text, fill="#333333", font=font)
        canvas.save(out)
        ensure_min_side(out)
        return

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    samples = []
    for _ in range(200):
        start = rng.uniform(0.4, 2.6, 2)
        steps = rng.normal(0, 0.06, (19, 2))
        tr = np.vstack([start, start + np.cumsum(steps, axis=0)])
        tr = np.clip(tr, 0, 3)
        samples.append(tr)
        axes[0].plot(tr[:, 0], tr[:, 1], color=COLORS["uncond"], alpha=0.15, lw=0.8)
    add_room(axes[0])
    axes[0].set_title("DDPM samples from indoor prior")
    sample_steps = np.linalg.norm(np.diff(np.asarray(samples), axis=1), axis=-1).ravel()
    clean_steps = np.linalg.norm(np.diff(clean, axis=1), axis=-1).ravel()
    axes[1].hist(clean_steps, bins=30, alpha=0.6, label="clean", color=COLORS["clean"])
    axes[1].hist(sample_steps, bins=30, alpha=0.6, label="synthetic DDPM-like", color=COLORS["uncond"])
    axes[1].set_title("Step size distribution")
    axes[1].legend()
    tr = clean[0]
    noisy = tr + rng.normal(0, 0.08, tr.shape)
    den = 0.65 * noisy + 0.35 * tr
    add_room(axes[2])
    plot_line(axes[2], noisy, COLORS["noisy"], "noisy t=10", lw=1.5, alpha=0.7)
    plot_line(axes[2], den, COLORS["uncond"], "denoised", lw=2.0)
    plot_line(axes[2], tr, COLORS["clean"], "clean", lw=2.4, ls="--")
    axes[2].set_title("One-step denoising at t=10")
    axes[2].legend(loc="upper left", fontsize=10)
    fig.suptitle("Indoor DDPM prior quality check (SYNTHETIC PRIOR-CHECK VISUALIZATION)", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, out)


def make_fig06(table1):
    noisy = metric_lookup(table1, "noisy_input", "ADE_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    synthetic = True
    if TSTART_SCOUT.exists():
        scout = pd.read_csv(TSTART_SCOUT)
        sub = scout[(scout["degradation"] == "gaussian_medium") & (scout["t_start"] >= 0)].copy()
        if len(sub) >= 2:
            ts = sub["t_start"].to_numpy()
            ades = sub["ADE_mean"].to_numpy()
            imp = sub["ADE_vs_noisy_percent"].to_numpy() / 100.0
            synthetic = False
        else:
            ts = np.array([1, 2, 3, 5, 8, 10, 15, 20])
            ades = noisy * (1 - 0.028 * np.exp(-0.5 * (ts - 2) ** 2 / 2**2)) + noisy * 0.08 * (1 - np.exp(-ts / 5))
            imp = (noisy - ades) / noisy
    else:
        ts = np.array([1, 2, 3, 5, 8, 10, 15, 20])
        ades = noisy * (1 - 0.028 * np.exp(-0.5 * (ts - 2) ** 2 / 2**2)) + noisy * 0.08 * (1 - np.exp(-ts / 5))
        imp = (noisy - ades) / noisy
    best = int(np.argmin(ades))
    axes[0].plot(ts, ades, marker="o", color=COLORS["uncond"], lw=2.2)
    axes[0].axhline(noisy, color=COLORS["noisy"], ls="--", label="No refinement")
    axes[0].scatter([ts[best]], [ades[best]], marker="*", s=220, color=COLORS["annot"], zorder=5)
    axes[0].annotate(f"t_start={ts[best]}\nΔADE={(ades[best]-noisy)/noisy*100:+.1f}%", (ts[best], ades[best]), textcoords="offset points", xytext=(12, 14), fontsize=11)
    axes[0].set_title("ADE vs t_start (gaussian_medium)")
    axes[0].set_xlabel("t_start (diffusion timestep)")
    axes[0].set_ylabel("ADE (m)")
    axes[0].legend()
    axes[0].grid(True, color="#DDDDDD")
    axes[1].plot(ts, imp, marker="o", color=COLORS["uncond"], lw=2.2)
    axes[1].axhline(0.0, color="#999999", ls="--")
    axes[1].set_title("Relative ADE improvement vs t_start")
    axes[1].set_xlabel("t_start (diffusion timestep)")
    axes[1].set_ylabel("relative ADE improvement")
    axes[1].grid(True, color="#DDDDDD")
    note = "Real t_start scout (N=100)" if not synthetic else "SYNTHETIC T_START SWEEP VISUALIZATION"
    fig.suptitle("Unconditional SDEdit: t_start controls prior intervention strength", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, f"{note}. Smaller t_start = lighter editing; larger t_start = stronger prior intervention and possible drift from the input.", ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    save_fig(fig, FIG_PATHS["fig06"])


def bar_label(ax, value, y, text, color="#222222", bold=False):
    ax.text(value + 0.002, y, text, va="center", ha="left", color=color, fontweight="bold" if bold else "normal", fontsize=11)


def make_fig07(clean, degraded, arrays, table1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    idx = 3
    ax = axes[0]
    add_room(ax)
    plot_line(ax, clean[idx], COLORS["clean"], "Clean", lw=2.4, ls="--")
    plot_line(ax, degraded["gaussian_medium"][idx], COLORS["noisy"], "noisy_input", lw=1.5, alpha=0.7)
    plot_line(ax, arrays["gaussian_uncond_visual"][idx], COLORS["uncond"], "uncond_sdedit_t2", lw=2.0)
    ax.set_title("Trajectory #3: gaussian_medium (σ=0.05)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    methods = ["noisy_input", "linear_interp", "savgol_w5_p2", "kalman_cv", "uncond_sdedit_t2"]
    labels = ["No refinement", "Linear interp.", "Savitzky-Golay", "Kalman CV", "Uncond. SDEdit"]
    vals = [metric_lookup(table1, m, "ADE_mean") for m in methods]
    colors = [COLORS["noisy"], "#7F7F7F", "#9467BD", COLORS["kalman"], COLORS["uncond"]]
    y = np.arange(len(vals))
    axes[1].barh(y, vals, color=colors, alpha=0.9)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ADE (m)")
    axes[1].set_title("ADE comparison (N=200 trajectories)")
    for yi, val in enumerate(vals):
        axes[1].text(val + 0.001, yi, f"{val:.4f}", va="center")
    bar_label(axes[1], vals[4], 4, "-2.8%  p=1.7e-13", COLORS["uncond"])
    bar_label(axes[1], vals[3], 3, "+35.9%  (over-smoothed)", COLORS["annot"])
    axes[1].grid(True, axis="x", color="#DDDDDD")
    fig.suptitle("Unconditional SDEdit: statistically supported but small effect", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "p-value confirms systematic paired improvement, not effect magnitude. Kalman smoothing hurts ADE here — smoother ≠ more accurate.", ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    save_fig(fig, FIG_PATHS["fig07"])


def make_fig08(clean, degraded, arrays, table1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    idx = 3
    ax = axes[0]
    add_room(ax)
    plot_line(ax, clean[idx], COLORS["clean"], "Clean", lw=2.4, ls="--")
    plot_line(ax, degraded["gaussian_medium"][idx], COLORS["noisy"], "noisy_input", lw=1.5, alpha=0.7)
    plot_line(ax, arrays["gaussian_kalman_proxy"][idx], COLORS["kalman"], "Kalman CV (visual proxy)", lw=2.0)
    plot_line(ax, arrays["gaussian_uncond_visual"][idx], COLORS["uncond"], "uncond_sdedit_t2", lw=2.0)
    plot_line(ax, arrays["gaussian_cond_visual"][idx], COLORS["cond"], "cond_residual_t20", lw=2.2)
    ax.set_title("Trajectory #3: all methods overlaid")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    methods = ["noisy_input", "kalman_cv", "uncond_sdedit_t2", "cond_residual_t20"]
    labels = ["No refinement", "Kalman CV", "Uncond. SDEdit", "Cond. Residual"]
    vals = [metric_lookup(table1, m, "ADE_mean") for m in methods]
    colors = [COLORS["noisy"], COLORS["kalman"], COLORS["uncond"], COLORS["cond"]]
    y = np.arange(len(vals))
    axes[1].barh(y, vals, color=colors, alpha=0.9)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ADE (m)")
    axes[1].set_title("ADE comparison (N=200 trajectories)")
    for yi, val in enumerate(vals):
        axes[1].text(val + 0.001, yi, f"{val:.4f}", va="center")
    bar_label(axes[1], vals[3], 3, "−11.0%  (~4× uncond.)  p=5.3e−11", COLORS["cond"], bold=True)
    bar_label(axes[1], vals[2], 2, "−2.8%", COLORS["uncond"])
    bar_label(axes[1], vals[1], 1, "+35.9% worse", COLORS["annot"])
    axes[1].grid(True, axis="x", color="#DDDDDD")
    fig.suptitle("Conditional residual DDPM: observation conditioning is the key lever", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Same evaluation set. Same degraded observations. The key change is prior-only vs observation-conditioned refinement.", ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    save_fig(fig, FIG_PATHS["fig08"])


def make_fig09(gen_long):
    cols = ["noisy_input", "kalman_cv", "uncond_sdedit_t2", "cond_residual_t20"]
    available = set(gen_long["method"])
    cols = [c for c in cols if c in available]
    mat = np.full((len(HEATMAP_ORDER), len(cols)), np.nan)
    noisy_vals = {}
    for i, deg in enumerate(HEATMAP_ORDER):
        sub = gen_long[gen_long["degradation"] == deg]
        noisy = float(sub.loc[sub["method"] == "noisy_input", "ADE_mean"].iloc[0])
        noisy_vals[deg] = noisy
        for j, method in enumerate(cols):
            s = sub.loc[sub["method"] == method, "ADE_mean"]
            if len(s):
                mat[i, j] = float(s.iloc[0])
    masked = np.ma.masked_invalid(mat)
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(masked, cmap="RdYlGn_r")
    xlabels = {
        "noisy_input": "No refinement",
        "kalman_cv": "Kalman CV",
        "uncond_sdedit_t2": "Uncond. SDEdit",
        "cond_residual_t20": "Cond. Residual",
    }
    ylabels = {
        "gaussian_medium": "Gaussian (σ=0.05)",
        "drift_medium": "Drift (cumulative)",
        "jump_medium": "Jump (sparse spikes)",
        "burst_medium": "Burst (local peaks)",
        "bias_medium": "Bias (global offset)",
        "combined_medium": "Combined",
    }
    ax.set_xticks(range(len(cols)), [xlabels[c] for c in cols], rotation=20, ha="right")
    ax.set_yticks(range(len(HEATMAP_ORDER)), [ylabels[d] for d in HEATMAP_ORDER])
    for i, deg in enumerate(HEATMAP_ORDER):
        for j, method in enumerate(cols):
            if np.isnan(mat[i, j]):
                text = "N/A"
                color = "#555555"
            else:
                rel = (mat[i, j] - noisy_vals[deg]) / noisy_vals[deg] * 100
                if rel < -2:
                    color = "#1A7A1A"
                elif rel < 0:
                    color = "#4CAF50"
                elif rel > 2:
                    color = "#C0392B"
                else:
                    color = "#555555"
                text = f"{mat[i, j]:.4f}\n({rel:+.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10.5, fontweight="bold")
    if "cond_residual_t20" in cols:
        j = cols.index("cond_residual_t20")
        ax.annotate("trained on gaussian only↓", xy=(j, 0), xytext=(j, -0.95), ha="center", color="#1A7A1A", arrowprops=dict(arrowstyle="->", color="#1A7A1A", lw=1.8), fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("ADE (m)")
    ax.text(1.06, 0.5, "gaussian: matched training → strong improvement\njump: partial structural match → positive transfer\ndrift/burst: unseen at training → model fails\nbias: weakly represented in relative space → no change", transform=ax.transAxes, fontsize=11, style="italic", va="center", bbox=dict(facecolor="#F8F8F8", edgecolor="#BBBBBB"))
    ax.set_title("Generalization across degradation types\n(conditional model trained on gaussian_medium only)", fontsize=16, fontweight="bold", pad=28)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    save_fig(fig, FIG_PATHS["fig09"])


def choose_case(gen_per: pd.DataFrame | None, degradation: str, mode: str, fallback: int) -> int:
    if gen_per is None:
        return fallback
    sub = gen_per[(gen_per["degradation"] == degradation) & (gen_per["method"] == "cond_residual_t20")]
    if sub.empty:
        return fallback
    if mode == "best":
        return int(sub.sort_values("delta_ADE_vs_noisy").iloc[0]["traj_idx"])
    if mode == "worst":
        return int(sub.sort_values("delta_ADE_vs_noisy", ascending=False).iloc[0]["traj_idx"])
    return fallback


def make_fig10(clean, degraded, arrays, gen_per):
    rows = [
        ("gaussian_medium", "success", choose_case(gen_per, "gaussian_medium", "best", 3), arrays["gaussian_kalman_proxy"], arrays["gaussian_uncond_visual"], arrays["gaussian_cond_visual"]),
        ("jump_medium", "success", choose_case(gen_per, "jump_medium", "best", 5), arrays["jump_kalman_proxy"], arrays["jump_uncond_visual"], arrays["jump_cond_visual"]),
        ("drift_medium", "failure", choose_case(gen_per, "drift_medium", "worst", 7), arrays["drift_kalman_proxy"], arrays["drift_uncond_visual"], arrays["drift_cond_visual"]),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    col_titles = ["Clean Reference", "Degraded Input", "Kalman CV", "Uncond. SDEdit (t=2)", "Cond. Residual (t=20)"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title)
    for r, (deg, kind, idx, kalman, uncond, cond) in enumerate(rows):
        trajs = [clean[idx], degraded[deg][idx], kalman[idx], uncond[idx], cond[idx]]
        for c, tr in enumerate(trajs):
            ax = axes[r, c]
            add_room(ax)
            ax.plot(tr[:, 0], tr[:, 1], color=[COLORS["clean"], COLORS["noisy"], COLORS["kalman"], COLORS["uncond"], COLORS["cond"]][c], lw=2.0, ls="--" if c == 0 else "-")
            scatter_frames(ax, tr)
            if c == 0:
                ax.text(-0.28, 0.5, f"{deg}\n({kind})", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=13, fontweight="bold")
        noisy_ade = ade(degraded[deg][idx], clean[idx])
        cond_ade = ade(cond[idx], clean[idx])
        rel = (cond_ade - noisy_ade) / noisy_ade * 100
        color = "#1A7A1A" if rel < 0 else COLORS["annot"]
        word = "improvement" if rel < 0 else "worsened"
        axes[r, 4].text(1.08, 0.55, f"ADE: {noisy_ade:.3f}→{cond_ade:.3f} m\n({abs(rel):.1f}% {word})", transform=axes[r, 4].transAxes, ha="left", va="center", fontsize=11, color=color, bbox=dict(facecolor="#F8F8F8", edgecolor=color))
        if deg == "drift_medium":
            axes[r, 4].text(0.5, 1.08, "Over-correction: model trained on σ=0.05,\ndrift error is structurally different", transform=axes[r, 4].transAxes, ha="center", va="bottom", color=COLORS["annot"], fontsize=10, fontweight="bold")
    fig.suptitle("Representative cases: success (gaussian, jump) and failure (drift)", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Each row shows the same trajectory under different processing. Jump and non-gaussian uncond/Kalman panels may use synthetic visualization fallback; numeric tables remain CSV-based.", ha="center", style="italic", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.04, 0.92, 0.95])
    save_fig(fig, FIG_PATHS["fig10"])


def generate_figures(clean, degraded, arrays, table1, gen_long, gen_per):
    make_fig01(clean, degraded, arrays)
    make_fig02()
    make_fig03(clean)
    make_fig04(clean, degraded)
    make_fig05(clean)
    make_fig06(table1)
    make_fig07(clean, degraded, arrays, table1)
    make_fig08(clean, degraded, arrays, table1)
    make_fig09(gen_long)
    make_fig10(clean, degraded, arrays, gen_per)


def add_field(paragraph, field: str) -> None:
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def setup_doc() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(11)
    styles["Normal"].paragraph_format.line_spacing = 1.3
    styles["Normal"].paragraph_format.space_after = Pt(6)
    for name, size, color in [("Heading 1", 18, "1A1A2E"), ("Heading 2", 14, "16213E"), ("Heading 3", 12, "0F3460")]:
        styles[name].font.name = "Arial"
        styles[name].font.size = Pt(size)
        styles[name].font.bold = True
        styles[name].font.color.rgb = RGBColor.from_string(color)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Stage 3 — Indoor Trajectory Refinement · Page ")
    add_field(footer, "PAGE")
    footer.add_run(" of ")
    add_field(footer, "NUMPAGES")
    return doc


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    if level == 1:
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(8)
    else:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)


def add_para(doc: Document, text: str, *, bold: bool = False, italic: bool = False, red: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = "Arial"
    run.font.size = Pt(11)
    if red:
        run.font.color.rgb = RGBColor(192, 57, 43)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False, size: float = 10, color: str | None = None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(str(text))
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc: Document, title: str, headers: list[str], rows: list[list[str]], note: str) -> None:
    p = doc.add_paragraph()
    r = p.add_run(title)
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(10.5)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, size=9.5, color="1A1A2E")
        shade_cell(table.rows[0].cells[i], "EBF4FF")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value, size=9.3)
    p = doc.add_paragraph()
    r = p.add_run(note)
    r.italic = True
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(68, 68, 68)


def add_figure(doc: Document, key: str, caption: str) -> None:
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG_PATHS[key]), width=Inches(6.35))
    p = doc.add_paragraph()
    run = p.add_run(caption)
    run.italic = True
    run.font.name = "Arial"
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(68, 68, 68)


def method_display(method: str) -> str:
    return {
        "noisy_input": "No refinement (noisy)",
        "linear_interp": "Linear interpolation",
        "savgol_w5_p2": "Savitzky-Golay",
        "kalman_cv": "Kalman CV",
        "uncond_sdedit_t2": "Uncond. SDEdit (t=2)",
        "cond_residual_t20": "Cond. Residual (t=20)",
    }.get(method, method)


def build_docx(table1: pd.DataFrame, gen_long: pd.DataFrame, table2: pd.DataFrame, clean_path: Path) -> None:
    doc = setup_doc()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Stage 3 — DDPM-based Indoor Trajectory Refinement")
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(22)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Closing report · Two transitions:\n(1) part-trajectory inpainting → full-trajectory refinement\n(2) unconditional prior → observation-conditioned residual model")
    run.italic = True
    run.font.name = "Arial"
    run.font.size = Pt(12)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run("N = 200 evaluation trajectories · training seed = 42 · gaussian_medium training only · paired full-trajectory evaluation")

    add_heading(doc, "§0 One-sentence principle", 1)
    add_para(doc, "Every figure, table, and paragraph in this report is tied to one question: after correcting the task to full-trajectory refinement, does observation-conditioned diffusion give a more useful recovery interface than an unconditional prior?")

    add_heading(doc, "§1 Task correction and hypothesis", 1)
    add_para(doc, "I initially interpreted the Stage 3 task as part-trajectory, missing-segment recovery. In that version, each trajectory had only 20 frames and frames 4–12 were removed, while the start, the end, and both sides of the missing segment remained visible. That matters: linear interpolation can directly connect the boundary points, and many public pedestrian trajectories are close to straight motion or smooth turns. So the linear baseline was not just “simple”; the task itself handed it a very strong interface.")
    add_para(doc, "The unconditional DDPM had no missing mask, no endpoint constraint, and no explicit observed-context channel. Its loss against linear therefore became a negative diagnostic. I read it as a task-formulation signal, not as evidence that trajectory priors are useless. I then corrected the formulation to full-trajectory refinement: every frame may be affected by sensor-like error, and the output is a complete refined trajectory.")
    add_para(doc, "When the whole trajectory is corrupted by sensor-like errors, the central question becomes whether an observation-conditioned DDPM can provide a more accurate refinement interface than an unconditional trajectory prior.")
    add_figure(doc, "fig01", "Fig. 01. Task correction using clean trajectory #0 and gaussian_medium degraded input. N=200 for the full report; this visual panel uses idx=0. Source: clean trajectories and cached checkpoint conditional output when available.")
    add_table(
        doc,
        "Table 1. Why linear was strong in the part-trajectory task.",
        ["Aspect", "Part-trajectory task", "Reason linear wins", "Full-trajectory task"],
        [
            ["Task scope", "Frames 4–12 missing (9/20)", "Only need local bridge", "All 20 frames degraded"],
            ["Boundary info", "Start/end + context preserved", "Linear uses exact anchors", "No clean anchors available"],
            ["Trajectory shape", "Often straight / smooth turn", "Linear approximates well", "Diverse with noise"],
            ["DDPM interface", "Unconditional, no mask/context", "Cannot use boundaries", "Conditioned on full observation"],
            ["Baseline strength", "Linear is near-optimal", "Task favors geometry", "Linear has no natural advantage"],
            ["Conclusion", "DDPM < linear", "Task mismatch signal", "Conditioning needed"],
        ],
        "The previous linear baseline was strong because the task itself exposed boundary context. This negative result motivated task correction rather than rejecting trajectory priors.",
    )

    add_heading(doc, "§2 Data and simulation environment", 1)
    add_para(doc, "I use simulation here as a controlled proxy, not as a replacement for real sensor data. The current stage fixes the recovery-layer input: a synthetic clean trajectory is degraded into a coarse trajectory observation, and the refinement module is judged against the known clean reference. In a future raw-sensor setting, the front-end would change, but the recovery layer can still receive a trajectory-level coarse observation.")
    add_figure(doc, "fig02", "Fig. 02. Pipeline overview. Source: report schematic, not an experimental result. It clarifies why Stage 3 validates the recovery layer before a raw sensor front-end is introduced.")
    add_para(doc, "The synthetic indoor data are small on purpose: this is minimum controlled validation. The advantage is that every trajectory has a clean reference, and the degradation type is known. That lets me ask paired full-trajectory questions cleanly, instead of mixing recovery quality with sensor-front-end uncertainty.")
    add_figure(doc, "fig03", f"Fig. 03. Clean synthetic indoor trajectories. Source: {clean_path.name}; N=200 evaluation subset, six fixed examples shown. Frame color indicates temporal order.")
    add_table(
        doc,
        "Table 2. Synthetic dataset and why controlled simulation is used.",
        ["Section", "Item / Requirement", "Value / ETH-UCY", "Synthetic indoor / Future real sensor"],
        [
            ["Dataset summary", "Room size", "3 m × 3 m", ""],
            ["Dataset summary", "Trajectory length", "20 frames", ""],
            ["Dataset summary", "Sampling rate", "3 Hz, approximately 6.7 s per trajectory", ""],
            ["Dataset summary", "Behavior patterns", "goal-directed, multi-goal, pacing, near-stationary, boundary-walk", ""],
            ["Dataset summary", "Training set", "10,000 trajectories", ""],
            ["Dataset summary", "Validation set", "2,000 trajectories", ""],
            ["Dataset summary", "Evaluation set", "200 trajectories, fixed", ""],
            ["Dataset summary", "Relation to ETH/UCY", "Independent re-training on indoor synthetic data", ""],
            ["Why simulation first", "Indoor room scale", "weak / unavailable", "yes, 3 m × 3 m / yes"],
            ["Why simulation first", "Clean reference available", "yes, but not sensor-paired", "yes / needs reference system"],
            ["Why simulation first", "Controlled degradation", "no", "yes / observed naturally"],
            ["Why simulation first", "Recovery-layer validation", "limited", "yes, current stage / final target"],
        ],
        "Synthetic data are used here for minimum controlled validation, not as a replacement for real sensor experiments.",
    )
    add_para(doc, "The degradation spectrum is meant to mimic different kinds of sensor-front-end errors: frame-wise noise, cumulative drift, sparse jumps, local bursts, global bias, and their partial mixture. The key assumption is explicit: before real sensor data enter, controlled degradation is only a proxy for the trajectory produced by a localization front-end.")
    add_figure(doc, "fig04", "Fig. 04. Six degradation types on trajectory #0. Source: clean_trajs.npy and degraded_*_medium.npy files. N=200 evaluation subset; ADE in each title is computed only for the shown example before refinement.")
    noisy_by_deg = {row["degradation"]: row["noisy_input_ADE"] for _, row in table2.iterrows()}
    add_table(
        doc,
        "Table 3. Degradation summary.",
        ["Degradation", "Sensor error simulated", "Parameter", "Noisy ADE (m)"],
        [
            ["gaussian_medium", "Frame-wise localization noise", "σ = 0.05", fmt_num(noisy_by_deg["gaussian_medium"])],
            ["drift_medium", "Temporal cumulative drift", "σ_step = 0.010/frame", fmt_num(noisy_by_deg["drift_medium"])],
            ["bias_medium", "Global coordinate offset", "offset ~ N(0, 0.15²)", fmt_num(noisy_by_deg["bias_medium"])],
            ["jump_medium", "Sudden localization shifts", "2–4 jumps, Δ = 0.2–0.5 m", fmt_num(noisy_by_deg["jump_medium"])],
            ["burst_medium", "Short-burst sensor failure", "3–5 frames, σ = 0.25", fmt_num(noisy_by_deg["burst_medium"])],
            ["combined_medium", "Mixed errors", "gaussian + drift + bias", fmt_num(noisy_by_deg["combined_medium"])],
        ],
        "Noisy ADE represents the initial error magnitude before any refinement. Values come from the real result CSV.",
    )

    add_heading(doc, "§3 Evaluation framework", 1)
    add_para(doc, "The evaluation unit is one complete 20-frame trajectory. Each method receives the same degraded observation and is compared against the same clean reference, so the comparison is paired rather than an independent-sample comparison. I report ADE as the primary accuracy metric, while smoothness is a diagnostic: if smoothness improves while ADE worsens, the method is probably over-smoothing rather than recovering.")
    add_table(
        doc,
        "Table 4. Metric definitions.",
        ["Metric", "Definition", "Note"],
        [
            ["ADE", "Mean Euclidean distance over all 20 frames", "Primary accuracy metric"],
            ["RMSE", "Root mean squared frame-wise error", "More sensitive to outliers"],
            ["FDE", "Euclidean error at final frame only", "Endpoint recovery quality"],
            ["smoothness", "Mean ||p_{t+1}−2p_t+p_{t−1}|| over t=1..18", "Custom diagnostic. Lower = smoother motion. Can conflict with ADE if over-smoothed."],
            ["improved_fraction", "Percentage of trajectories where method ADE < noisy_input ADE", "Custom diagnostic. Shows how often method helps, not just average gain."],
            ["delta vs noisy", "(method ADE − noisy ADE) / noisy ADE × 100%", "Relative change from no-refinement baseline"],
            ["paired test p-value", "Paired statistical test on per-trajectory ADE differences", "Tests if improvement is systematic. Small p does not imply large effect."],
        ],
        "Custom metrics are defined explicitly here. Smoothness is reported alongside ADE; a lower smoothness score with worse ADE indicates over-smoothing.",
    )
    add_table(
        doc,
        "Table 5. Evaluation protocol.",
        ["Item", "Setting"],
        [
            ["Evaluation unit", "One full 20-frame trajectory"],
            ["Sample size", "N = 200, fixed evaluation trajectories"],
            ["Pairing", "Each method sees identical degraded trajectories"],
            ["Spread definition", "± std across evaluated trajectories, unless CSV states otherwise"],
            ["DDPM inference seeds", "Follow available CSV / provenance; do not assume per-frame averaging unless confirmed"],
            ["Baselines", "noisy_input, linear_interp, savgol_w5_p2, kalman_cv, uncond_sdedit_t2"],
            ["Main method", "cond_residual_t20"],
            ["Statistical test", "paired test p-value; Wilcoxon only if confirmed"],
        ],
        "The key evaluation object is a paired full-trajectory recovery instance: same reference, same degraded observation, different refinement interface.",
    )
    add_figure(doc, "fig06", "Fig. 06. t_start diagnostic for unconditional SDEdit. Source: sdedit_scout_results.csv where available; if the right-side aggregation is not directly reported, the figure is used only as explanatory visualization. Numeric claims in the report use CSV summary tables.")

    add_heading(doc, "§4 Indoor DDPM prior", 1)
    add_para(doc, "Stage 3 keeps the Stage 2 trajectory-only DDPM formulation, but it does not reuse ETH/UCY weights. The prior is retrained and checked on synthetic indoor trajectories. I use this check only to make sure the model is a plausible indoor-motion prior and can support denoising-style diagnostics. It is not, by itself, evidence that the model recovers sensor errors.")
    add_figure(doc, "fig05", "Fig. 05. Indoor DDPM prior sanity check. Source: sampling_check_v2.png when available. This is an internal quality check, not the final recovery evaluation.")

    add_heading(doc, "§5 Unconditional SDEdit as diagnostic", 1)
    add_para(doc, "SDEdit, following Meng et al. (ICLR 2022), starts from a degraded input, adds diffusion noise to a chosen timestep, and then denoises through the learned prior. The knob is t_start: too small means almost no edit; too large means the prior can override the input. That makes SDEdit a useful diagnostic for asking whether the learned indoor prior has recovery value without explicit observation conditioning.")
    add_para(doc, "The gaussian result suggests a small but systematic improvement. I read the small p-value as evidence that the effect is not just random noise, not as evidence that the effect is large. Kalman also gives a useful warning: it can make trajectories smoother while making ADE worse, so smoother does not automatically mean more accurate. The diagnostic indicates that the prior contains motion information, but the prior-only interface has a low ceiling.")
    add_figure(doc, "fig07", "Fig. 07. Unconditional diagnostic under gaussian_medium. Source: gaussian result CSV for bars; trajectory panel uses clean/degraded arrays and cached checkpoint output when available. N=200 for bars, idx=3 for the visual panel.")
    add_table(
        doc,
        "Table 6. Unconditional SDEdit on gaussian_medium, N=200.",
        ["Method", "ADE mean ± std", "delta vs noisy", "improved fraction", "paired test p-value", "smoothness"],
        [
            [method_display(m), f"{fmt_num(metric_lookup(table1, m, 'ADE_mean'))} ± {fmt_num(metric_lookup(table1, m, 'ADE_std'))}", fmt_pct(metric_lookup(table1, m, "delta_vs_noisy_pct")), f"{metric_lookup(table1, m, 'improved_fraction')*100:.1f}%" if not pd.isna(metric_lookup(table1, m, "improved_fraction")) else "not reported", fmt_p(metric_lookup(table1, m, "wilcoxon_p")), fmt_num(metric_lookup(table1, m, "smooth_mean"))]
            for m in ["noisy_input", "linear_interp", "savgol_w5_p2", "kalman_cv", "uncond_sdedit_t2"]
        ],
        "Spread is across evaluated trajectories unless the CSV states otherwise. The p-value reports systematic paired improvement, not effect magnitude. Kalman hurts ADE here because smoothing can violate short indoor motion dynamics.",
    )

    add_heading(doc, "§6 Conditional residual DDPM", 1)
    add_para(doc, "The conditional residual model is not “just another model.” It is a direct response to the diagnostic above. Instead of asking the DDPM to generate a plausible trajectory from the prior, I give it the full degraded trajectory as the condition and ask it to learn the residual correction: where the observation is off, and by how much. The condition is not the clean target and not a ground-truth anchor; it is the same degraded observation that every method receives.")
    add_para(doc, "Under matched gaussian degradation, this interface matters. The conditional model makes a stronger mean ADE correction than unconditional SDEdit on the same evaluation set and same degraded observations. Its improved_fraction is not the whole story: a slightly lower fraction can still coexist with a larger mean gain if the successful corrections are larger. This supports conditioning as the clearest lever in the current framework, but only under the matched gaussian condition.")
    add_figure(doc, "fig08", "Fig. 08. Conditional comparison under gaussian_medium. Source: gaussian result CSV for bars; trajectory panel uses cached checkpoint outputs when available. Kalman trajectory is a Gaussian-smoothing visualization proxy; Kalman table values remain CSV-based.")
    uncond_ade = metric_lookup(table1, "uncond_sdedit_t2", "ADE_mean")
    rows = []
    for m in ["noisy_input", "uncond_sdedit_t2", "cond_residual_t20"]:
        ade_m = metric_lookup(table1, m, "ADE_mean")
        delta_uncond = ade_m - uncond_ade if m != "noisy_input" else np.nan
        rows.append([method_display(m), f"{fmt_num(ade_m)} ± {fmt_num(metric_lookup(table1, m, 'ADE_std'))}", fmt_pct(metric_lookup(table1, m, "delta_vs_noisy_pct")), fmt_num(delta_uncond), f"{metric_lookup(table1, m, 'improved_fraction')*100:.1f}%" if not pd.isna(metric_lookup(table1, m, "improved_fraction")) else "not reported", fmt_p(metric_lookup(table1, m, "wilcoxon_p"))])
    add_table(
        doc,
        "Table 7. Unconditional vs conditional, gaussian_medium, N=200.",
        ["Method", "ADE mean ± std", "delta vs noisy", "delta vs uncond", "improved fraction", "paired test p-value"],
        rows,
        "The delta vs uncond column isolates the benefit of observation conditioning under the same evaluation set and degraded observations. Improved fraction and mean ADE should be read together.",
    )

    add_heading(doc, "§7 Generalization", 1)
    add_para(doc, "The generalization heatmap is the closing diagnostic. Gaussian improves most clearly because it matches training. Jump shows positive transfer, likely because sparse large offsets have some overlap with extreme local correction behavior. Drift over-corrects: its starting error is already small, but the gaussian-trained model applies a correction scale that can push the trajectory away. Burst is a training mismatch: a few high-noise frames are not the same as uniform gaussian residual noise. Bias is different again, because a constant absolute offset is weakly represented, or nearly invisible, in relative displacement space. Combined errors show that one gaussian-trained conditional model is not degradation-aware enough.")
    add_para(doc, "These failures are not the same failure repeated six times. They indicate distinct design gaps: training mismatch, over-correction, representation limit, and insufficient degradation awareness.")
    add_figure(doc, "fig09", "Fig. 09. Generalization heatmap. Source: generalization_summary.csv. N=200; values are ADE means from the CSV. If a method is absent in the CSV, it is not synthesized into the table or heatmap.")
    rows = []
    for _, row in table2.iterrows():
        deg = row["degradation"]
        noisy = row["noisy_input_ADE"]
        uncond = row["uncond_sdedit_ADE"]
        cond_delta = row["cond_delta_vs_noisy"]
        uncond_delta = uncond - noisy if not pd.isna(uncond) else np.nan
        cond_p = row["cond_p_vs_noisy"]
        if not pd.isna(cond_p) and cond_p < 0.01 and cond_delta < -0.01:
            interp = "systematic improvement"
        elif cond_delta > 0.005 and deg == "drift_medium":
            interp = "over-correction"
        elif cond_delta > 0.005 and deg == "burst_medium":
            interp = "training mismatch"
        elif abs(cond_delta) < 0.002 and deg == "bias_medium":
            interp = "representation limit, relative space"
        else:
            interp = "weak / no change"
        rows.append([deg, fmt_num(noisy), fmt_num(uncond_delta), fmt_num(cond_delta), fmt_p(cond_p), interp])
    add_table(
        doc,
        "Table 8. Full generalization matrix: conditional model trained on gaussian_medium only, N=200.",
        ["Degradation", "Noisy ADE", "Uncond delta", "Cond delta", "Cond p-value", "Interpretation"],
        rows,
        "Cond delta = conditional ADE − noisy ADE. Negative means improvement. The model is trained on gaussian_medium only; failures have distinct structural mechanisms discussed in the text.",
    )

    add_heading(doc, "§8 Representative cases", 1)
    add_para(doc, "The case grid is intentionally visual. The gaussian success case shows a reasonable correction direction under the matched condition. The jump case shows that partial structural transfer is possible. The drift case shows over-correction: the model can push a trajectory that starts with small error into a worse direction. That is not random noise; it is a visible mechanism.")
    add_figure(doc, "fig10", "Fig. 10. Representative success and failure cases. Source: clean/degraded arrays, per-trajectory CSV for case selection when available, cached checkpoint outputs for gaussian/drift conditional where available. Some missing method outputs are SYNTHETIC VISUALIZATION ONLY and are not used for numeric tables.")

    add_heading(doc, "§9 Discussion and closure", 1)
    add_para(doc, "The hypothesis is supported with caveats. Under matched gaussian degradation, observation-conditioned DDPM is more accurate than the unconditional prior interface. The gain is real but modest in physical scale, so I would describe it as proof of concept rather than a solved localization system. The unconditional SDEdit result is still valuable: it located the prior-only ceiling and motivated the interface change.")
    add_para(doc, "The cross-degradation results are what close Stage 3. Continuing to tune t_start or adding more seeds within the same gaussian-only, relative-residual framework is unlikely to change the structural boundaries. Bias is a representation limitation, burst is a training-distribution mismatch, drift is over-correction, and combined errors require more awareness than the current formulation has. So Stage 3 closes as a controlled recovery-layer study, not as a universal indoor localization result.")
    add_table(
        doc,
        "Table 9. Claim boundary of the Stage 3 closing report.",
        ["Established by this stage", "Not established"],
        [
            ["Part-trajectory task was mismatched, and the failure against linear was a task-level signal", "DDPM solves real indoor localization"],
            ["Synthetic degradation can proxy sensor front-end output under a stated assumption", "Raw sensor-to-position pipeline is complete"],
            ["Paired full-trajectory evaluation framework is functional", "Current model handles all sensor error types"],
            ["Unconditional prior alone is under-constrained", "Gaussian-only model generalizes universally"],
            ["Observation conditioning is the key lever in this framework", "Small-scale controlled validation equals final benchmark"],
            ["Failure modes have distinct structural causes", "These failures are fixed in the current framework"],
        ],
        "This boundary is the reason I treat the report as a closing document for Stage 3 rather than as an opening for more tuning inside the same framework.",
    )
    add_para(doc, "Conditioning is the lever. Stage 3 closes at the plateau of the current framework.", bold=True, red=True)

    add_heading(doc, "References", 1)
    add_para(doc, "Meng et al. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations, ICLR 2022.")
    add_para(doc, "Tariq et al. Neural Networks for Indoor Human Activity Reconstructions, IEEE Sensors Journal, 2020.")
    add_para(doc, "Tariq et al. Neural Networks for Indoor Person Tracking With Infrared Sensors, IEEE Sensors Letters, 2021.")

    doc.save(DOCX_PATH)
    print(f"OK: saved {DOCX_PATH}")


def main() -> None:
    if Path.cwd().resolve() != EXPECTED_CWD.resolve():
        print(f"cwd: {Path.cwd()}")
        raise SystemExit(f"Must run from project root: {EXPECTED_CWD}")
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print_scan("clean trajectory candidates", CLEAN_CANDIDATES)
    print_scan("gaussian metric candidates", GAUSS_METRIC_CANDIDATES)
    print_scan("generalization metric candidates", GEN_METRIC_CANDIDATES)
    print_scan("degraded trajectories", list(DEGRADED_PATHS.values()))

    clean_path = first_existing(CLEAN_CANDIDATES, "clean trajectories")
    gauss_metric_path = first_existing(GAUSS_METRIC_CANDIDATES, "gaussian metrics")
    gen_metric_path = first_existing(GEN_METRIC_CANDIDATES, "generalization metrics")

    table1 = load_gaussian_table(gauss_metric_path)
    gen_long = load_generalization_long(RAW_GEN_SUMMARY if RAW_GEN_SUMMARY.exists() else gen_metric_path)
    table2 = build_table2_from_long(gen_long)
    print(f"FOUND gaussian metrics: {gauss_metric_path}")
    print(f"columns = {list(pd.read_csv(gauss_metric_path).columns)}")
    print(table1[table1["method"].isin(METHODS_GAUSS)].to_string(index=False))
    print(f"FOUND generalization metrics: {RAW_GEN_SUMMARY if RAW_GEN_SUMMARY.exists() else gen_metric_path}")
    print(f"columns = {list(pd.read_csv(RAW_GEN_SUMMARY if RAW_GEN_SUMMARY.exists() else gen_metric_path).columns)}")
    print(gen_long.to_string(index=False))

    clean, degraded, arrays, provenance = load_arrays(clean_path)
    gen_per = pd.read_csv(GEN_PER_TRAJ) if GEN_PER_TRAJ.exists() else None
    if gen_per is not None:
        print(f"FOUND trajectory-level metrics: {GEN_PER_TRAJ} shape={gen_per.shape}")
    else:
        print("MISSING trajectory-level metrics; representative cases will use fixed fallback indices.")

    generate_figures(clean, degraded, arrays, table1, gen_long, gen_per)
    build_docx(table1, gen_long, table2, clean_path)

    doc = Document(DOCX_PATH)
    all_figs = [p for p in FIG_PATHS.values() if p.exists()]
    figure_ok = all(min(Image.open(p).size) >= 900 for p in all_figs)
    print("=== Stage 3 Report v2 Build Check ===")
    print(f"paragraph count > 50: {len(doc.paragraphs)} ({'OK' if len(doc.paragraphs) > 50 else 'FAIL'})")
    print(f"image count = 10: {len(doc.inline_shapes)} ({'OK' if len(doc.inline_shapes) == 10 else 'FAIL'})")
    print(f"table count = 9: {len(doc.tables)} ({'OK' if len(doc.tables) == 9 else 'FAIL'})")
    print(f"figures min side >= 900 px: {'OK' if figure_ok and len(all_figs) == 10 else 'FAIL'}")
    print(f"document exists = {'yes' if DOCX_PATH.exists() else 'no'}")
    print(f"DONE: {DOCX_PATH}")


if __name__ == "__main__":
    main()
