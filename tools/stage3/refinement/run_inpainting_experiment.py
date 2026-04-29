"""run_inpainting_experiment.py

Validates the hypothesis that the unconditional DDPM prior fails for conditional
trajectory reconstruction (v1/v2 post-hoc projection) because the model sees no
context during reverse sampling.  RePaint-style inpainting (v3) clamps observed
frames back at every reverse step, testing whether continuous context conditioning
improves missing-segment quality.

Usage:
    .venv/bin/python -m tools.stage3.refinement.run_inpainting_experiment
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Experiment constants ───────────────────────────────────────────────────────
N_EXPERIMENT  = 1024    # first N trajectories (documents computation budget)
BEST_ALPHA_V2 = 0.1     # best blend alpha from prior alpha-sweep
NUM_SEEDS_V3  = 5       # independent reverse-diffusion passes for v3
SEED_BASE_V3  = 42
TAG           = "span20_fixed_seed42"

DEGRADATION_NAMES  = [
    "missing_only",
    "missing_noise",
    "missing_drift",
    "missing_noise_drift",
]
DEGRADATION_LABELS = {
    "missing_only":        "Missing Only",
    "missing_noise":       "Missing + Noise",
    "missing_drift":       "Missing + Drift",
    "missing_noise_drift": "Missing + Noise + Drift",
}
METRIC_NAMES = [
    "ADE", "RMSE", "masked_ADE", "masked_RMSE",
    "wall_crossing_count", "off_map_ratio",
]
METHOD_ORDER = [
    "linear_interp",
    "savgol_w5_p2",
    "kalman_cv_dt1.0_q1e-3_r1e-2",
    "ddpm_v1_masked_replace",
    "ddpm_v2_blend_alpha0.1",
    "ddpm_v3_inpainting",
]
METHOD_LABELS = {
    "linear_interp":                "Linear",
    "savgol_w5_p2":                 "Savitzky-Golay",
    "kalman_cv_dt1.0_q1e-3_r1e-2": "Kalman",
    "ddpm_v1_masked_replace":       "DDPM v1 (masked-replace)",
    "ddpm_v2_blend_alpha0.1":       f"DDPM v2 (blend α={BEST_ALPHA_V2})",
    "ddpm_v3_inpainting":           "DDPM v3 (RePaint inpainting)",
}

# ── Paths ──────────────────────────────────────────────────────────────────────
CTRL_DEGRAD = (
    PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark" / "degradation"
)
CTRL_RECON = (
    PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark" / "reconstruction"
)
EXP_ROOT   = PROJECT_ROOT / "outputs" / "stage3" / "inpainting_experiment"
PLOT_DIR   = EXP_ROOT / "trajectory_plots"
FULL_CSV   = EXP_ROOT / "full_results.csv"
VAR_CSV    = EXP_ROOT / "variance_decomposition.csv"
REPORT_MD  = EXP_ROOT / "REPORT.md"

# ── Imports ────────────────────────────────────────────────────────────────────
from tools.stage3.refinement.ddpm_refiner import (
    DDPMPriorInterfaceConfig,
    ddpm_prior_masked_replace_v1,
    ddpm_prior_masked_blend_v2,
    ddpm_prior_inpainting_v3,
)
from tools.stage3.controlled.evaluate_coarse_reconstruction import room3_map_meta


# ── Utility ────────────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path, allow_pickle=False)


# ── Per-trajectory metrics ─────────────────────────────────────────────────────
def compute_metrics_per_traj(
    clean: np.ndarray,     # (N, T, 2)
    pred: np.ndarray,      # (N, T, 2)
    obs_mask: np.ndarray,  # (N, T)  uint8
    map_meta: dict,
) -> dict[str, np.ndarray]:
    """Return one scalar per trajectory for each metric."""
    N, T, _ = clean.shape
    diff  = pred - clean                           # (N, T, 2)
    pe    = np.linalg.norm(diff, axis=-1)          # (N, T)
    ade   = pe.mean(axis=1)                        # (N,)
    rmse  = np.sqrt((diff ** 2).sum(axis=-1).mean(axis=1))  # (N,)

    miss_mask = obs_mask == 0                      # (N, T) bool
    has_miss  = miss_mask.any(axis=1)              # (N,) bool

    masked_ade  = np.full(N, np.nan)
    masked_rmse = np.full(N, np.nan)
    idx = np.where(has_miss)[0]
    for i in idx:
        m = miss_mask[i]
        masked_ade[i]  = pe[i, m].mean()
        masked_rmse[i] = np.sqrt((diff[i][m] ** 2).mean())

    # Geometry (empty room: no internal walls → wall_crossing_count = 0)
    xlo, xhi = map_meta["x_min"], map_meta["x_max"]
    ylo, yhi = map_meta["y_min"], map_meta["y_max"]
    outside  = (
        (pred[:, :, 0] < xlo) | (pred[:, :, 0] > xhi) |
        (pred[:, :, 1] < ylo) | (pred[:, :, 1] > yhi)
    )
    off_map = outside.mean(axis=1).astype(float)
    wall_x  = np.zeros(N, dtype=float)

    return {
        "ADE":                 ade,
        "RMSE":                rmse,
        "masked_ADE":          masked_ade,
        "masked_RMSE":         masked_rmse,
        "wall_crossing_count": wall_x,
        "off_map_ratio":       off_map,
    }


def summarize_arr(arr: np.ndarray) -> dict:
    """Summary statistics ignoring NaN."""
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        nan = float("nan")
        return dict(mean=nan, std=nan, min=nan, max=nan, median=nan, p25=nan, p75=nan)
    return dict(
        mean=float(np.mean(a)),
        std=float(np.std(a)),
        min=float(np.min(a)),
        max=float(np.max(a)),
        median=float(np.median(a)),
        p25=float(np.percentile(a, 25)),
        p75=float(np.percentile(a, 75)),
    )


# ── CSV builders ───────────────────────────────────────────────────────────────
FULL_FIELDS = [
    "method", "degradation", "metric",
    "mean", "std", "min", "max", "median", "p25", "p75",
    "n_trajectories", "n_seeds", "n_total",
]
VAR_FIELDS = [
    "method", "degradation", "metric",
    "std_across_trajectories", "std_across_seeds", "std_total",
]


def build_full_rows(
    method: str,
    degradation: str,
    metrics_ns: dict[str, np.ndarray],   # metric → (N,) or (N, S)
) -> list[dict]:
    rows = []
    for metric in METRIC_NAMES:
        arr = metrics_ns[metric]
        N   = arr.shape[0]
        S   = arr.shape[1] if arr.ndim == 2 else 1
        flat = arr.flatten()
        stats = summarize_arr(flat)
        n_total = int(np.isfinite(flat).sum())
        rows.append({
            "method": method, "degradation": degradation, "metric": metric,
            **stats,
            "n_trajectories": N, "n_seeds": S, "n_total": n_total,
        })
    return rows


def build_var_rows(
    method: str,
    degradation: str,
    metrics_ns: dict[str, np.ndarray],   # metric → (N,) or (N, S)
) -> list[dict]:
    rows = []
    for metric in METRIC_NAMES:
        arr = metrics_ns[metric]
        N   = arr.shape[0]
        S   = arr.shape[1] if arr.ndim == 2 else 1

        flat = arr.flatten()
        std_total = float(np.nanstd(flat[np.isfinite(flat)])) if np.isfinite(flat).any() else float("nan")

        if S > 1:
            # std_across_trajectories: per-seed std over N, averaged across seeds
            std_traj = float(np.mean([np.nanstd(arr[:, s]) for s in range(S)]))
            # std_across_seeds: per-trajectory std over seeds, averaged across trajs
            std_seed = float(np.mean([np.nanstd(arr[i, :]) for i in range(N)]))
        else:
            # Deterministic: only trajectory-level variance
            col = arr[:, 0] if arr.ndim == 2 else arr
            std_traj = float(np.nanstd(col[np.isfinite(col)]))
            std_seed = float("nan")

        rows.append({
            "method": method, "degradation": degradation, "metric": metric,
            "std_across_trajectories": std_traj,
            "std_across_seeds":        std_seed,
            "std_total":               std_total,
        })
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


# ── Trajectory plots ────────────────────────────────────────────────────────────
def draw_traj(ax, traj, obs_mask, color, lw=2.0, alpha=1.0, label=None):
    """Draw a 2-D trajectory; for degraded input only draw observed segments."""
    if obs_mask is not None:
        drawn = False
        for t in range(traj.shape[0] - 1):
            if obs_mask[t] and obs_mask[t + 1]:
                kw = {"label": label} if (not drawn and label) else {}
                ax.plot(traj[t:t + 2, 0], traj[t:t + 2, 1],
                        color=color, lw=lw, alpha=alpha, **kw)
                drawn = True
    else:
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, alpha=alpha, label=label)


def plot_case(
    pct: int,
    sample_idx: int,
    clean: np.ndarray,       # (T, 2)
    degraded: np.ndarray,    # (T, 2)
    coarse: np.ndarray,      # (T, 2)
    v3_samples: np.ndarray,  # (S, T, 2)
    obs_mask: np.ndarray,    # (T,)
    masked_ade_coarse: float,
    masked_ade_v3: float,
    span_start: int,
    span_end: int,
    out_path: Path,
) -> None:
    S = v3_samples.shape[0]
    v3_mean = v3_samples.mean(axis=0)   # (T, 2)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), squeeze=True)
    COLORS = ["#1f77b4", "#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c"]

    improvement = (masked_ade_coarse - masked_ade_v3) / (masked_ade_coarse + 1e-9) * 100
    col_titles = [
        "Col 1 – Clean Target",
        f"Col 2 – Degraded\n(span {span_start}:{span_end})",
        f"Col 3 – Coarse (Linear)\nmADE={masked_ade_coarse:.4f}",
        "Col 4 – v3 Single Sample",
        f"Col 5 – v3 Mean ({S} seeds)\nmADE={masked_ade_v3:.4f} ({improvement:+.1f}%)",
    ]

    # Col 1: clean target
    draw_traj(axes[0], clean, None, COLORS[0])
    # mark missing segment on clean
    axes[0].plot(clean[span_start:span_end + 1, 0],
                 clean[span_start:span_end + 1, 1],
                 "o--", color=COLORS[0], ms=4, alpha=0.6)

    # Col 2: degraded (observed frames only)
    draw_traj(axes[1], degraded, obs_mask, COLORS[1])

    # Col 3: coarse reconstruction
    draw_traj(axes[2], coarse, None, COLORS[2])
    axes[2].plot(coarse[span_start:span_end + 1, 0],
                 coarse[span_start:span_end + 1, 1],
                 "o", color=COLORS[2], ms=4, alpha=0.5)

    # Col 4: v3 single sample (seed 0)
    draw_traj(axes[3], v3_samples[0], None, COLORS[3])
    axes[3].plot(v3_samples[0, span_start:span_end + 1, 0],
                 v3_samples[0, span_start:span_end + 1, 1],
                 "o", color=COLORS[3], ms=4, alpha=0.5)

    # Col 5: v3 mean + all samples as thin lines
    for s in range(S):
        draw_traj(axes[4], v3_samples[s], None, COLORS[4], lw=0.8, alpha=0.3)
    draw_traj(axes[4], v3_mean, None, COLORS[4], lw=2.5)
    axes[4].plot(v3_mean[span_start:span_end + 1, 0],
                 v3_mean[span_start:span_end + 1, 1],
                 "o", color=COLORS[4], ms=4)

    for i, ax in enumerate(axes):
        ax.set_xlim(-0.15, 3.15)
        ax.set_ylim(-0.15, 3.15)
        ax.set_aspect("equal")
        ax.set_title(col_titles[i], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)

    fig.suptitle(
        f"p{pct} case  |  sample_idx={sample_idx}  |  span {span_start}:{span_end}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Report ─────────────────────────────────────────────────────────────────────
def _md_table(df_rows: list[dict], cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines  = [header, sep]
    for row in df_rows:
        cells = []
        for c in cols:
            val = row.get(c, "")
            if isinstance(val, float):
                f = fmt.get(c, ".5f")
                cells.append(f"{val:{f}}" if not (val != val) else "nan")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_report(
    full_rows: list[dict],
    var_rows: list[dict],
    n_trajectories: int,
    n_seeds: int,
    elapsed_sec: float,
) -> str:
    """Build REPORT.md content from result rows."""

    # ── Extract key tables ────────────────────────────────────────────────────
    def get_metric(rows, method, degradation, metric):
        for r in rows:
            if r["method"] == method and r["degradation"] == degradation and r["metric"] == metric:
                return r
        return {}

    # masked_ADE summary table for all methods × missing_only
    masked_ade_rows = [
        {
            "Method": METHOD_LABELS.get(m, m),
            **{
                DEGRADATION_LABELS[d]: f"{get_metric(full_rows, m, d, 'masked_ADE').get('mean', float('nan')):.5f}"
                for d in DEGRADATION_NAMES
            },
        }
        for m in METHOD_ORDER
    ]

    # Check whether v3 improves over v1/v2 on masked_ADE (across degradations)
    v3_beats_v1 = []
    v3_beats_v2 = []
    for d in DEGRADATION_NAMES:
        v3_mean   = get_metric(full_rows, "ddpm_v3_inpainting",    d, "masked_ADE").get("mean", float("inf"))
        v1_mean   = get_metric(full_rows, "ddpm_v1_masked_replace", d, "masked_ADE").get("mean", float("inf"))
        v2_mean   = get_metric(full_rows, "ddpm_v2_blend_alpha0.1", d, "masked_ADE").get("mean", float("inf"))
        v3_beats_v1.append(v3_mean < v1_mean)
        v3_beats_v2.append(v3_mean < v2_mean)

    n_beats_v1 = sum(v3_beats_v1)
    n_beats_v2 = sum(v3_beats_v2)
    hypothesis_supported = (n_beats_v1 >= 3) and (n_beats_v2 >= 3)

    verdict_str = "**SUPPORTED**" if hypothesis_supported else "**NOT SUPPORTED**"
    verdict_detail = (
        f"v3 outperforms v1 in {n_beats_v1}/4 degradations "
        f"and v2 in {n_beats_v2}/4 degradations on masked_ADE."
    )

    # Best method per degradation on masked_ADE
    best_rows = []
    for d in DEGRADATION_NAMES:
        d_rows = [(m, get_metric(full_rows, m, d, "masked_ADE").get("mean", float("inf")))
                  for m in METHOD_ORDER]
        best_m, best_v = min(d_rows, key=lambda x: x[1])
        best_rows.append({
            "Degradation": DEGRADATION_LABELS[d],
            "Best Method": METHOD_LABELS.get(best_m, best_m),
            "masked_ADE":  f"{best_v:.5f}",
        })

    DDPM_METHODS_SET = {"ddpm_v1_masked_replace", "ddpm_v2_blend_alpha0.1", "ddpm_v3_inpainting"}

    # Variance decomposition table
    var_table_rows = [
        {
            "Method": METHOD_LABELS.get(r["method"], r["method"]),
            "Degradation": DEGRADATION_LABELS.get(r["degradation"], r["degradation"]),
            "Metric": r["metric"],
            "std_across_trajectories": r["std_across_trajectories"],
            "std_across_seeds":        r["std_across_seeds"],
            "std_total":               r["std_total"],
        }
        for r in var_rows
        if r["method"] in DDPM_METHODS_SET and r["metric"] == "masked_ADE"
    ]

    # ── Build markdown ─────────────────────────────────────────────────────────
    lines = []

    lines += [
        "# Stage 3 Inpainting Experiment — Hypothesis Validation Report",
        "",
        f"*Generated automatically. Experiment completed in {elapsed_sec:.0f} s.*",
        "",
        "---",
        "",
        "## §1 Hypothesis",
        "",
        "**One-sentence hypothesis:**",
        "> The DDPM prior fails for trajectory reconstruction because it is trained *unconditionally*",
        "> and is applied *outside* the reverse sampling loop (v1/v2), preventing the missing segment",
        "> from being conditioned on the observed context during generation.",
        "",
        "**Why inpainting-style sampling should fix this:**",
        "RePaint (v3) clamps the observed relative displacements back to their forward-diffused level",
        "at every reverse step (t → t−1). This forces the reverse chain to remain consistent with the",
        "observed frames throughout the entire denoising trajectory, rather than generating a",
        "context-free sample and only applying the mask post-hoc (v1) or blending it weakly (v2).",
        "",
        "---",
        "",
        "## §2 Method Rationale",
        "",
        "| Method | Category | Description |",
        "| --- | --- | --- |",
        "| Linear | Classical baseline | Linear interpolation across the missing span |",
        "| Savitzky-Golay | Classical baseline | SG filter with w=5, p=2, then gap-fill |",
        "| Kalman | Classical baseline | Constant-velocity Kalman filter |",
        f"| DDPM v1 | Learned (post-hoc) | v0 single-shot projection → hard-replace missing |",
        f"| DDPM v2 | Learned (post-hoc) | v0 projection → blend α={BEST_ALPHA_V2} on missing |",
        f"| DDPM v3 | Learned (inpainting) | RePaint-style: clamp known frames at every reverse step |",
        "",
        "**v3 known limitations:**",
        "- The prior was trained on ETH+UCY relative displacements; room3 coordinates are a",
        "  linearly rescaled version of the same data. Any distributional shift between training",
        "  and test coordinate scales will affect all DDPM variants equally.",
        "- The prior is *unconditional* — it models the marginal p(trajectory), not",
        "  p(trajectory | start, end). v3 constrains the observed steps but does not inject",
        "  endpoint information in a structured way.",
        "- `num_samples_per_traj=5` seeds are reported; spread estimates may be noisy.",
        "",
        "---",
        "",
        "## §3 Experiment Setup",
        "",
        f"- **Data source:** `datasets/processed/data_eth_ucy_20.npy` normalized to",
        "  canonical room3 [0,3]×[0,3] via `build_imputation_dataset.py`.",
        f"- **N trajectories used:** {n_trajectories} (first {n_trajectories} of 36073;",
        "  chosen to keep wall-clock time under 10 min on CPU).",
        f"- **Degradation pipeline:** 4 synthetic settings (missing_only, missing_noise,",
        "  missing_drift, missing_noise_drift) with fixed span [8,11] (4 frames / 20%",
        "  of T=20), seed=42.",
        f"- **Metrics:** ADE, RMSE (full trajectory); masked_ADE, masked_RMSE (missing",
        "  segment only); wall_crossing_count, off_map_ratio (geometry, empty-room).",
        "- **Averaging:** For deterministic methods (baselines, v1, v2), one value per",
        "  trajectory → statistics across N trajectories. For v3, each trajectory has",
        f"  {n_seeds} independent reverse-process samples → per-trajectory mean reported",
        "  in the primary table; raw (N, S) values used for variance decomposition.",
        "- **Geometry declaration:** `wall_crossing_count` and `off_map_ratio` are",
        "  *evaluation-only* metrics. No geometry conditioning or loss term is used in",
        "  any method. Empty room3 has no internal walls, so wall_crossing_count = 0",
        "  for all methods by construction.",
        "",
        "---",
        "",
        "## §4 Full Results",
        "",
        "### masked_ADE by method and degradation (primary metric)",
        "",
    ]

    # masked_ADE table
    cols4 = ["Method"] + [DEGRADATION_LABELS[d] for d in DEGRADATION_NAMES]
    lines.append(_md_table(masked_ade_rows, cols4))
    lines += ["", "### Best method per degradation (masked_ADE)", ""]
    lines.append(_md_table(best_rows, ["Degradation", "Best Method", "masked_ADE"]))
    lines += [
        "",
        "*(See `full_results.csv` for complete statistics including ADE, RMSE, masked_RMSE,*",
        " *wall_crossing_count, off_map_ratio, n_trajectories, n_seeds, n_total for every*",
        " *method × degradation × metric combination.)*",
        "",
        "---",
        "",
        "## §5 Figures",
        "",
        "Trajectory visualisations are in `trajectory_plots/`. Five cases are shown,",
        "selected at the p10, p25, p50, p75, p90 percentiles of v3 masked_ADE on",
        "`missing_only` degradation (to avoid cherry-picking).",
        "",
        "Each figure shows 5 columns:",
        "- **Col 1** – Clean target (ground truth, with missing segment dotted)",
        "- **Col 2** – Degraded input (observed frames only)",
        "- **Col 3** – Coarse reconstruction (Linear interpolation)",
        "- **Col 4** – v3 single sample (seed 0)",
        f"- **Col 5** – v3 mean over {n_seeds} seeds (thin lines = individual samples,",
        "  thick line = mean)",
        "",
        "Figure annotation: `sample_idx`, `span_start:span_end`, `masked_ADE_coarse`,",
        "`masked_ADE_v3`, `improvement_pct` (negative = worse).",
        "",
        "**Figure 3 / Figure 5 spread note:**",
        "The `std_across_seeds` column in `variance_decomposition.csv` directly quantifies",
        "within-trajectory spread across the 5 reverse-diffusion seeds. This answers the",
        "question of whether observed spread in trajectory figures reflects population",
        "heterogeneity (std_across_trajectories) or DDPM stochasticity (std_across_seeds).",
        "",
        "---",
        "",
        "## §6 Discussion",
        "",
        f"### Verdict: hypothesis is {verdict_str}",
        "",
        f"*{verdict_detail}*",
        "",
    ]

    if hypothesis_supported:
        lines += [
            "**Interpretation (hypothesis supported):**",
            "Clamping observed frames at every reverse step consistently improves masked_ADE",
            "over the post-hoc projection approaches (v1/v2). This is strong evidence that",
            "the prior-task mismatch identified in §1 is real: when the reverse chain can",
            "'see' the boundary observations, it generates missing segments that connect",
            "smoothly to the context.",
            "",
            "**However, caveats remain:**",
            "1. Classical baselines (Linear, SG) may still outperform v3 on clean missing-only",
            "   cases, because they directly interpolate between observed endpoints while v3",
            "   must discover the constraint implicitly through the inpainting clamp.",
            "2. If v3 does not outperform classical baselines, the residual gap is likely due",
            "   to coordinate-scale domain mismatch (ETH+UCY training scale ≠ room3 scale).",
        ]
    else:
        lines += [
            "**Interpretation (hypothesis not supported):**",
            "The inpainting-style conditioning did not consistently improve over the post-hoc",
            "projection baselines (v1/v2). Two plausible next-level explanations:",
            "",
            "1. **Coordinate-scale domain mismatch.** The prior was trained on ETH+UCY",
            "   relative displacements. Room3 data is the same dataset rescaled to [0,3]×[0,3].",
            "   The scale of relative steps in room3 differs from the training distribution.",
            "   Even with perfect inpainting conditioning, the prior's generative mode may not",
            "   produce room3-scale displacements. Fix: retrain on room3-scale data, or",
            "   normalize inputs to the prior's training distribution before inpainting.",
            "",
            "2. **Short sequence / large missing fraction.** With T=20 and span_len=4 (20%),",
            "   the observed context is limited (8 frames on one side, 8 on the other). A",
            "   simple interpolation (Linear) that directly uses the boundary endpoints will",
            "   dominate. The DDPM prior adds unnecessary distributional noise on top of what",
            "   is essentially a short-range interpolation problem.",
        ]

    lines += [
        "",
        "---",
        "",
        "## §7 Next Hypothesis",
        "",
    ]

    if hypothesis_supported:
        lines += [
            "**Next hypothesis:** The residual gap between v3 and classical baselines is caused",
            "by coordinate-scale domain mismatch between the ETH+UCY training distribution and",
            "the room3 test distribution. Retraining (or fine-tuning) the DDPM prior on",
            "room3-scale trajectories should close this gap.",
            "",
            "*Proposed validation:* Rescale room3 inputs to the ETH+UCY training scale before",
            "feeding to the prior (and rescale outputs back). If this closes the gap without",
            "retraining, the scale mismatch is confirmed as the primary remaining factor.",
        ]
    else:
        lines += [
            "**Next hypothesis:** The primary failure mode is coordinate-scale domain mismatch.",
            "Normalizing room3 relative displacements to the ETH+UCY training distribution",
            "(mean / std standardization using training statistics) before calling the DDPM",
            "prior, and denormalizing afterward, should substantially improve inpainting v3.",
            "",
            "*Proposed validation:* Compute mean and std of relative steps from",
            "`data_eth_ucy_20_rel.npy`; apply z-score normalization to room3 inputs before",
            "the DDPM reverse chain; apply inverse normalization to outputs. Re-run this",
            "experiment and compare masked_ADE before and after normalization.",
        ]

    lines += [
        "",
        "---",
        "",
        "## §8 Variance Decomposition (masked_ADE for DDPM methods)",
        "",
        "| Method | Degradation | std_across_trajectories | std_across_seeds | std_total |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in var_table_rows:
        v = r['std_across_seeds']
        if isinstance(v, str):
            std_seed_str = v
        elif v != v:  # NaN check
            std_seed_str = "nan"
        else:
            std_seed_str = f"{v:.5f}"
        lines.append(
            f"| {r['Method']} | {r['Degradation']} | "
            f"{r['std_across_trajectories']:.5f} | "
            f"{std_seed_str} | "
            f"{r['std_total']:.5f} |"
        )

    lines += [
        "",
        "*(std_across_seeds = nan for deterministic methods v1 and v2)*",
        "",
    ]

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_dirs()
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data (first {N_EXPERIMENT} trajectories) …")
    clean   = load_npy(CTRL_DEGRAD / "clean.npy")[:N_EXPERIMENT]                # (N, T, 2)
    mask    = load_npy(CTRL_DEGRAD / f"mask_{TAG}.npy")[:N_EXPERIMENT]          # (N, T)
    N, T, _ = clean.shape

    # Span boundaries (fixed for all trajectories)
    miss_cols   = np.where(mask[0] == 0)[0]
    span_start  = int(miss_cols[0])
    span_end    = int(miss_cols[-1])

    map_meta = room3_map_meta()
    config   = DDPMPriorInterfaceConfig(device="auto")

    all_full_rows: list[dict] = []
    all_var_rows:  list[dict] = []

    # Per-degradation storage for v3 (for trajectory plots on missing_only)
    v3_metrics_missing_only: dict[str, np.ndarray] | None = None
    v3_preds_missing_only:   np.ndarray | None = None
    degraded_missing_only:   np.ndarray | None = None

    DDPM_METHODS = {"ddpm_v1_masked_replace", "ddpm_v2_blend_alpha0.1", "ddpm_v3_inpainting"}

    for deg in DEGRADATION_NAMES:
        print(f"\n{'='*60}")
        print(f"Degradation: {deg}")
        print('='*60)

        degraded = load_npy(CTRL_DEGRAD / f"degraded_{deg}_{TAG}.npy")[:N_EXPERIMENT]

        # Coarse reconstruction files
        recon: dict[str, np.ndarray] = {}
        for baseline in ["linear_interp", "savgol_w5_p2", "kalman_cv_dt1.0_q1e-3_r1e-2"]:
            recon[baseline] = load_npy(
                CTRL_RECON / f"recon_{deg}_{baseline}_{TAG}.npy"
            )[:N_EXPERIMENT]

        # ── Baselines ─────────────────────────────────────────────────────────
        for method, pred in recon.items():
            print(f"  {method} …", end=" ", flush=True)
            m = compute_metrics_per_traj(clean, pred, mask, map_meta)
            m_1d = {k: v[:, None] for k, v in m.items()}   # shape (N,1) for uniformity
            all_full_rows.extend(build_full_rows(method, deg, m_1d))
            all_var_rows.extend(build_var_rows(method, deg, m_1d))
            print(f"masked_ADE={m['masked_ADE'].mean():.4f}")

        # ── DDPM v1 (masked replace) ───────────────────────────────────────────
        print("  ddpm_v1_masked_replace …", end=" ", flush=True)
        coarse_for_ddpm = recon["linear_interp"]
        v1_pred, _ = ddpm_prior_masked_replace_v1(coarse_for_ddpm, mask, config=config)
        v1_m = compute_metrics_per_traj(clean, v1_pred, mask, map_meta)
        v1_m_1d = {k: v[:, None] for k, v in v1_m.items()}
        all_full_rows.extend(build_full_rows("ddpm_v1_masked_replace", deg, v1_m_1d))
        all_var_rows.extend(build_var_rows("ddpm_v1_masked_replace", deg, v1_m_1d))
        print(f"masked_ADE={v1_m['masked_ADE'].mean():.4f}")

        # ── DDPM v2 (masked blend) ────────────────────────────────────────────
        print(f"  ddpm_v2_blend_alpha{BEST_ALPHA_V2} …", end=" ", flush=True)
        v2_pred, _ = ddpm_prior_masked_blend_v2(
            coarse_for_ddpm, mask, alpha=BEST_ALPHA_V2, config=config
        )
        v2_m = compute_metrics_per_traj(clean, v2_pred, mask, map_meta)
        v2_m_1d = {k: v[:, None] for k, v in v2_m.items()}
        all_full_rows.extend(build_full_rows("ddpm_v2_blend_alpha0.1", deg, v2_m_1d))
        all_var_rows.extend(build_var_rows("ddpm_v2_blend_alpha0.1", deg, v2_m_1d))
        print(f"masked_ADE={v2_m['masked_ADE'].mean():.4f}")

        # ── DDPM v3 (RePaint inpainting) ──────────────────────────────────────
        print(f"  ddpm_v3_inpainting ({NUM_SEEDS_V3} seeds × 100 steps) …", flush=True)
        t_v3 = time.time()
        v3_preds = ddpm_prior_inpainting_v3(
            degraded,
            mask,
            num_samples_per_traj=NUM_SEEDS_V3,
            seed_base=SEED_BASE_V3,
            config=config,
        )   # (N, S, T, 2)
        dt_v3 = time.time() - t_v3
        print(f"    done in {dt_v3:.1f} s")

        # Per-seed metrics → (N, S) arrays
        seed_metric_lists: dict[str, list[np.ndarray]] = {k: [] for k in METRIC_NAMES}
        for s in range(NUM_SEEDS_V3):
            m_s = compute_metrics_per_traj(clean, v3_preds[:, s], mask, map_meta)
            for k in METRIC_NAMES:
                seed_metric_lists[k].append(m_s[k])
        v3_metrics_ns: dict[str, np.ndarray] = {
            k: np.stack(v, axis=1) for k, v in seed_metric_lists.items()
        }   # (N, S)

        all_full_rows.extend(build_full_rows("ddpm_v3_inpainting", deg, v3_metrics_ns))
        all_var_rows.extend(build_var_rows("ddpm_v3_inpainting", deg, v3_metrics_ns))
        print(f"  masked_ADE={v3_metrics_ns['masked_ADE'].mean():.4f}")

        if deg == "missing_only":
            v3_metrics_missing_only = v3_metrics_ns
            v3_preds_missing_only   = v3_preds
            degraded_missing_only   = degraded

    # ── Write CSVs ───────────────────────────────────────────────────────────
    print("\nWriting CSVs …")
    write_csv(FULL_CSV, FULL_FIELDS, all_full_rows)
    write_csv(VAR_CSV,  VAR_FIELDS,  all_var_rows)
    print(f"  full_results.csv   → {FULL_CSV}")
    print(f"  variance_decomp.csv → {VAR_CSV}")

    # ── Trajectory plots ─────────────────────────────────────────────────────
    print("\nGenerating trajectory plots …")
    if v3_metrics_missing_only is not None and v3_preds_missing_only is not None:
        # Per-trajectory masked_ADE for v3: mean over seeds
        v3_mADE_per_traj = np.nanmean(v3_metrics_missing_only["masked_ADE"], axis=1)   # (N,)

        # Coarse (linear) masked_ADE for same degradation
        lin_mADE_per_traj = np.array([
            r["mean"] for r in all_full_rows
            if r["method"] == "linear_interp"
            and r["degradation"] == "missing_only"
            and r["metric"] == "masked_ADE"
        ])
        # Recompute per-traj coarse for plot
        lin_pred = load_npy(
            CTRL_RECON / f"recon_missing_only_linear_interp_{TAG}.npy"
        )[:N_EXPERIMENT]
        lin_m = compute_metrics_per_traj(clean, lin_pred, mask, map_meta)
        coarse_mADE_per_traj = lin_m["masked_ADE"]   # (N,)

        percentiles = [10, 25, 50, 75, 90]
        for pct in percentiles:
            threshold = np.nanpercentile(v3_mADE_per_traj, pct)
            # Find trajectory closest to that percentile
            dists     = np.abs(v3_mADE_per_traj - threshold)
            idx       = int(np.nanargmin(dists))

            out_path = PLOT_DIR / f"case_p{pct}_sample{idx}.png"
            plot_case(
                pct=pct,
                sample_idx=idx,
                clean=clean[idx],
                degraded=degraded_missing_only[idx],
                coarse=lin_pred[idx],
                v3_samples=v3_preds_missing_only[idx],   # (S, T, 2)
                obs_mask=mask[idx],
                masked_ade_coarse=float(coarse_mADE_per_traj[idx]),
                masked_ade_v3=float(v3_mADE_per_traj[idx]),
                span_start=span_start,
                span_end=span_end,
                out_path=out_path,
            )
            print(f"  p{pct}  sample_idx={idx}  mADE_coarse={coarse_mADE_per_traj[idx]:.4f}"
                  f"  mADE_v3={v3_mADE_per_traj[idx]:.4f}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\nGenerating REPORT.md …")
    elapsed = time.time() - t0
    report_txt = generate_report(
        full_rows=all_full_rows,
        var_rows=all_var_rows,
        n_trajectories=N,
        n_seeds=NUM_SEEDS_V3,
        elapsed_sec=elapsed,
    )
    REPORT_MD.write_text(report_txt, encoding="utf-8")
    print(f"  {REPORT_MD}")

    # ── Self-check ────────────────────────────────────────────────────────────
    print("\n── Self-check ──────────────────────────────────────────────────────")
    expected_rows = len(METHOD_ORDER) * len(DEGRADATION_NAMES) * len(METRIC_NAMES)
    print(f"  full_results.csv rows: {len(all_full_rows)} (expected {expected_rows})")
    missing_combos = [
        (m, d, k)
        for m in METHOD_ORDER
        for d in DEGRADATION_NAMES
        for k in METRIC_NAMES
        if not any(r["method"] == m and r["degradation"] == d and r["metric"] == k
                   for r in all_full_rows)
    ]
    if missing_combos:
        print(f"  MISSING CELLS: {missing_combos}")
    else:
        print("  ✓ All method × degradation × metric cells present")

    plot_files = list(PLOT_DIR.glob("case_p*.png"))
    print(f"  Trajectory plots: {len(plot_files)} (expected 5)")
    for f in sorted(plot_files):
        print(f"    {f.name}")

    print(f"\nTotal elapsed: {elapsed:.0f} s")
    print("Done.")


if __name__ == "__main__":
    main()
