from __future__ import annotations

import argparse
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

from tools.stage3.baselines.run_kalman import kalman_reconstruct, validate_sample as validate_kalman_sample
from tools.stage3.baselines.run_linear_interp import interpolate_sample
from tools.stage3.baselines.run_savgol import validate_and_interp
from tools.stage3.controlled.evaluate_coarse_reconstruction import room3_map_meta
from tools.stage3.refinement.ddpm_refiner import (
    DDPMPriorInterfaceConfig,
    ddpm_conditional_sample_v4,
    ddpm_prior_inpainting_v3,
)


N_EXPERIMENT = 1024
NUM_SEEDS_DEFAULT = 5
SEED_BASE_DEFAULT = 42
DEGRADATION_DEFAULT = "missing_only"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "conditional_experiment"
PLOT_DIR = OUTPUT_ROOT / "trajectory_plots"
CTRL_DEGRAD = PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark" / "degradation"
CTRL_RECON = PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark" / "reconstruction"

METHOD_LABELS = {
    "linear_interp": "Linear interpolation",
    "linear_extrapolate": "Linear extrapolation",
    "ddpm_v3_anchored": "DDPM v3 anchored",
    "ddpm_v4_conditional": "DDPM v4 conditional",
}
METRIC_NAMES = [
    "ADE",
    "RMSE",
    "FDE",
    "masked_ADE",
    "masked_RMSE",
    "endpoint_error",
    "path_length_error",
    "acceleration_error",
    "wall_crossing_count",
    "off_map_ratio",
]


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path, allow_pickle=False)


def build_obs_mask(num_samples: int, seq_len: int, gap_type: str) -> tuple[np.ndarray, int, int]:
    obs_mask = np.ones((num_samples, seq_len), dtype=np.uint8)
    if gap_type == "interior":
        span_start, span_end = 8, 11
    elif gap_type == "suffix":
        span_start, span_end = 16, 19
    else:
        raise ValueError(f"Unsupported gap_type: {gap_type}")
    obs_mask[:, span_start : span_end + 1] = 0
    return obs_mask, span_start, span_end


def build_degraded_missing_only(clean: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    degraded = clean.copy()
    degraded[obs_mask == 0] = np.nan
    return degraded


def run_linear_interp_batch(traj_obs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(traj_obs, dtype=np.float32)
    for idx in range(traj_obs.shape[0]):
        out[idx] = interpolate_sample(traj_obs[idx], obs_mask[idx], index=idx)
    return out


def run_linear_extrapolate_batch(traj_obs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    out = traj_obs.copy().astype(np.float32)
    for idx in range(traj_obs.shape[0]):
        obs = obs_mask[idx]
        last_obs_idx = int(np.where(obs == 1)[0][-1])
        if last_obs_idx >= 1:
            velocity = traj_obs[idx, last_obs_idx] - traj_obs[idx, last_obs_idx - 1]
        else:
            velocity = np.zeros(2, dtype=np.float32)
        pred = traj_obs[idx].copy()
        for t in range(last_obs_idx + 1, len(obs)):
            steps = t - last_obs_idx
            pred[t] = traj_obs[idx, last_obs_idx] + steps * velocity
        out[idx] = pred.astype(np.float32)
    return out


def run_savgol_batch(traj_obs: np.ndarray, obs_mask: np.ndarray, window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    from scipy.signal import savgol_filter

    out = np.zeros_like(traj_obs, dtype=np.float32)
    for idx in range(traj_obs.shape[0]):
        filled = validate_and_interp(traj_obs[idx], obs_mask[idx], index=idx)
        for dim in range(filled.shape[1]):
            out[idx, :, dim] = savgol_filter(
                filled[:, dim],
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            ).astype(np.float32)
    return out


def run_kalman_batch(
    traj_obs: np.ndarray,
    obs_mask: np.ndarray,
    dt: float = 1.0,
    process_var: float = 1e-3,
    measure_var: float = 1e-2,
) -> np.ndarray:
    out = np.zeros_like(traj_obs, dtype=np.float32)
    for idx in range(traj_obs.shape[0]):
        validate_kalman_sample(obs_mask[idx], index=idx)
        out[idx] = kalman_reconstruct(
            traj_obs=traj_obs[idx],
            mask=obs_mask[idx],
            dt=dt,
            process_var=process_var,
            measure_var=measure_var,
        )
    return out


def anchor_missing_spans(pred_ns: np.ndarray, observed_abs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    pred_ns = pred_ns.astype(np.float32, copy=True)
    n, s, _, _ = pred_ns.shape
    for traj_idx in range(n):
        missing_idx = np.where(obs_mask[traj_idx] == 0)[0]
        if missing_idx.size == 0:
            pred_ns[traj_idx] = observed_abs[traj_idx][None, :, :]
            continue
        left = int(missing_idx[0] - 1) if missing_idx[0] > 0 else None
        right = int(missing_idx[-1] + 1) if missing_idx[-1] + 1 < obs_mask.shape[1] else None
        observed_bool = obs_mask[traj_idx] == 1
        for seed_idx in range(s):
            if left is not None and right is not None:
                segment = pred_ns[traj_idx, seed_idx, left : right + 1].copy()
                seg_len = right - left
                left_delta = observed_abs[traj_idx, left] - segment[0]
                right_delta = observed_abs[traj_idx, right] - segment[-1]
                for local_idx in range(seg_len + 1):
                    lam = local_idx / seg_len if seg_len > 0 else 0.0
                    segment[local_idx] = segment[local_idx] + (1.0 - lam) * left_delta + lam * right_delta
                segment[0] = observed_abs[traj_idx, left]
                segment[-1] = observed_abs[traj_idx, right]
                pred_ns[traj_idx, seed_idx, left : right + 1] = segment
            pred_ns[traj_idx, seed_idx, observed_bool] = observed_abs[traj_idx, observed_bool]
    return pred_ns


def compute_metrics_per_traj(clean: np.ndarray, pred: np.ndarray, obs_mask: np.ndarray, map_meta: dict) -> dict[str, np.ndarray]:
    n, _, _ = clean.shape
    diff = pred - clean
    pe = np.linalg.norm(diff, axis=-1)
    ade = pe.mean(axis=1)
    rmse = np.sqrt((diff ** 2).sum(axis=-1).mean(axis=1))
    fde = pe[:, -1]

    miss_mask = obs_mask == 0
    masked_ade = np.full(n, np.nan, dtype=np.float64)
    masked_rmse = np.full(n, np.nan, dtype=np.float64)
    endpoint_error = np.full(n, np.nan, dtype=np.float64)
    path_length_error = np.full(n, np.nan, dtype=np.float64)
    acceleration_error = np.full(n, np.nan, dtype=np.float64)
    for idx in range(n):
        m = miss_mask[idx]
        if not np.any(m):
            continue
        masked_ade[idx] = pe[idx, m].mean()
        masked_rmse[idx] = np.sqrt((diff[idx][m] ** 2).mean())
        last_missing = int(np.where(m)[0][-1])
        endpoint_error[idx] = pe[idx, last_missing]
        pred_len = np.linalg.norm(np.diff(pred[idx], axis=0), axis=-1).sum()
        clean_len = np.linalg.norm(np.diff(clean[idx], axis=0), axis=-1).sum()
        path_length_error[idx] = abs(pred_len - clean_len)
        pred_acc = np.diff(pred[idx], n=2, axis=0)
        clean_acc = np.diff(clean[idx], n=2, axis=0)
        acceleration_error[idx] = np.sqrt(((pred_acc - clean_acc) ** 2).sum(axis=-1).mean())

    xlo, xhi = map_meta["x_min"], map_meta["x_max"]
    ylo, yhi = map_meta["y_min"], map_meta["y_max"]
    outside = (
        (pred[:, :, 0] < xlo) | (pred[:, :, 0] > xhi) |
        (pred[:, :, 1] < ylo) | (pred[:, :, 1] > yhi)
    )
    off_map = outside.mean(axis=1).astype(float)
    wall_x = np.zeros(n, dtype=float)

    return {
        "ADE": ade,
        "RMSE": rmse,
        "FDE": fde,
        "masked_ADE": masked_ade,
        "masked_RMSE": masked_rmse,
        "endpoint_error": endpoint_error,
        "path_length_error": path_length_error,
        "acceleration_error": acceleration_error,
        "wall_crossing_count": wall_x,
        "off_map_ratio": off_map,
    }


def summarize_arr(arr: np.ndarray) -> dict[str, float]:
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "median": nan, "p25": nan, "p75": nan}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
    }


FULL_FIELDS = [
    "method", "degradation", "gap_type", "metric",
    "mean", "std", "min", "max", "median", "p25", "p75",
    "n_trajectories", "n_seeds", "n_total",
]
VAR_FIELDS = [
    "method", "degradation", "gap_type", "metric",
    "std_across_trajectories", "std_across_seeds", "std_total",
]


def build_full_rows(method: str, degradation: str, gap_type: str, metrics_ns: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    for metric in METRIC_NAMES:
        arr = metrics_ns[metric]
        n = arr.shape[0]
        s = arr.shape[1] if arr.ndim == 2 else 1
        flat = arr.flatten()
        stats = summarize_arr(flat)
        rows.append(
            {
                "method": method,
                "degradation": degradation,
                "gap_type": gap_type,
                "metric": metric,
                **stats,
                "n_trajectories": n,
                "n_seeds": s,
                "n_total": int(np.isfinite(flat).sum()),
            }
        )
    return rows


def build_var_rows(method: str, degradation: str, gap_type: str, metrics_ns: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    for metric in METRIC_NAMES:
        arr = metrics_ns[metric]
        n = arr.shape[0]
        s = arr.shape[1] if arr.ndim == 2 else 1
        flat = arr.flatten()
        std_total = float(np.nanstd(flat[np.isfinite(flat)])) if np.isfinite(flat).any() else float("nan")
        if s > 1:
            std_traj = float(np.mean([np.nanstd(arr[:, seed]) for seed in range(s)]))
            std_seed = float(np.mean([np.nanstd(arr[idx, :]) for idx in range(n)]))
        else:
            col = arr[:, 0] if arr.ndim == 2 else arr
            std_traj = float(np.nanstd(col[np.isfinite(col)]))
            std_seed = float("nan")
        rows.append(
            {
                "method": method,
                "degradation": degradation,
                "gap_type": gap_type,
                "metric": metric,
                "std_across_trajectories": std_traj,
                "std_across_seeds": std_seed,
                "std_total": std_total,
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def draw_traj(ax, traj, obs_mask, color, lw=2.0, alpha=1.0, label=None):
    if obs_mask is not None:
        drawn = False
        for t in range(traj.shape[0] - 1):
            if obs_mask[t] and obs_mask[t + 1]:
                kw = {"label": label} if (not drawn and label) else {}
                ax.plot(traj[t:t + 2, 0], traj[t:t + 2, 1], color=color, lw=lw, alpha=alpha, **kw)
                drawn = True
    else:
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, alpha=alpha, label=label)


def plot_case(
    title: str,
    sample_idx: int,
    clean: np.ndarray,
    degraded: np.ndarray,
    coarse: np.ndarray,
    ddpm_candidate: np.ndarray,
    final_refined: np.ndarray,
    obs_mask: np.ndarray,
    span_start: int,
    span_end: int,
    out_path: Path,
    delta_masked_ade: float,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), squeeze=True)
    cols = [
        ("Clean target", clean, None, "#1f77b4"),
        ("Degraded input", degraded, obs_mask, "#7f7f7f"),
        ("Coarse reconstruction", coarse, None, "#d62728"),
        ("DDPM candidate", ddpm_candidate, None, "#ff7f0e"),
        ("Final refined output", final_refined, None, "#2ca02c"),
    ]
    for ax, (col_title, traj, mask, color) in zip(axes, cols):
        draw_traj(ax, traj, mask, color)
        ax.plot(traj[span_start:span_end + 1, 0], traj[span_start:span_end + 1, 1], "o", color=color, ms=4, alpha=0.6)
        ax.set_xlim(-0.15, 3.15)
        ax.set_ylim(-0.15, 3.15)
        ax.set_aspect("equal")
        ax.set_title(col_title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.5)
    fig.suptitle(f"{title} | sample_idx={sample_idx} | span={span_start}:{span_end} | delta_masked_ADE={delta_masked_ade:+.4f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_summary_mean(rows: list[dict], method: str, metric: str) -> float:
    for row in rows:
        if row["method"] == method and row["metric"] == metric:
            return float(row["mean"])
    return float("nan")


def select_and_plot_cases(
    clean: np.ndarray,
    degraded: np.ndarray,
    coarse: np.ndarray,
    cond_preds: np.ndarray,
    cond_metrics: dict[str, np.ndarray],
    obs_mask: np.ndarray,
    span_start: int,
    span_end: int,
    gap_type: str,
) -> list[dict]:
    cond_mean = np.nanmean(cond_metrics["masked_ADE"], axis=1)
    coarse_metric = compute_metrics_per_traj(clean, coarse, obs_mask, room3_map_meta())["masked_ADE"]
    improvement = coarse_metric - cond_mean

    threshold = np.nanpercentile(cond_mean, 50)
    median_idx = int(np.nanargmin(np.abs(cond_mean - threshold)))
    best_idx = int(np.nanargmax(improvement))
    worst_idx = int(np.nanargmin(improvement))
    cases = [
        ("median", median_idx, PLOT_DIR / f"{gap_type}_case_median.png"),
        ("best_improvement", best_idx, PLOT_DIR / f"{gap_type}_case_best_improvement.png"),
        ("worst_degradation", worst_idx, PLOT_DIR / f"{gap_type}_case_worst_degradation.png"),
    ]
    selected = []
    for name, idx, out_path in cases:
        final_refined = np.nanmean(cond_preds[idx], axis=0)
        ddpm_candidate = cond_preds[idx, 0]
        delta = float(coarse_metric[idx] - cond_mean[idx])
        plot_case(
            title=f"{gap_type} | conditional v4 | {name}",
            sample_idx=idx,
            clean=clean[idx],
            degraded=degraded[idx],
            coarse=coarse[idx],
            ddpm_candidate=ddpm_candidate,
            final_refined=final_refined,
            obs_mask=obs_mask[idx],
            span_start=span_start,
            span_end=span_end,
            out_path=out_path,
            delta_masked_ade=delta,
        )
        selected.append(
            {
                "name": name,
                "sample_idx": idx,
                "plot_path": str(out_path),
                "delta_masked_ADE_definition": "coarse_masked_ADE - final_masked_ADE",
                "delta_masked_ADE": delta,
            }
        )
    return selected


def generate_report(
    args,
    full_rows: list[dict],
    var_rows: list[dict],
    selected_cases: list[dict],
    elapsed_sec: float,
    span_start: int,
    span_end: int,
) -> str:
    lines = [
        "# Conditional DDPM Experiment Report",
        "",
        f"*Generated automatically in {elapsed_sec:.0f} s for gap_type=`{args.gap_type}`.*",
        "",
        "## §1 Goal",
        "",
        "Test whether moving conditioning into the DDPM training stage improves missing-trajectory reconstruction quality relative to the existing unconditional inpainting interface.",
        "",
        "## §2 Protocol",
        "",
        f"- dataset: canonical room3 (`outputs/stage3/controlled_benchmark/degradation/clean.npy`), first {N_EXPERIMENT} trajectories",
        f"- degradation: `{args.degradation}`",
        f"- gap_type: `{args.gap_type}`",
        f"- span: frames {span_start}-{span_end} inclusive",
        f"- methods: `{args.methods}`",
        f"- n_seeds per DDPM method: `{args.n_seeds}`",
        "",
        "## §3 Methods",
        "",
        "- `ddpm_v3_anchored`: existing unconditional DDPM inpainting plus endpoint/observed anchoring",
        "- `ddpm_v4_conditional`: conditional DDPM trained with observed mask and masked observations concatenated into the denoiser input",
        "- `linear_interp` / `linear_extrapolate`: deterministic coarse baseline depending on gap type",
        "",
        "## §4 Primary Results (masked_ADE mean)",
        "",
    ]
    lines.append("| Method | masked_ADE mean | masked_ADE std | FDE mean |")
    lines.append("| --- | --- | --- | --- |")
    for method in args.methods.split(","):
        m_ade = get_summary_mean([r for r in full_rows if r["metric"] == "masked_ADE"], method, "masked_ADE")
        fde = get_summary_mean([r for r in full_rows if r["metric"] == "FDE"], method, "FDE")
        std = next(r["std"] for r in full_rows if r["method"] == method and r["metric"] == "masked_ADE")
        lines.append(f"| {method} | {m_ade:.6f} | {std:.6f} | {fde:.6f} |")
    lines += [
        "",
        "## §5 Variance",
        "",
        "See `variance_decomposition_*.csv`. `std_across_trajectories` averages per-seed trajectory spread; `std_across_seeds` averages per-trajectory seed spread; `std_total` mixes both sources.",
        "",
        "## §6 Evidence Chain",
        "",
        "1. Compare deterministic coarse baseline against `ddpm_v3_anchored` to see whether the existing unconditional inpainting pipeline helps at all under the selected gap type.",
        "2. Compare `ddpm_v3_anchored` against `ddpm_v4_conditional` to test whether moving conditioning into training improves the learned reconstruction interface.",
        "3. Inspect representative five-column trajectory figures to verify whether gains come from cleaner missing-span geometry rather than trivial endpoint restoration.",
        "",
        "## §7 Verdict Rule",
        "",
        "- Primary pass criterion: `ddpm_v4_conditional` has lower `masked_ADE` mean than `ddpm_v3_anchored` on the same gap type.",
        "- Secondary support: `std_across_seeds` should not explode relative to v3.",
        "- Practical support: representative figures should show smoother missing-span reconstruction, not only endpoint clamping.",
        "",
        "## Representative Cases",
        "",
    ]
    for case in selected_cases:
        lines.append(f"- {case['name']}: sample_idx={case['sample_idx']}, delta_masked_ADE={case['delta_masked_ADE']:+.6f}, plot={case['plot_path']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap_type", type=str, default="interior", choices=["interior", "suffix"])
    parser.add_argument("--methods", type=str, default="linear_interp,ddpm_v3_anchored,ddpm_v4_conditional")
    parser.add_argument("--degradation", type=str, default=DEGRADATION_DEFAULT)
    parser.add_argument("--n_seeds", type=int, default=NUM_SEEDS_DEFAULT)
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.time()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if args.degradation != "missing_only":
        raise ValueError("This conditional experiment currently supports only degradation=missing_only.")

    print(f"Loading clean data (first {N_EXPERIMENT} trajectories) …")
    clean = load_npy(CTRL_DEGRAD / "clean.npy")[:N_EXPERIMENT]
    n, seq_len, _ = clean.shape
    obs_mask, span_start, span_end = build_obs_mask(n, seq_len, args.gap_type)

    if args.gap_type == "interior":
        degraded = load_npy(CTRL_DEGRAD / "degraded_missing_only_span20_fixed_seed42.npy")[:N_EXPERIMENT]
        linear_pred = load_npy(CTRL_RECON / "recon_missing_only_linear_interp_span20_fixed_seed42.npy")[:N_EXPERIMENT]
    else:
        degraded = build_degraded_missing_only(clean, obs_mask)
        linear_pred = run_linear_extrapolate_batch(degraded, obs_mask)

    map_meta = room3_map_meta()
    config = DDPMPriorInterfaceConfig(device="auto")
    full_rows: list[dict] = []
    var_rows: list[dict] = []
    selected_cases: list[dict] = []

    cache_det = {
        "linear_interp": linear_pred,
        "linear_extrapolate": linear_pred,
    }

    conditional_preds = None
    conditional_metrics = None
    coarse_for_plot = linear_pred

    for method in methods:
        print(f"Running {method} …")
        if method in {"linear_interp", "linear_extrapolate", "savgol_w5_p2", "kalman_cv_dt1.0_q1e-3_r1e-2"}:
            if method not in cache_det:
                if method == "savgol_w5_p2":
                    if args.gap_type == "suffix":
                        raise ValueError("savgol_w5_p2 is not supported for suffix gap because it requires a right boundary observation.")
                    cache_det[method] = load_npy(CTRL_RECON / "recon_missing_only_savgol_w5_p2_span20_fixed_seed42.npy")[:N_EXPERIMENT]
                elif method == "kalman_cv_dt1.0_q1e-3_r1e-2":
                    if args.gap_type == "suffix":
                        raise ValueError("kalman_cv_dt1.0_q1e-3_r1e-2 is not supported for suffix gap because it requires a terminal observation.")
                    cache_det[method] = load_npy(CTRL_RECON / "recon_missing_only_kalman_cv_dt1.0_q1e-3_r1e-2_span20_fixed_seed42.npy")[:N_EXPERIMENT]
            pred = cache_det[method]
            metrics = compute_metrics_per_traj(clean, pred, obs_mask, map_meta)
            metrics_ns = {k: v[:, None] for k, v in metrics.items()}
        elif method == "ddpm_v3_anchored":
            raw_preds = ddpm_prior_inpainting_v3(
                degraded,
                obs_mask,
                num_samples_per_traj=args.n_seeds,
                seed_base=SEED_BASE_DEFAULT,
                config=config,
            )
            anchored = anchor_missing_spans(raw_preds, degraded, obs_mask)
            metrics_ns = {
                k: np.stack(
                    [compute_metrics_per_traj(clean, anchored[:, s], obs_mask, map_meta)[k] for s in range(args.n_seeds)],
                    axis=1,
                )
                for k in METRIC_NAMES
            }
        elif method == "ddpm_v4_conditional":
            conditional_preds = ddpm_conditional_sample_v4(
                degraded,
                obs_mask,
                num_samples=args.n_seeds,
                seed_base=SEED_BASE_DEFAULT,
                config=config,
            )
            conditional_metrics = {
                k: np.stack(
                    [compute_metrics_per_traj(clean, conditional_preds[:, s], obs_mask, map_meta)[k] for s in range(args.n_seeds)],
                    axis=1,
                )
                for k in METRIC_NAMES
            }
            metrics_ns = conditional_metrics
        else:
            raise ValueError(f"Unsupported method: {method}")

        full_rows.extend(build_full_rows(method, args.degradation, args.gap_type, metrics_ns))
        var_rows.extend(build_var_rows(method, args.degradation, args.gap_type, metrics_ns))
        print(f"  masked_ADE={np.nanmean(metrics_ns['masked_ADE']):.6f}")

    if conditional_preds is not None and conditional_metrics is not None:
        selected_cases = select_and_plot_cases(
            clean=clean,
            degraded=degraded,
            coarse=coarse_for_plot,
            cond_preds=conditional_preds,
            cond_metrics=conditional_metrics,
            obs_mask=obs_mask,
            span_start=span_start,
            span_end=span_end,
            gap_type=args.gap_type,
        )
        with (OUTPUT_ROOT / "selected_cases.json").open("w", encoding="utf-8") as f:
            json.dump(selected_cases, f, indent=2)

    suffix = "interior" if args.gap_type == "interior" else "suffix"
    full_csv = OUTPUT_ROOT / f"full_results_{suffix}.csv"
    var_csv = OUTPUT_ROOT / f"variance_decomposition_{suffix}.csv"
    report_md = OUTPUT_ROOT / "REPORT.md"
    write_csv(full_csv, FULL_FIELDS, full_rows)
    write_csv(var_csv, VAR_FIELDS, var_rows)

    elapsed = time.time() - t0
    report_txt = generate_report(args, full_rows, var_rows, selected_cases, elapsed, span_start, span_end)
    report_md.write_text(report_txt, encoding="utf-8")

    print(f"full_results   -> {full_csv}")
    print(f"variance_csv   -> {var_csv}")
    print(f"report         -> {report_md}")
    print(f"trajectory_dir -> {PLOT_DIR}")
    print(f"elapsed_sec    -> {elapsed:.1f}")


if __name__ == "__main__":
    main()
