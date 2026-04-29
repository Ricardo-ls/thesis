from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_mplconfig")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3.refinement.ddpm_refiner import (
    DDPMPriorInterfaceConfig,
    abs_to_rel,
    ddpm_prior_inpainting_v3,
    rel_to_abs,
)
from tools.stage3.controlled.evaluate_coarse_reconstruction import room3_map_meta


N_TRAJ = 1024
N_SEEDS = 5
SEED_BASE = 42
TAG = "span20_fixed_seed42"
DEGRADATION = "missing_only"

DATA_ETH_PATH = PROJECT_ROOT / "datasets" / "processed" / "data_eth_ucy_20.npy"
ROOM3_CLEAN_PATH = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "data" / "clean_windows_room3.npz"
CONTROLLED_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark"
DEGRADED_PATH = CONTROLLED_ROOT / "degradation" / f"degraded_{DEGRADATION}_{TAG}.npy"
CLEAN_PATH = CONTROLLED_ROOT / "degradation" / "clean.npy"
MASK_PATH = CONTROLLED_ROOT / "degradation" / f"mask_{TAG}.npy"
LINEAR_PATH = CONTROLLED_ROOT / "reconstruction" / f"recon_{DEGRADATION}_linear_interp_{TAG}.npy"

OUT_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "diagnosis"
SCALE_CSV = OUT_ROOT / "scale_comparison.csv"
TABLE_CSV = OUT_ROOT / "v3_diagnosis_table.csv"
REPORT_MD = OUT_ROOT / "DIAGNOSIS.md"


@dataclass
class MethodSummary:
    method: str
    n_trajectories: int
    n_seeds: int
    masked_ADE_mean: float
    masked_ADE_std: float
    masked_RMSE_mean: float
    masked_RMSE_std: float
    off_map_ratio_mean: float


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(path, allow_pickle=False)


def load_room3_clean() -> np.ndarray:
    if not ROOM3_CLEAN_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {ROOM3_CLEAN_PATH}")
    payload = np.load(ROOM3_CLEAN_PATH, allow_pickle=False)
    return payload["traj_abs"].astype(np.float32)


def step_size_stats(abs_traj: np.ndarray) -> dict[str, float]:
    rel = np.diff(abs_traj.astype(np.float32), axis=1)
    step = np.linalg.norm(rel, axis=-1).reshape(-1)
    return {
        "mean_step": float(np.mean(step)),
        "std_step": float(np.std(step)),
        "p50_step": float(np.percentile(step, 50)),
        "p95_step": float(np.percentile(step, 95)),
    }


def write_scale_csv(eth_stats: dict[str, float], room3_stats: dict[str, float], ratio: float) -> None:
    fields = ["dataset", "mean_step", "std_step", "p50_step", "p95_step", "mean_step_ratio_eth_over_room3"]
    rows = [
        {
            "dataset": "eth_ucy_train",
            **eth_stats,
            "mean_step_ratio_eth_over_room3": ratio,
        },
        {
            "dataset": "room3_clean",
            **room3_stats,
            "mean_step_ratio_eth_over_room3": ratio,
        },
        {
            "dataset": "ratio_summary",
            "mean_step": eth_stats["mean_step"] / max(room3_stats["mean_step"], 1e-12),
            "std_step": float("nan"),
            "p50_step": float("nan"),
            "p95_step": float("nan"),
            "mean_step_ratio_eth_over_room3": ratio,
        },
    ]
    with SCALE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics_per_traj(clean: np.ndarray, pred: np.ndarray, obs_mask: np.ndarray, map_meta: dict) -> dict[str, np.ndarray]:
    diff = pred - clean
    point_error = np.linalg.norm(diff, axis=-1)
    miss_mask = obs_mask == 0

    masked_ade = np.full(clean.shape[0], np.nan, dtype=np.float32)
    masked_rmse = np.full(clean.shape[0], np.nan, dtype=np.float32)
    valid_rows = np.where(miss_mask.any(axis=1))[0]
    for i in valid_rows:
        m = miss_mask[i]
        masked_ade[i] = float(point_error[i, m].mean())
        masked_rmse[i] = float(np.sqrt(np.mean(diff[i, m] ** 2)))

    xlo, xhi = map_meta["x_min"], map_meta["x_max"]
    ylo, yhi = map_meta["y_min"], map_meta["y_max"]
    outside = (
        (pred[:, :, 0] < xlo)
        | (pred[:, :, 0] > xhi)
        | (pred[:, :, 1] < ylo)
        | (pred[:, :, 1] > yhi)
    )
    off_map_ratio = outside.mean(axis=1).astype(np.float32)

    return {
        "masked_ADE": masked_ade,
        "masked_RMSE": masked_rmse,
        "off_map_ratio": off_map_ratio,
    }


def summarize_method(method: str, metric_ns: dict[str, np.ndarray]) -> MethodSummary:
    masked_ade = metric_ns["masked_ADE"].reshape(-1)
    masked_rmse = metric_ns["masked_RMSE"].reshape(-1)
    off_map_ratio = metric_ns["off_map_ratio"].reshape(-1)

    return MethodSummary(
        method=method,
        n_trajectories=int(metric_ns["masked_ADE"].shape[0]),
        n_seeds=int(metric_ns["masked_ADE"].shape[1]) if metric_ns["masked_ADE"].ndim == 2 else 1,
        masked_ADE_mean=float(np.nanmean(masked_ade)),
        masked_ADE_std=float(np.nanstd(masked_ade)),
        masked_RMSE_mean=float(np.nanmean(masked_rmse)),
        masked_RMSE_std=float(np.nanstd(masked_rmse)),
        off_map_ratio_mean=float(np.nanmean(off_map_ratio)),
    )


def write_table_csv(rows: list[MethodSummary]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(MethodSummary("", 0, 0, math.nan, math.nan, math.nan, math.nan, math.nan)).keys())
    with TABLE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def stack_seed_metrics(clean: np.ndarray, preds_ns: np.ndarray, obs_mask: np.ndarray, map_meta: dict) -> dict[str, np.ndarray]:
    metric_lists: dict[str, list[np.ndarray]] = {
        "masked_ADE": [],
        "masked_RMSE": [],
        "off_map_ratio": [],
    }
    for seed_idx in range(preds_ns.shape[1]):
        metrics = compute_metrics_per_traj(clean, preds_ns[:, seed_idx], obs_mask, map_meta)
        for key, value in metrics.items():
            metric_lists[key].append(value)
    return {key: np.stack(values, axis=1) for key, values in metric_lists.items()}


def apply_scale_calibration(pred_ns: np.ndarray, scale_ratio: float, observed_abs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    n, s, t, _ = pred_ns.shape
    flat = pred_ns.reshape(n * s, t, 2)
    rel = abs_to_rel(flat)
    scaled_rel = rel / float(scale_ratio)
    start_points = observed_abs[:, 0, :].astype(np.float32)
    start_points = np.repeat(start_points, s, axis=0)
    scaled_abs = rel_to_abs(scaled_rel, start_points).reshape(n, s, t, 2)
    return anchor_missing_spans(scaled_abs, observed_abs, obs_mask)


def apply_endpoint_anchoring(pred_ns: np.ndarray, observed_abs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    return anchor_missing_spans(pred_ns, observed_abs, obs_mask)


def anchor_missing_spans(pred_ns: np.ndarray, observed_abs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    pred_ns = pred_ns.astype(np.float32, copy=True)
    n, s, t, _ = pred_ns.shape
    observed = observed_abs.astype(np.float32)

    for i in range(n):
        miss = np.where(obs_mask[i] == 0)[0]
        if len(miss) == 0:
            pred_ns[i] = observed[i][None, :, :]
            continue
        left = int(miss[0] - 1)
        right = int(miss[-1] + 1)
        if left < 0 or right >= t:
            raise ValueError("Expected a bounded contiguous missing span.")

        left_obs = observed[i, left]
        right_obs = observed[i, right]
        observed_mask = obs_mask[i] == 1

        for seed_idx in range(s):
            traj = pred_ns[i, seed_idx]
            segment = traj[left : right + 1].copy()
            seg_len = right - left
            left_delta = left_obs - segment[0]
            right_delta = right_obs - segment[-1]
            for local_idx in range(seg_len + 1):
                lam = local_idx / seg_len if seg_len > 0 else 0.0
                segment[local_idx] = segment[local_idx] + (1.0 - lam) * left_delta + lam * right_delta
            segment[0] = left_obs
            segment[-1] = right_obs
            traj[left : right + 1] = segment
            traj[observed_mask] = observed[i, observed_mask]
            pred_ns[i, seed_idx] = traj
    return pred_ns


def diagnose_verdict(v3_mean: float, scaled_mean: float, anchored_mean: float) -> tuple[str, list[str]]:
    lines: list[str] = []
    scaled_major = scaled_mean < 0.5 * v3_mean
    anchored_major = anchored_mean < 0.5 * v3_mean

    if scaled_major and not anchored_major:
        verdict = "scale mismatch 占主导"
        lines += [
            "- `ddpm_v3_inpainting_scaled` 将 masked_ADE 降到原始 v3 的 50% 以下。",
            "- `ddpm_v3_inpainting_anchored` 没有达到同等级改善。",
            "- 下一步假设：先验主要失败在位移尺度不匹配，条件化结构问题是次要项。",
        ]
    elif anchored_major and not scaled_major:
        verdict = "endpoint conditioning 失效是主导"
        lines += [
            "- `ddpm_v3_inpainting_anchored` 将 masked_ADE 降到原始 v3 的 50% 以下。",
            "- `ddpm_v3_inpainting_scaled` 没有达到同等级改善。",
            "- 下一步假设：真正缺的是 bridge / endpoint-conditioned generation，而不是单纯尺度变换。",
        ]
    elif scaled_major and anchored_major:
        if scaled_mean <= anchored_mean:
            verdict = "scale mismatch 占主导，但 endpoint conditioning 也明显失效"
        else:
            verdict = "endpoint conditioning 失效是主导，但 scale mismatch 也明显存在"
        lines += [
            "- `ddpm_v3_inpainting_scaled` 与 `ddpm_v3_inpainting_anchored` 都把 masked_ADE 降到原始 v3 的 50% 以下。",
            "- 这说明两个因素都在起作用，需要同时控制尺度与端点条件。",
        ]
    else:
        verdict = "unconditional prior 在 conditional 任务上根本不可用"
        lines += [
            "- `ddpm_v3_inpainting_scaled` 和 `ddpm_v3_inpainting_anchored` 都没有把 masked_ADE 降到原始 v3 的 50% 以下。",
            "- 下一步假设：问题不只是尺度或末端对齐，而是无条件先验本身不适合该 conditional missing-span reconstruction 任务。",
        ]

    return verdict, lines


def markdown_table(rows: list[MethodSummary]) -> str:
    header = (
        "| method | n_trajectories | n_seeds | masked_ADE_mean | masked_ADE_std | "
        "masked_RMSE_mean | masked_RMSE_std | off_map_ratio_mean |"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for row in rows:
        body.append(
            f"| {row.method} | {row.n_trajectories} | {row.n_seeds} | "
            f"{row.masked_ADE_mean:.6f} | {row.masked_ADE_std:.6f} | "
            f"{row.masked_RMSE_mean:.6f} | {row.masked_RMSE_std:.6f} | "
            f"{row.off_map_ratio_mean:.6f} |"
        )
    return "\n".join([header, sep, *body])


def write_report(
    eth_stats: dict[str, float],
    room3_stats: dict[str, float],
    scale_ratio: float,
    rows: list[MethodSummary],
    verdict: str,
    verdict_lines: list[str],
    elapsed_sec: float,
) -> None:
    by_method = {row.method: row for row in rows}
    lines = [
        "# v3 Failure Diagnosis",
        "",
        f"*Generated in {elapsed_sec:.1f} s on {N_TRAJ} trajectories × {N_SEEDS} seeds for `missing_only`.*",
        "",
        "## Step A: Scale Comparison",
        "",
        f"- ETH+UCY mean step: `{eth_stats['mean_step']:.6f}`",
        f"- room3 mean step: `{room3_stats['mean_step']:.6f}`",
        f"- ratio `r = mean_step_eth / mean_step_room3`: `{scale_ratio:.6f}`",
        "",
        "| dataset | mean_step | std_step | p50_step | p95_step |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| eth_ucy_train | {eth_stats['mean_step']:.6f} | {eth_stats['std_step']:.6f} | {eth_stats['p50_step']:.6f} | {eth_stats['p95_step']:.6f} |",
        f"| room3_clean | {room3_stats['mean_step']:.6f} | {room3_stats['std_step']:.6f} | {room3_stats['p50_step']:.6f} | {room3_stats['p95_step']:.6f} |",
        "",
        "## Step D: Missing-only Comparison",
        "",
        markdown_table(rows),
        "",
        "## Step E: Final Diagnosis",
        "",
        f"**判决：{verdict}**",
        "",
        f"- 原始 `ddpm_v3_inpainting` masked_ADE mean: `{by_method['ddpm_v3_inpainting'].masked_ADE_mean:.6f}`",
        f"- `ddpm_v3_inpainting_scaled` masked_ADE mean: `{by_method['ddpm_v3_inpainting_scaled'].masked_ADE_mean:.6f}`",
        f"- `ddpm_v3_inpainting_anchored` masked_ADE mean: `{by_method['ddpm_v3_inpainting_anchored'].masked_ADE_mean:.6f}`",
        "",
        *verdict_lines,
        "",
        "## Notes",
        "",
        "- `ddpm_v3_inpainting_scaled`: v3 sample → relative displacement → divide by `r` → absolute trajectory → missing-span endpoint anchoring.",
        "- `ddpm_v3_inpainting_anchored`: no scale calibration; only apply missing-span endpoint anchoring.",
        "- Evaluation metrics are computed only on the existing `missing_only` controlled benchmark subset; Stage 2 training is unchanged.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    t0 = time.time()

    eth_abs = load_npy(DATA_ETH_PATH).astype(np.float32)
    room3_abs = load_room3_clean()
    eth_stats = step_size_stats(eth_abs)
    room3_stats = step_size_stats(room3_abs)
    scale_ratio = eth_stats["mean_step"] / max(room3_stats["mean_step"], 1e-12)
    write_scale_csv(eth_stats, room3_stats, scale_ratio)

    clean = load_npy(CLEAN_PATH)[:N_TRAJ].astype(np.float32)
    degraded = load_npy(DEGRADED_PATH)[:N_TRAJ].astype(np.float32)
    obs_mask = load_npy(MASK_PATH)[:N_TRAJ].astype(np.uint8)
    linear_pred = load_npy(LINEAR_PATH)[:N_TRAJ].astype(np.float32)
    map_meta = room3_map_meta()

    linear_metrics = compute_metrics_per_traj(clean, linear_pred, obs_mask, map_meta)
    linear_metrics_ns = {key: value[:, None] for key, value in linear_metrics.items()}

    config = DDPMPriorInterfaceConfig(device="auto")
    v3_preds = ddpm_prior_inpainting_v3(
        degraded,
        obs_mask,
        num_samples_per_traj=N_SEEDS,
        seed_base=SEED_BASE,
        config=config,
    ).astype(np.float32)

    v3_scaled = apply_scale_calibration(v3_preds, scale_ratio, degraded, obs_mask)
    v3_anchored = apply_endpoint_anchoring(v3_preds, degraded, obs_mask)

    v3_metrics_ns = stack_seed_metrics(clean, v3_preds, obs_mask, map_meta)
    v3_scaled_metrics_ns = stack_seed_metrics(clean, v3_scaled, obs_mask, map_meta)
    v3_anchored_metrics_ns = stack_seed_metrics(clean, v3_anchored, obs_mask, map_meta)

    rows = [
        summarize_method("linear_interp", linear_metrics_ns),
        summarize_method("ddpm_v3_inpainting", v3_metrics_ns),
        summarize_method("ddpm_v3_inpainting_scaled", v3_scaled_metrics_ns),
        summarize_method("ddpm_v3_inpainting_anchored", v3_anchored_metrics_ns),
    ]
    write_table_csv(rows)

    verdict, verdict_lines = diagnose_verdict(
        v3_mean=rows[1].masked_ADE_mean,
        scaled_mean=rows[2].masked_ADE_mean,
        anchored_mean=rows[3].masked_ADE_mean,
    )
    write_report(
        eth_stats=eth_stats,
        room3_stats=room3_stats,
        scale_ratio=scale_ratio,
        rows=rows,
        verdict=verdict,
        verdict_lines=verdict_lines,
        elapsed_sec=time.time() - t0,
    )

    print("=" * 72)
    print(f"scale_comparison_csv = {SCALE_CSV}")
    print(f"diagnosis_table_csv  = {TABLE_CSV}")
    print(f"diagnosis_report_md  = {REPORT_MD}")
    print("=" * 72)
    for row in rows:
        print(asdict(row))


if __name__ == "__main__":
    main()
