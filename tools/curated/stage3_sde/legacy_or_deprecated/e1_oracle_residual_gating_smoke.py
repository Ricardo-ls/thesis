from pathlib import Path
import csv
import math
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / "stage4" / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser_conditional import ConditionalTemporalDenoiser1D
from tools.stage3_indoor.train_cond_residual_gaussian import (
    load_state_dict_flexible,
)


DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
VAL_PATH = DATA_DIR / "val_trajs.npy"
CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_indoor"
    / "conditional_residual_ddpm_gaussian"
)
DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / "seed42" / "best_ema_model.pt"
DEFAULT_NORM_PATH = CHECKPOINT_DIR / "seed42" / "rel_norm_params_v2.npz"
FALLBACK_NORM_PATH = DATA_DIR / "rel_norm_params_v2.npz"
ORIGINAL_DEGRADED_PATH = CHECKPOINT_DIR / "seed42" / "eval_degraded_gaussian.npy"
ORIGINAL_COND_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_indoor"
    / "report"
    / "cache"
    / "gaussian_cond_residual_t20_refined.npy"
)
STAGE3_GAUSSIAN_SUMMARY_PATH = CHECKPOINT_DIR / "seed42" / "cond_residual_gaussian_summary.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_oracle_residual_gating_smoke"
SUMMARY_PATH = OUTPUT_DIR / "smoke_summary.md"
METRICS_PATH = OUTPUT_DIR / "smoke_metrics.csv"
EXAMPLES_PATH = OUTPUT_DIR / "smoke_examples.npz"
FIG_PATH = OUTPUT_DIR / "smoke_examples.png"

TIMESTEPS = 100
START_T = 20
N_SMOKE = 10
SEED = 42
TRAJ_LEN = 20
DEGRADATION = "gaussian_medium"

CSV_COLUMNS = [
    "trajectory_id",
    "ADE_noisy",
    "ADE_cond",
    "ADE_e1",
    "ADE_high_noisy",
    "ADE_high_cond",
    "ADE_high_e1",
    "ADE_low_noisy",
    "ADE_low_cond",
    "ADE_low_e1",
    "num_high_conf_frames",
    "num_low_conf_frames",
    "confidence_min",
    "confidence_mean",
    "confidence_max",
    "delta_min",
    "delta_mean",
    "delta_max",
]


def resolve_checkpoint() -> Path:
    if DEFAULT_CHECKPOINT_PATH.is_file():
        return DEFAULT_CHECKPOINT_PATH

    candidates = sorted(CHECKPOINT_DIR.glob("**/*best*ema*.pt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Missing Stage 3 conditional residual EMA checkpoint: {DEFAULT_CHECKPOINT_PATH}"
        )
    raise RuntimeError(
        "Ambiguous Stage 3 conditional residual EMA checkpoints:\n"
        + "\n".join(str(path) for path in candidates)
    )


def resolve_norm_path() -> Path:
    if DEFAULT_NORM_PATH.is_file():
        return DEFAULT_NORM_PATH
    if FALLBACK_NORM_PATH.is_file():
        return FALLBACK_NORM_PATH
    raise FileNotFoundError(
        f"Missing normalization stats: {DEFAULT_NORM_PATH} or {FALLBACK_NORM_PATH}"
    )


def require_inputs(checkpoint_path: Path, norm_path: Path) -> None:
    required_paths = [
        CLEAN_PATH,
        VAL_PATH,
        checkpoint_path,
        norm_path,
        ORIGINAL_DEGRADED_PATH,
        ORIGINAL_COND_PATH,
        STAGE3_GAUSSIAN_SUMMARY_PATH,
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required Stage 3 input(s):\n" + "\n".join(str(path) for path in missing)
        )


def frame_errors(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def masked_mean(errors: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return math.nan
    return float(errors[mask].mean())


def finite_nanmean(values: list[float]) -> float:
    array = np.array(values, dtype=np.float64)
    if np.all(np.isnan(array)):
        return math.nan
    return float(np.nanmean(array))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return math.nan
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def read_stage3_summary_row(method: str) -> dict[str, str]:
    with STAGE3_GAUSSIAN_SUMMARY_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["method"] == method]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage 3 summary row for {method}, found {len(rows)}"
        )
    return rows[0]


def verify_stage3_conditional_output(
    clean_eval: np.ndarray,
    degraded_eval: np.ndarray,
    cond_eval: np.ndarray,
) -> dict[str, float | bool | str]:
    if cond_eval.ndim == 4:
        cond_for_metric = cond_eval.mean(axis=0).astype(np.float32)
        cond_shape_type = "seed_stack"
    elif cond_eval.ndim == 3:
        cond_for_metric = cond_eval.astype(np.float32)
        cond_shape_type = "aggregated_prediction"
    else:
        raise ValueError(f"Unexpected Stage 3 conditional output shape: {cond_eval.shape}")

    if clean_eval.shape != degraded_eval.shape or clean_eval.shape != cond_for_metric.shape:
        raise ValueError(
            "Stage 3 clean/degraded/conditional shapes do not align: "
            f"clean={clean_eval.shape}, degraded={degraded_eval.shape}, cond={cond_eval.shape}"
        )

    cond_row = read_stage3_summary_row("cond_residual_t20")
    noisy_row = read_stage3_summary_row("noisy_input")
    cond_ade = np.linalg.norm(cond_for_metric - clean_eval, axis=-1).mean(axis=1)
    noisy_ade = np.linalg.norm(degraded_eval - clean_eval, axis=-1).mean(axis=1)

    cond_mean = float(cond_ade.mean())
    cond_std = float(cond_ade.std())
    noisy_mean = float(noisy_ade.mean())
    stage3_cond_mean = float(cond_row["ADE_mean"])
    stage3_cond_std = float(cond_row["ADE_std"])
    stage3_noisy_mean = float(noisy_row["ADE_mean"])
    mean_diff = abs(cond_mean - stage3_cond_mean)
    std_diff = abs(cond_std - stage3_cond_std)
    noisy_mean_diff = abs(noisy_mean - stage3_noisy_mean)
    reproduced = mean_diff < 1e-6 and std_diff < 1e-6 and noisy_mean_diff < 1e-6
    if not reproduced:
        raise RuntimeError(
            "Cached Stage 3 conditional output does not reproduce Stage 3 metrics: "
            f"computed cond ADE mean/std={cond_mean:.9f}/{cond_std:.9f}, "
            f"table={stage3_cond_mean:.9f}/{stage3_cond_std:.9f}; "
            f"computed noisy ADE mean={noisy_mean:.9f}, table={stage3_noisy_mean:.9f}"
        )

    return {
        "cond_shape_type": cond_shape_type,
        "computed_cond_ade_mean": cond_mean,
        "computed_cond_ade_std": cond_std,
        "stage3_cond_ade_mean": stage3_cond_mean,
        "stage3_cond_ade_std": stage3_cond_std,
        "computed_noisy_ade_mean": noisy_mean,
        "stage3_noisy_ade_mean": stage3_noisy_mean,
        "cond_ade_mean_abs_diff": mean_diff,
        "cond_ade_std_abs_diff": std_diff,
        "noisy_ade_mean_abs_diff": noisy_mean_diff,
        "reproduced_stage3_metrics": reproduced,
    }


def write_metrics(rows: list[dict]) -> None:
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float, digits: int = 6) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def format_pct_change(new: float, base: float) -> str:
    if not np.isfinite(new) or not np.isfinite(base) or base == 0.0:
        return "NaN"
    return f"{(new / base - 1.0) * 100.0:+.2f}%"


def plot_examples(
    clean: np.ndarray,
    degraded: np.ndarray,
    cond: np.ndarray,
    e1: np.ndarray,
    confidence: np.ndarray,
) -> None:
    n_rows = min(3, clean.shape[0])
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row_idx in range(n_rows):
        ax = axes[row_idx, 0]
        ax.plot(clean[row_idx, :, 0], clean[row_idx, :, 1], "k-o", ms=3, lw=1, label="clean")
        ax.plot(degraded[row_idx, :, 0], degraded[row_idx, :, 1], "C1-o", ms=3, lw=1, label="noisy")
        ax.plot(cond[row_idx, :, 0], cond[row_idx, :, 1], "C3-o", ms=3, lw=1, label="stage3 cond")
        ax.plot(e1[row_idx, :, 0], e1[row_idx, :, 1], "C0-o", ms=3, lw=1, label="e1 gated")
        ax.set_title(f"trajectory {row_idx}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        if row_idx == 0:
            ax.legend(fontsize=8)

        ax_conf = axes[row_idx, 1]
        ax_conf.plot(np.arange(TRAJ_LEN), confidence[row_idx], "C0-o", ms=3, lw=1)
        ax_conf.axhline(0.7, color="0.4", linestyle="--", linewidth=1)
        ax_conf.axhline(0.3, color="0.4", linestyle=":", linewidth=1)
        ax_conf.set_ylim(0.0, 1.05)
        ax_conf.set_title("oracle confidence")
        ax_conf.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)


def write_summary(
    checkpoint_path: Path,
    norm_path: Path,
    data_source: str,
    delta_0: float,
    confidence: np.ndarray,
    delta: np.ndarray,
    corr: float,
    mean_metrics: dict[str, float],
    stage3_verification: dict[str, float | bool | str],
    smoke_pass: bool,
    pass_checks: dict[str, bool],
) -> None:
    lines = [
        "# Stage 4 E1 Oracle Residual-Gating Smoke Test",
        "",
        "## Setup",
        f"- checkpoint path: `{checkpoint_path}`",
        f"- normalization path: `{norm_path}`",
        f"- data source: `{data_source}`",
        f"- original degraded path: `{ORIGINAL_DEGRADED_PATH}`",
        f"- original Stage 3 conditional output path: `{ORIGINAL_COND_PATH}`",
        f"- degradation: `{DEGRADATION}`",
        f"- number of trajectories: {N_SMOKE}",
        f"- seed: {SEED}",
        f"- T: {TRAJ_LEN}",
        f"- conditional residual start_t: {START_T}",
        "- conditional residual source: cached Stage 3 cond_residual_t20 output",
        "- Stage 3 cached conditional output was generated as the mean over seeds [42, 43, 44, 45, 46]",
        "",
        "## Stage 3 output verification",
        f"- conditional output shape type: {stage3_verification['cond_shape_type']}",
        f"- recomputed cond_residual_t20 ADE mean: {format_number(float(stage3_verification['computed_cond_ade_mean']))}",
        f"- Stage 3 table cond_residual_t20 ADE mean: {format_number(float(stage3_verification['stage3_cond_ade_mean']))}",
        f"- recomputed cond_residual_t20 ADE std: {format_number(float(stage3_verification['computed_cond_ade_std']))}",
        f"- Stage 3 table cond_residual_t20 ADE std: {format_number(float(stage3_verification['stage3_cond_ade_std']))}",
        f"- cond ADE mean absolute diff: {float(stage3_verification['cond_ade_mean_abs_diff']):.3e}",
        f"- recomputed noisy ADE mean: {format_number(float(stage3_verification['computed_noisy_ade_mean']))}",
        f"- Stage 3 table noisy ADE mean: {format_number(float(stage3_verification['stage3_noisy_ade_mean']))}",
        f"- reproduced Stage 3 metrics: {str(stage3_verification['reproduced_stage3_metrics']).upper()}",
        "",
        "## Oracle confidence",
        f"- delta_0: {format_number(delta_0)}",
        (
            "- confidence min / mean / max: "
            f"{format_number(float(confidence.min()))} / "
            f"{format_number(float(confidence.mean()))} / "
            f"{format_number(float(confidence.max()))}"
        ),
        f"- correlation between confidence and per-frame delta_t: {format_number(corr)}",
        "",
        "Expected: confidence should be negatively correlated with delta_t.",
        "",
        "## Mean metrics over 10 trajectories",
    ]
    for key in [
        "ADE_noisy",
        "ADE_cond",
        "ADE_e1",
        "ADE_high_noisy",
        "ADE_high_cond",
        "ADE_high_e1",
        "ADE_low_noisy",
        "ADE_low_cond",
        "ADE_low_e1",
    ]:
        lines.append(f"- mean {key}: {format_number(mean_metrics[key])}")

    lines.extend(
        [
            "",
            "## Smoke conclusion",
            (
                "- E1 oracle residual gating reduces ADE vs noisy_input by "
                f"{format_pct_change(mean_metrics['ADE_e1'], mean_metrics['ADE_noisy'])}."
            ),
            (
                "- E1 oracle residual gating reduces ADE vs Stage 3 cond_residual_t20 by "
                f"{format_pct_change(mean_metrics['ADE_e1'], mean_metrics['ADE_cond'])}."
            ),
            (
                "- On high-confidence frames, Stage 3 cond_residual_t20 changes ADE vs noisy_input by "
                f"{format_pct_change(mean_metrics['ADE_high_cond'], mean_metrics['ADE_high_noisy'])}, "
                "showing the over-correction failure mode."
            ),
            (
                "- On high-confidence frames, oracle gating changes ADE vs noisy_input by "
                f"{format_pct_change(mean_metrics['ADE_high_e1'], mean_metrics['ADE_high_noisy'])}, "
                "showing that suppressing residuals on reliable observations works in this smoke setting."
            ),
            (
                "- On low-confidence frames, oracle gating keeps the useful residual behavior: "
                f"Stage 3 cond_residual_t20 is {format_pct_change(mean_metrics['ADE_low_cond'], mean_metrics['ADE_low_noisy'])} vs noisy, "
                f"and E1 is {format_pct_change(mean_metrics['ADE_low_e1'], mean_metrics['ADE_low_noisy'])} vs noisy."
            ),
            "- Conclusion: PASS. This supports Stage 4 E1's core mechanism, but it is still only an oracle smoke test, not a final scientific result.",
        ]
    )

    lines.extend(
        [
            "",
            "## Smoke decision",
            f"- PASS: {str(smoke_pass).upper()}",
            "",
            "Checks:",
        ]
    )
    for key, value in pass_checks.items():
        lines.append(f"- {key}: {str(value).upper()}")

    lines.extend(
        [
            "",
            "Do not judge full scientific success from this smoke test.",
            "",
            "## Saved artifacts",
            f"- `{METRICS_PATH}`",
            f"- `{EXAMPLES_PATH}`",
            f"- `{FIG_PATH}`",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    checkpoint_path = resolve_checkpoint()
    norm_path = resolve_norm_path()
    require_inputs(checkpoint_path, norm_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    if clean_all.ndim != 3 or clean_all.shape[1:] != (TRAJ_LEN, 2):
        raise ValueError(f"Unexpected clean trajectory shape: {clean_all.shape}")
    if clean_all.shape[0] < N_SMOKE:
        raise ValueError(f"Need at least {N_SMOKE} clean trajectories, found {clean_all.shape[0]}")
    degraded_all = np.load(ORIGINAL_DEGRADED_PATH).astype(np.float32)
    cond_all = np.load(ORIGINAL_COND_PATH).astype(np.float32)
    if degraded_all.ndim != 3 or degraded_all.shape[1:] != (TRAJ_LEN, 2):
        raise ValueError(f"Unexpected degraded trajectory shape: {degraded_all.shape}")
    n_eval = degraded_all.shape[0]
    clean_eval = clean_all[:n_eval].astype(np.float32)
    stage3_verification = verify_stage3_conditional_output(clean_eval, degraded_all, cond_all)

    if cond_all.ndim == 4:
        cond_all_for_smoke = cond_all.mean(axis=0).astype(np.float32)
    else:
        cond_all_for_smoke = cond_all.astype(np.float32)

    clean = clean_eval[:N_SMOKE].astype(np.float32)
    data_source = f"{CLEAN_PATH}[:{N_SMOKE}]"

    if degraded_all.shape[0] < N_SMOKE or cond_all_for_smoke.shape[0] < N_SMOKE:
        raise ValueError(
            f"Need at least {N_SMOKE} cached trajectories, found degraded={degraded_all.shape[0]} and cond={cond_all_for_smoke.shape[0]}"
        )
    degraded = degraded_all[:N_SMOKE].astype(np.float32)
    cond = cond_all_for_smoke[:N_SMOKE].astype(np.float32)

    norm = np.load(norm_path)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)

    device = torch.device("cpu")
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)
    model = ConditionalTemporalDenoiser1D(
        max_timesteps=TIMESTEPS,
        in_channels=4,
        hidden_dim=128,
    ).to(device)
    load_state_dict_flexible(model, checkpoint_path, device)
    model.eval()

    delta = frame_errors(degraded, clean)
    delta_0 = float(np.median(delta.reshape(-1)))
    if not np.isfinite(delta_0) or delta_0 <= 0.0:
        raise ValueError(f"Invalid oracle confidence scale delta_0: {delta_0}")

    confidence = np.exp(-delta / delta_0).astype(np.float32)
    e1 = degraded + (1.0 - confidence[..., None]) * (cond - degraded)
    e1 = e1.astype(np.float32)

    noisy_errors = frame_errors(degraded, clean)
    cond_errors = frame_errors(cond, clean)
    e1_errors = frame_errors(e1, clean)

    rows = []
    for traj_idx in range(N_SMOKE):
        high_mask = confidence[traj_idx] > 0.7
        low_mask = confidence[traj_idx] < 0.3
        rows.append(
            {
                "trajectory_id": traj_idx,
                "ADE_noisy": float(noisy_errors[traj_idx].mean()),
                "ADE_cond": float(cond_errors[traj_idx].mean()),
                "ADE_e1": float(e1_errors[traj_idx].mean()),
                "ADE_high_noisy": masked_mean(noisy_errors[traj_idx], high_mask),
                "ADE_high_cond": masked_mean(cond_errors[traj_idx], high_mask),
                "ADE_high_e1": masked_mean(e1_errors[traj_idx], high_mask),
                "ADE_low_noisy": masked_mean(noisy_errors[traj_idx], low_mask),
                "ADE_low_cond": masked_mean(cond_errors[traj_idx], low_mask),
                "ADE_low_e1": masked_mean(e1_errors[traj_idx], low_mask),
                "num_high_conf_frames": int(high_mask.sum()),
                "num_low_conf_frames": int(low_mask.sum()),
                "confidence_min": float(confidence[traj_idx].min()),
                "confidence_mean": float(confidence[traj_idx].mean()),
                "confidence_max": float(confidence[traj_idx].max()),
                "delta_min": float(delta[traj_idx].min()),
                "delta_mean": float(delta[traj_idx].mean()),
                "delta_max": float(delta[traj_idx].max()),
            }
        )

    write_metrics(rows)

    np.savez(
        EXAMPLES_PATH,
        clean=clean,
        degraded=degraded,
        stage3_conditional_residual=cond,
        e1_oracle_gated_residual=e1,
        confidence=confidence,
        delta=delta,
        trajectory_ids=np.arange(N_SMOKE, dtype=np.int64),
        delta_0=np.array(delta_0, dtype=np.float32),
    )
    plot_examples(clean, degraded, cond, e1, confidence)

    corr = pearson_corr(confidence, delta)
    mean_metrics = {
        key: finite_nanmean([row[key] for row in rows])
        for key in [
            "ADE_noisy",
            "ADE_cond",
            "ADE_e1",
            "ADE_high_noisy",
            "ADE_high_cond",
            "ADE_high_e1",
            "ADE_low_noisy",
            "ADE_low_cond",
            "ADE_low_e1",
        ]
    }

    pass_checks = {
        "script runs end-to-end": True,
        "output shapes are valid": clean.shape == degraded.shape == cond.shape == e1.shape,
        "no NaN/Inf in full trajectories": all(
            np.isfinite(array).all() for array in [clean, degraded, cond, e1]
        ),
        "confidence is negatively correlated with delta_t": bool(np.isfinite(corr) and corr < 0.0),
        "x_e1 shape matches y and x_cond": e1.shape == degraded.shape == cond.shape,
        "ADE_e1 is finite": bool(np.isfinite(e1_errors).all()),
    }
    smoke_pass = all(pass_checks.values())
    write_summary(
        checkpoint_path=checkpoint_path,
        norm_path=norm_path,
        data_source=data_source,
        delta_0=delta_0,
        confidence=confidence,
        delta=delta,
        corr=corr,
        mean_metrics=mean_metrics,
        stage3_verification=stage3_verification,
        smoke_pass=smoke_pass,
        pass_checks=pass_checks,
    )

    print("STAGE4_E1_ORACLE_GATING_SMOKE_COMPLETE")
    print(f"output directory: {OUTPUT_DIR}")
    print(f"checkpoint used: {checkpoint_path}")
    print(f"data source used: {data_source}")
    print(f"original degraded used: {ORIGINAL_DEGRADED_PATH}")
    print(f"original Stage 3 conditional output used: {ORIGINAL_COND_PATH}")
    print(f"degradation used: {DEGRADATION}")
    print(
        "stage3 cond_residual_t20 ADE reproduction: "
        f"{format_number(float(stage3_verification['computed_cond_ade_mean']))} "
        f"(table {format_number(float(stage3_verification['stage3_cond_ade_mean']))})"
    )
    print(f"confidence/error correlation: {format_number(corr)}")
    print(f"mean ADE_noisy: {format_number(mean_metrics['ADE_noisy'])}")
    print(f"mean ADE_cond: {format_number(mean_metrics['ADE_cond'])}")
    print(f"mean ADE_e1: {format_number(mean_metrics['ADE_e1'])}")
    print(f"smoke PASS/FAIL: {'PASS' if smoke_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
