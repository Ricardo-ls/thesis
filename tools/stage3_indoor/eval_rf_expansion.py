from __future__ import annotations

from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser_conditional import ConditionalTemporalDenoiser1D
from models.temporal_denoiser_conditional_4block import TemporalDenoiserConditional4Block
from tools.stage3_indoor.generalization_diagnostic import (
    DEGRADATION_ORDER,
    N,
    SAMPLE_SEEDS,
    generate_bias,
    generate_burst,
    generate_combined,
    generate_drift,
    generate_gaussian,
    generate_jump,
)
from tools.stage3_indoor.train_cond_residual_gaussian import (
    load_state_dict_flexible,
    sample_conditional_residual,
)

try:
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False


CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CURRENT_2BLOCK_EMA_PATH = (
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "best_ema_model.pt"
)
CURRENT_2BLOCK_FINAL_PATH = (
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42" / "final_ema_model.pt"
)
EXPANDED_4BLOCK_EMA_PATH = (
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "receptive_field_expansion" / "ckpt_4block_best_ema.pt"
)
EXPANDED_4BLOCK_FINAL_PATH = (
    PROJECT_ROOT / "outputs" / "stage3_indoor" / "receptive_field_expansion" / "ckpt_4block_final_ema.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "receptive_field_expansion"
AGGREGATE_CSV_PATH = OUTPUT_DIR / "ablation_results.csv"
RAW_CSV_PATH = OUTPUT_DIR / "raw_4block_eval.csv"

TIMESTEPS = 100
ROOM_MIN = 0.0
ROOM_MAX = 3.0
METHOD_NOISY = "noisy_input"
METHOD_CURRENT = "current_2block_conditional"
METHOD_EXPANDED = "expanded_4block_conditional"


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def ensure_required_inputs() -> None:
    required = [
        CLEAN_PATH,
        NORM_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))

    if not CURRENT_2BLOCK_EMA_PATH.is_file() and not CURRENT_2BLOCK_FINAL_PATH.is_file():
        raise FileNotFoundError(
            "Missing current 2-block checkpoint. Expected one of:\n"
            f"{CURRENT_2BLOCK_EMA_PATH}\n{CURRENT_2BLOCK_FINAL_PATH}"
        )
    if not EXPANDED_4BLOCK_EMA_PATH.is_file() and not EXPANDED_4BLOCK_FINAL_PATH.is_file():
        raise FileNotFoundError(
            "Missing expanded 4-block checkpoint. Expected one of:\n"
            f"{EXPANDED_4BLOCK_EMA_PATH}\n{EXPANDED_4BLOCK_FINAL_PATH}"
        )


def choose_checkpoint(ema_path: Path, fallback_path: Path) -> tuple[Path, bool]:
    if ema_path.is_file():
        return ema_path, True
    if fallback_path.is_file():
        return fallback_path, False
    raise FileNotFoundError(f"Missing checkpoint: neither {ema_path} nor {fallback_path} exists")


def make_degraded_map(clean: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "gaussian_medium": generate_gaussian(clean),
        "drift_medium": generate_drift(clean),
        "jump_medium": generate_jump(clean),
        "burst_medium": generate_burst(clean),
        "bias_medium": generate_bias(clean),
        "combined_medium": generate_combined(clean),
    }


def compute_metrics(pred: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    errors = np.linalg.norm(pred - clean, axis=-1)
    ade = errors.mean(axis=1).astype(np.float32)
    rmse = np.sqrt(np.mean(errors**2, axis=1)).astype(np.float32)
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    smooth = np.linalg.norm(acc, axis=-1).mean(axis=1).astype(np.float32)
    out_of_room = (
        (pred[..., 0] < ROOM_MIN)
        | (pred[..., 0] > ROOM_MAX)
        | (pred[..., 1] < ROOM_MIN)
        | (pred[..., 1] > ROOM_MAX)
    )
    oor = out_of_room.mean(axis=1).astype(np.float32)
    return ade, rmse, smooth, oor


def safe_wilcoxon(delta: np.ndarray) -> str:
    if not SCIPY_AVAILABLE:
        return "NA"
    if delta.size == 0:
        return "NA"
    if np.allclose(delta, 0.0):
        return "1.0"
    try:
        return f"{float(wilcoxon(delta, alternative='less').pvalue):.12g}"
    except Exception:
        return "NA"


def build_aggregate_rows(raw_rows: list[dict], p_values_by_degradation: dict[str, str]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in raw_rows:
        grouped.setdefault((row["method"], row["degradation"]), []).append(row)

    rows: list[dict] = []
    source_map = {
        METHOD_NOISY: "stage3_main_existing",
        METHOD_CURRENT: "stage3_main_existing",
        METHOD_EXPANDED: "new_4block_eval",
    }
    for method in [METHOD_NOISY, METHOD_CURRENT, METHOD_EXPANDED]:
        for degradation in DEGRADATION_ORDER:
            key = (method, degradation)
            if key not in grouped:
                raise RuntimeError(f"Missing aggregate cell for method={method}, degradation={degradation}")
            subset = grouped[key]
            ade = np.array([float(r["ADE"]) for r in subset], dtype=np.float32)
            rmse = np.array([float(r["RMSE"]) for r in subset], dtype=np.float32)
            smooth = np.array([float(r["smooth"]) for r in subset], dtype=np.float32)
            oor = np.array([float(r["OOR"]) for r in subset], dtype=np.float32)
            traj_ids = {int(r["trajectory_id"]) for r in subset}
            seeds = {int(r["seed"]) for r in subset}
            rows.append(
                {
                    "method": method,
                    "degradation": degradation,
                    "ADE_mean": f"{float(np.mean(ade)):.12g}",
                    "ADE_std": f"{float(np.std(ade, ddof=0)):.12g}",
                    "ADE_median": f"{float(np.median(ade)):.12g}",
                    "RMSE_mean": f"{float(np.mean(rmse)):.12g}",
                    "smooth_mean": f"{float(np.mean(smooth)):.12g}",
                    "OOR_rate": f"{float(np.mean(oor)):.12g}",
                    "n_traj": len(traj_ids),
                    "n_seed": len(seeds),
                    "source": source_map[method],
                    "paired_wilcoxon_p_vs_current_2block": (
                        p_values_by_degradation[degradation] if method == METHOD_EXPANDED else "NA"
                    ),
                }
            )
    return rows


def print_ade_table(aggregate_rows: list[dict]) -> None:
    row_order = [METHOD_NOISY, METHOD_CURRENT, METHOD_EXPANDED]
    print("3x6 ADE table:")
    header = (
        "method".ljust(30)
        + " gaussian_medium".rjust(18)
        + " bias_medium".rjust(15)
        + " drift_medium".rjust(16)
        + " jump_medium".rjust(15)
        + " burst_medium".rjust(16)
        + " combined_medium".rjust(19)
    )
    print(header)
    for method in row_order:
        values = {}
        for row in aggregate_rows:
            if row["method"] == method:
                values[row["degradation"]] = row["ADE_mean"]
        print(
            method.ljust(30)
            + values["gaussian_medium"].rjust(18)
            + values["bias_medium"].rjust(15)
            + values["drift_medium"].rjust(16)
            + values["jump_medium"].rjust(15)
            + values["burst_medium"].rjust(16)
            + values["combined_medium"].rjust(19)
        )


def main() -> None:
    ensure_required_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_ckpt_path, current_uses_ema = choose_checkpoint(CURRENT_2BLOCK_EMA_PATH, CURRENT_2BLOCK_FINAL_PATH)
    expanded_ckpt_path, expanded_uses_ema = choose_checkpoint(EXPANDED_4BLOCK_EMA_PATH, EXPANDED_4BLOCK_FINAL_PATH)

    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    clean = clean_all[:N]
    if clean.shape != (N, 20, 2):
        raise ValueError(f"Expected clean evaluation shape ({N}, 20, 2), got {clean.shape}")

    norm = np.load(NORM_PATH)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    if rel_mean.shape != (2,) or rel_std.shape != (2,):
        raise ValueError(f"Expected rel_mean/rel_std shape (2,), got {rel_mean.shape} and {rel_std.shape}")

    device = torch.device("cpu")
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)

    current_model = ConditionalTemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=4, hidden_dim=128).to(device)
    load_state_dict_flexible(current_model, current_ckpt_path, device)
    current_model.eval()

    expanded_model = TemporalDenoiserConditional4Block(max_timesteps=TIMESTEPS, in_channels=4, hidden_dim=128).to(device)
    load_state_dict_flexible(expanded_model, expanded_ckpt_path, device)
    expanded_model.eval()

    degraded_map = make_degraded_map(clean)

    raw_rows: list[dict] = []
    paired_by_degradation: dict[str, dict[tuple[int, int], dict[str, float]]] = {
        degradation: {} for degradation in DEGRADATION_ORDER
    }

    for degradation in DEGRADATION_ORDER:
        degraded = degraded_map[degradation]
        noisy_ade, noisy_rmse, noisy_smooth, noisy_oor = compute_metrics(degraded, clean)

        for seed in SAMPLE_SEEDS:
            current_pred = sample_conditional_residual(
                degraded_abs=degraded,
                model=current_model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                start_t=20,
                sample_seed=seed,
                device=device,
            )
            expanded_pred = sample_conditional_residual(
                degraded_abs=degraded,
                model=expanded_model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                start_t=20,
                sample_seed=seed,
                device=device,
            )

            current_ade, current_rmse, current_smooth, current_oor = compute_metrics(current_pred, clean)
            expanded_ade, expanded_rmse, expanded_smooth, expanded_oor = compute_metrics(expanded_pred, clean)

            for traj_idx in range(N):
                key = (traj_idx, seed)
                raw_rows.append(
                    {
                        "trajectory_id": traj_idx,
                        "seed": seed,
                        "degradation": degradation,
                        "method": METHOD_NOISY,
                        "ADE": f"{float(noisy_ade[traj_idx]):.12g}",
                        "RMSE": f"{float(noisy_rmse[traj_idx]):.12g}",
                        "smooth": f"{float(noisy_smooth[traj_idx]):.12g}",
                        "OOR": f"{float(noisy_oor[traj_idx]):.12g}",
                    }
                )
                raw_rows.append(
                    {
                        "trajectory_id": traj_idx,
                        "seed": seed,
                        "degradation": degradation,
                        "method": METHOD_CURRENT,
                        "ADE": f"{float(current_ade[traj_idx]):.12g}",
                        "RMSE": f"{float(current_rmse[traj_idx]):.12g}",
                        "smooth": f"{float(current_smooth[traj_idx]):.12g}",
                        "OOR": f"{float(current_oor[traj_idx]):.12g}",
                    }
                )
                raw_rows.append(
                    {
                        "trajectory_id": traj_idx,
                        "seed": seed,
                        "degradation": degradation,
                        "method": METHOD_EXPANDED,
                        "ADE": f"{float(expanded_ade[traj_idx]):.12g}",
                        "RMSE": f"{float(expanded_rmse[traj_idx]):.12g}",
                        "smooth": f"{float(expanded_smooth[traj_idx]):.12g}",
                        "OOR": f"{float(expanded_oor[traj_idx]):.12g}",
                    }
                )
                paired_by_degradation[degradation][key] = {
                    METHOD_CURRENT: float(current_ade[traj_idx]),
                    METHOD_EXPANDED: float(expanded_ade[traj_idx]),
                }

    p_values_by_degradation: dict[str, str] = {}
    for degradation in DEGRADATION_ORDER:
        aligned = paired_by_degradation[degradation]
        if not aligned:
            p_values_by_degradation[degradation] = "NA"
            continue
        deltas = np.array(
            [values[METHOD_EXPANDED] - values[METHOD_CURRENT] for _, values in sorted(aligned.items())],
            dtype=np.float32,
        )
        p_values_by_degradation[degradation] = safe_wilcoxon(deltas)

    save_csv(
        RAW_CSV_PATH,
        raw_rows,
        ["trajectory_id", "seed", "degradation", "method", "ADE", "RMSE", "smooth", "OOR"],
    )

    aggregate_rows = build_aggregate_rows(raw_rows, p_values_by_degradation)
    save_csv(
        AGGREGATE_CSV_PATH,
        aggregate_rows,
        [
            "method",
            "degradation",
            "ADE_mean",
            "ADE_std",
            "ADE_median",
            "RMSE_mean",
            "smooth_mean",
            "OOR_rate",
            "n_traj",
            "n_seed",
            "source",
            "paired_wilcoxon_p_vs_current_2block",
        ],
    )

    expected_cells = 3 * len(DEGRADATION_ORDER)
    if len(aggregate_rows) != expected_cells:
        raise RuntimeError(f"Expected {expected_cells} aggregate rows, got {len(aggregate_rows)}")

    print_ade_table(aggregate_rows)
    print("Wilcoxon p-value summary:")
    for degradation in DEGRADATION_ORDER:
        print(f"{degradation}: {p_values_by_degradation[degradation]}")

    print("No missing cells in the 3x6 aggregate table: yes")
    print("PHASE 3 COMPLETE.")
    print("Generated:")
    print("- outputs/stage3_indoor/receptive_field_expansion/ablation_results.csv")
    print("- outputs/stage3_indoor/receptive_field_expansion/raw_4block_eval.csv")
    print_ade_table(aggregate_rows)
    print("Wilcoxon p-value summary:")
    for degradation in DEGRADATION_ORDER:
        print(f"{degradation}: {p_values_by_degradation[degradation]}")
    print(f"expanded_4block_conditional checkpoint used: {expanded_ckpt_path}")
    print(f"EMA used: {expanded_uses_ema}")
    print(f"current_2block_conditional checkpoint used: {current_ckpt_path}")
    print(f"Current 2-block EMA used: {current_uses_ema}")


if __name__ == "__main__":
    main()
