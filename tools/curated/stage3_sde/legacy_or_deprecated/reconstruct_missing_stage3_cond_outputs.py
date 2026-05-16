from pathlib import Path
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser_conditional import ConditionalTemporalDenoiser1D
from tools.stage3_indoor.train_cond_residual_gaussian import (
    load_state_dict_flexible,
    sample_conditional_residual,
)


DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
STAGE3_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "conditional_residual_ddpm_gaussian" / "seed42"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "reconstructed_stage3_cond_outputs"

CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
CKPT_PATH = STAGE3_DIR / "best_ema_model.pt"
NORM_PATH = STAGE3_DIR / "rel_norm_params_v2.npz"
SUMMARY_PATH = STAGE3_DIR / "generalization_summary.csv"

DEGRADED_PATHS = {
    "jump_medium": STAGE3_DIR / "generalization_degraded_jump.npy",
    "combined_medium": STAGE3_DIR / "generalization_degraded_combined.npy",
}
OUTPUT_PATHS = {
    "jump_medium": OUT_DIR / "jump_cond_residual_t20_refined_reconstructed.npy",
    "combined_medium": OUT_DIR / "combined_cond_residual_t20_refined_reconstructed.npy",
}

TIMESTEPS = 100
START_T = 20
SAMPLE_SEEDS = [42, 43, 44, 45, 46]
N_EVAL = 200


def require_inputs() -> None:
    required = [CLEAN_PATH, CKPT_PATH, NORM_PATH, SUMMARY_PATH, *DEGRADED_PATHS.values()]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required reconstruction inputs:\n" + "\n".join(str(path) for path in missing))


def load_stage3_target_ade() -> dict[str, float]:
    targets: dict[str, float] = {}
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["degradation"] in DEGRADED_PATHS and row["method"] == "cond_residual_t20":
                targets[row["degradation"]] = float(row["ADE_mean"])
    missing = sorted(set(DEGRADED_PATHS) - set(targets))
    if missing:
        raise RuntimeError(f"Missing Stage 3 target ADE rows for: {missing}")
    return targets


def compute_ade(pred: np.ndarray, clean: np.ndarray) -> tuple[float, float]:
    per_traj = np.linalg.norm(pred - clean, axis=-1).mean(axis=1)
    return float(per_traj.mean()), float(per_traj.std())


def reconstruct_for_degradation(
    degraded: np.ndarray,
    model: ConditionalTemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    preds = []
    for sample_seed in SAMPLE_SEEDS:
        preds.append(
            sample_conditional_residual(
                degraded_abs=degraded,
                model=model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                start_t=START_T,
                sample_seed=sample_seed,
                device=device,
            )
        )
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


def main() -> None:
    require_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    clean = clean_all[:N_EVAL].astype(np.float32)
    if clean.shape != (N_EVAL, 20, 2):
        raise ValueError(f"Unexpected clean eval shape: {clean.shape}")

    norm = np.load(NORM_PATH)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    target_ade = load_stage3_target_ade()

    device = torch.device("cpu")
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)
    model = ConditionalTemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=4, hidden_dim=128).to(device)
    load_state_dict_flexible(model, CKPT_PATH, device)
    model.eval()

    verification_rows = []
    for degradation, degraded_path in DEGRADED_PATHS.items():
        degraded = np.load(degraded_path).astype(np.float32)
        if degraded.shape != clean.shape:
            raise ValueError(f"Shape mismatch for {degradation}: degraded={degraded.shape}, clean={clean.shape}")
        pred = reconstruct_for_degradation(degraded, model, diffusion, rel_mean, rel_std, device)
        out_path = OUTPUT_PATHS[degradation]
        np.save(out_path, pred.astype(np.float32))

        ade_mean, ade_std = compute_ade(pred, clean)
        diff = abs(ade_mean - target_ade[degradation])
        verification_rows.append(
            {
                "degradation": degradation,
                "output_path": str(out_path),
                "shape": str(tuple(pred.shape)),
                "computed_ADE_mean": ade_mean,
                "computed_ADE_std": ade_std,
                "stage3_summary_ADE_mean": target_ade[degradation],
                "abs_diff_ADE_mean": diff,
                "matches_stage3_summary": diff < 1e-6,
            }
        )

    csv_path = OUT_DIR / "reconstruction_verification.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(verification_rows[0].keys()))
        writer.writeheader()
        writer.writerows(verification_rows)

    manifest = {
        "provenance": "Reconstructed in Stage 4 from Stage 3 saved degraded arrays, Stage 3 conditional residual EMA checkpoint, rel_norm_params_v2, t_start=20, and seeds [42,43,44,45,46]. These are not Stage 3 originally saved per-frame artifacts.",
        "clean_path": str(CLEAN_PATH),
        "checkpoint_path": str(CKPT_PATH),
        "norm_path": str(NORM_PATH),
        "summary_path": str(SUMMARY_PATH),
        "sample_seeds": SAMPLE_SEEDS,
        "start_t": START_T,
        "outputs": verification_rows,
    }
    (OUT_DIR / "reconstruction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Reconstructed Missing Stage 3 Conditional Outputs",
        "",
        "These files were reconstructed after Stage 3 using the original Stage 3 evaluation inputs and checkpoint. They are not original Stage 3 saved artifacts.",
        "",
        f"- checkpoint: `{CKPT_PATH}`",
        f"- normalization: `{NORM_PATH}`",
        f"- clean eval: `{CLEAN_PATH}[:{N_EVAL}]`",
        f"- t_start: {START_T}",
        f"- seeds: {SAMPLE_SEEDS}",
        "",
        "| degradation | output | shape | computed ADE | Stage 3 ADE | abs diff | match |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in verification_rows:
        lines.append(
            f"| {row['degradation']} | `{row['output_path']}` | {row['shape']} | "
            f"{row['computed_ADE_mean']:.9f} | {row['stage3_summary_ADE_mean']:.9f} | "
            f"{row['abs_diff_ADE_mean']:.3e} | {row['matches_stage3_summary']} |"
        )
    (OUT_DIR / "reconstruction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("RECONSTRUCT_MISSING_STAGE3_COND_OUTPUTS_COMPLETE")
    for row in verification_rows:
        print(
            f"{row['degradation']}: ADE={row['computed_ADE_mean']:.9f}, "
            f"stage3={row['stage3_summary_ADE_mean']:.9f}, match={row['matches_stage3_summary']}"
        )
    print(f"output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
