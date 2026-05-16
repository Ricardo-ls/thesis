from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3_dps_calibrated_phaseA_mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D
from tools.stage3_indoor import degrade as stage3_degrade
from tools.stage3_indoor.generate_indoor_trajs import (
    BEHAVIORS,
    CLIP_MAX,
    CLIP_MIN,
    FPS,
    ROOM_MAX,
    ROOM_MIN,
    T,
    add_micro_jitter,
    generate_boundary_walk,
    generate_goal_directed,
    generate_multi_goal,
    generate_pacing,
    generate_stationary,
)
from tools.stage3_indoor.sdedit_gaussian_full import run_sdedit


CONDITIONS = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
SEED_START = 14000
SEED_END = 14499
N = 500
SDEDIT_SEEDS = [42, 43, 44, 45, 46]
TAU_HIGH = 0.7
GAMMA = 2
T_SDEDIT = 1
T_DPS_START = 5
TIMESTEPS = 100
SIGMA0 = 0.05
KAPPA = 1.0
ZETAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
EPS = 1e-12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = PROJECT_ROOT / "data" / "stage4" / "e3_dps_calibration_500"
CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
DATA_META_PATH = DATA_DIR / "metadata.json"
DEGRADATION_META_PATH = DATA_DIR / "degradation_metadata.json"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_dps_calibrated_phaseA"
ARRAY_DIR = OUT_DIR / "arrays"
FIG_DIR = OUT_DIR / "figures"
METRICS_PATH = OUT_DIR / "e3_dps_calibrated_phaseA_metrics.csv"
CONDITION_SUMMARY_PATH = OUT_DIR / "e3_dps_calibrated_phaseA_condition_summary.csv"
ZETA_SUMMARY_PATH = OUT_DIR / "e3_dps_calibrated_phaseA_zeta_summary.csv"
STEP_DIAG_PATH = OUT_DIR / "e3_dps_calibrated_phaseA_step_diagnostics.csv"
SUMMARY_PATH = OUT_DIR / "e3_dps_calibrated_phaseA_decision_summary.md"

NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CKPT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    for path in [DATA_DIR, OUT_DIR, ARRAY_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(path)} written")


def print_file(path: Path) -> None:
    print(f"[FILE] {rel(path)} written")


def require_inputs() -> None:
    missing = [path for path in [NORM_PATH, CKPT_PATH] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))


def generate_one_from_seed(seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    behavior_names = [name for name, _ in BEHAVIORS]
    behavior_probs = [weight for _, weight in BEHAVIORS]
    behavior = str(rng.choice(behavior_names, p=behavior_probs))
    if behavior == "goal_directed":
        traj = generate_goal_directed(rng)
    elif behavior == "multi_goal":
        traj = generate_multi_goal(rng)
    elif behavior == "pacing":
        traj = generate_pacing(rng)
    elif behavior == "stationary":
        traj = generate_stationary(rng)
    elif behavior == "boundary_walk":
        traj = generate_boundary_walk(rng)
    else:
        raise ValueError(f"Unsupported behavior: {behavior}")
    traj = add_micro_jitter(traj, rng)
    traj = np.clip(traj, CLIP_MIN, CLIP_MAX).astype(np.float32)
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    return traj, {"seed": int(seed), "behavior": behavior, "mean_speed": float(steps.mean() * FPS)}


def generate_clean() -> np.ndarray:
    clean = np.zeros((N, T, 2), dtype=np.float32)
    meta_rows = []
    for idx, seed in enumerate(range(SEED_START, SEED_END + 1)):
        traj, meta = generate_one_from_seed(seed)
        clean[idx] = traj
        meta["trajectory_id"] = idx
        meta_rows.append(meta)
    np.save(CLEAN_PATH, clean)
    print_file(CLEAN_PATH)
    counts = {name: 0 for name, _ in BEHAVIORS}
    for row in meta_rows:
        counts[row["behavior"]] += 1
    write_json(
        DATA_META_PATH,
        {
            "seed_start": SEED_START,
            "seed_end": SEED_END,
            "n_trajectories": N,
            "trajectory_length": T,
            "shape": list(clean.shape),
            "room_size": [ROOM_MIN, ROOM_MAX],
            "clip_range": [CLIP_MIN, CLIP_MAX],
            "behavior_families": [{"name": name, "weight": weight} for name, weight in BEHAVIORS],
            "behavior_counts": counts,
            "generation_script_path": rel(Path(__file__).resolve()),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "overlap_check": {
                "no_overlap_with_e3_holdout1000_13000_13999": True,
                "no_overlap_with_train_1000_10999": True,
                "no_overlap_with_val_11000_12999": True,
            },
        },
    )
    return clean


def generate_degradations(clean: np.ndarray) -> dict[str, np.ndarray]:
    degraded = {
        "gaussian_medium": stage3_degrade.apply_gaussian(clean, sigma=0.05),
        "drift_medium": stage3_degrade.apply_drift(clean, sigma_step=0.010),
        "burst_medium": stage3_degrade.apply_burst(clean, burst_sigma=0.25, background_sigma=0.01),
        "bias_medium": stage3_degrade.apply_bias(clean, sigma=0.15),
        "jump_medium": stage3_degrade.apply_jump(clean),
        "combined_medium": stage3_degrade.apply_combined(clean, sigma_g=0.05, sigma_b=0.15, sigma_d=0.010),
    }
    meta: dict[str, dict] = {
        "input_clean_path": rel(CLEAN_PATH),
        "no_old_files_overwritten": True,
        "seed_rule_default": f"{stage3_degrade.BASE_SEED} + i",
        "combined_seed_rules": {"gaussian": "42 + i", "bias": "42000 + i", "drift": "84000 + i"},
    }
    for condition, arr in degraded.items():
        if arr.shape != clean.shape:
            raise ValueError(f"{condition}: degraded shape {arr.shape} != clean {clean.shape}")
        path = DATA_DIR / f"degraded_{condition}.npy"
        np.save(path, arr.astype(np.float32))
        print_file(path)
        meta[condition] = {"path": rel(path), "shape": list(arr.shape)}
    write_json(DEGRADATION_META_PATH, meta)
    return degraded


def load_model() -> tuple[TemporalDenoiser1D, DDPMForwardProcess, np.ndarray, np.ndarray]:
    norm = np.load(NORM_PATH)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=DEVICE)
    model = TemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=2, hidden_dim=128).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, diffusion, rel_mean, rel_std


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def per_traj_metrics(pred: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    err = frame_error(pred, clean)
    ade = err.mean(axis=1)
    rmse = np.sqrt(np.mean(err**2, axis=1))
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    acc_rms = np.sqrt(np.mean(np.sum(acc**2, axis=-1), axis=1))
    return ade.astype(np.float32), rmse.astype(np.float32), acc_rms.astype(np.float32)


def bin_masks(confidence: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "high": confidence > 0.7,
        "mid": (confidence >= 0.3) & (confidence <= 0.7),
        "low": confidence < 0.3,
    }


def masked_ade(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    err = frame_error(pred, clean)
    return float(err[mask].mean()) if int(mask.sum()) else math.nan


def compute_confidence(degraded: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, float]:
    delta0 = float(np.median(frame_error(degraded, clean).reshape(-1)))
    confidence = np.exp(-frame_error(degraded, clean) / delta0).astype(np.float32)
    return confidence, delta0


def hard_fuse(degraded: np.ndarray, sdedit: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    correction = sdedit - degraded
    lam = np.power(1.0 - confidence, GAMMA).astype(np.float32)
    lam = np.where(confidence > TAU_HIGH, 0.0, lam).astype(np.float32)
    return (degraded + lam[..., None] * correction).astype(np.float32)


def to_rel(traj: np.ndarray) -> np.ndarray:
    return (traj[:, 1:, :] - traj[:, :-1, :]).astype(np.float32)


def normalize_rel(rel: np.ndarray, rel_mean: np.ndarray, rel_std: np.ndarray) -> np.ndarray:
    return ((rel - rel_mean[None, None, :]) / rel_std[None, None, :]).astype(np.float32)


def denormalize_rel(rel_norm: np.ndarray, rel_mean: np.ndarray, rel_std: np.ndarray) -> np.ndarray:
    return (rel_norm * rel_std[None, None, :] + rel_mean[None, None, :]).astype(np.float32)


def reconstruct_abs(anchor: np.ndarray, rel: np.ndarray) -> np.ndarray:
    out = np.zeros((rel.shape[0], 20, 2), dtype=np.float32)
    out[:, 0, :] = anchor.astype(np.float32)
    out[:, 1:, :] = anchor[:, None, :] + np.cumsum(rel, axis=1)
    return out


def sigma_obs(confidence: np.ndarray) -> np.ndarray:
    return np.sqrt(SIGMA0**2 * (1.0 + KAPPA * (1.0 - confidence))).astype(np.float32)


def likelihood_norm_per_traj(pred_abs: np.ndarray, degraded_abs: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    weighted = (pred_abs - degraded_abs) / sigma_obs(confidence)[..., None]
    return np.linalg.norm(weighted.reshape(weighted.shape[0], -1), axis=1).astype(np.float32)


def ddpm_step(x_t: torch.Tensor, eps_pred: torch.Tensor, t_idx: int, diffusion: DDPMForwardProcess) -> torch.Tensor:
    alpha_t = diffusion.alphas[t_idx]
    alpha_bar_t = diffusion.alpha_bars[t_idx]
    beta_t = diffusion.betas[t_idx]
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_pred)
    if t_idx > 0:
        return mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
    return mean


def x0_hat_abs_uncond(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t_idx: int,
    anchor_t: torch.Tensor,
    rel_mean_t: torch.Tensor,
    rel_std_t: torch.Tensor,
    diffusion: DDPMForwardProcess,
) -> torch.Tensor:
    alpha_bar = diffusion.alpha_bars[t_idx]
    x0_norm = (x_t - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
    rel_norm = x0_norm.permute(0, 2, 1)
    rel = rel_norm * rel_std_t.view(1, 1, 2) + rel_mean_t.view(1, 1, 2)
    cumsum = torch.cumsum(rel, dim=1)
    return torch.cat([anchor_t[:, None, :], anchor_t[:, None, :] + cumsum], dim=1)


def final_abs_from_xt(x_t: torch.Tensor, anchor: np.ndarray, rel_mean: np.ndarray, rel_std: np.ndarray) -> np.ndarray:
    rel_norm = x_t.detach().permute(0, 2, 1).cpu().numpy().astype(np.float32)
    rel = denormalize_rel(rel_norm, rel_mean, rel_std)
    return reconstruct_abs(anchor, rel)


def run_sdedit_t1(
    condition: str,
    degraded: np.ndarray,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
) -> np.ndarray:
    out_path = ARRAY_DIR / f"{condition}_sdedit_t1.npy"
    if out_path.is_file():
        cached = np.load(out_path).astype(np.float32)
        if cached.shape == degraded.shape:
            print(f"[CACHE] {rel(out_path)}")
            return cached
    preds = []
    for seed in SDEDIT_SEEDS:
        preds.append(
            run_sdedit(
                degraded_abs=degraded,
                model=model,
                diffusion=diffusion,
                rel_mean=rel_mean,
                rel_std=rel_std,
                t_start=T_SDEDIT,
                sdedit_seed=seed,
                device=DEVICE,
            ).astype(np.float32)
        )
    out = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)
    np.save(out_path, out)
    print(f"[FILE] {rel(out_path)} written")
    return out


def run_dps(
    condition: str,
    zeta: float,
    arrays: dict[str, np.ndarray],
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
) -> tuple[np.ndarray, list[dict], dict]:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    fused = arrays["fused"]
    init_rel_norm = normalize_rel(to_rel(fused), rel_mean, rel_std)
    anchor = degraded[:, 0, :].astype(np.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    x0 = torch.from_numpy(init_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32)
    t = torch.full((x0.shape[0],), T_DPS_START, device=DEVICE, dtype=torch.long)
    x_t, _ = diffusion.q_sample(x0, t)
    y_t = torch.from_numpy(degraded).to(DEVICE, dtype=torch.float32)
    conf_t = torch.from_numpy(confidence).to(DEVICE, dtype=torch.float32)
    anchor_t = torch.from_numpy(anchor).to(DEVICE, dtype=torch.float32)
    rel_mean_t = torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32)
    trace_rows = []
    max_grad = 0.0
    grad_sum = 0.0
    update_max = 0.0
    update_sum = 0.0
    step_count = 0
    likelihood_increase_steps = 0
    ade_increase_steps = 0
    nonfinite_cases = 0
    prev_ade = float(per_traj_metrics(fused, clean)[0].mean())
    for t_idx in reversed(range(T_DPS_START + 1)):
        x_t = x_t.detach().requires_grad_(True)
        t_cur = torch.full((x_t.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
        eps_pred = model(x_t, t_cur)
        abs_before = x0_hat_abs_uncond(x_t, eps_pred, t_idx, anchor_t, rel_mean_t, rel_std_t, diffusion)
        sigma_t = torch.sqrt(SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t)))
        weighted = (abs_before - y_t) / sigma_t[..., None]
        per_sample_norm = torch.linalg.vector_norm(weighted.reshape(weighted.shape[0], -1), dim=1)
        loss = per_sample_norm.sum()
        grad = torch.autograd.grad(loss, x_t, retain_graph=False)[0]
        with torch.no_grad():
            x_base = ddpm_step(x_t, eps_pred, t_idx, diffusion)
            update = float(zeta) * grad
            update_norm_per = torch.linalg.vector_norm(update.reshape(update.shape[0], -1), dim=1)
            xbase_norm_per = torch.linalg.vector_norm(x_base.reshape(x_base.shape[0], -1), dim=1)
            update_ratio_per = update_norm_per / (xbase_norm_per + EPS)
            x_after = x_base - update
            t_after = torch.full((x_after.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
            eps_after = model(x_after, t_after)
            abs_after = x0_hat_abs_uncond(x_after, eps_after, t_idx, anchor_t, rel_mean_t, rel_std_t, diffusion)
            weighted_after = (abs_after - y_t) / sigma_t[..., None]
            per_after = torch.linalg.vector_norm(weighted_after.reshape(weighted_after.shape[0], -1), dim=1)
            before_l = float(per_sample_norm.mean().cpu())
            after_l = float(per_after.mean().cpu())
            abs_after_np = abs_after.detach().cpu().numpy().astype(np.float32)
            after_ade = float(per_traj_metrics(abs_after_np, clean)[0].mean())
            grad_norm_per = torch.linalg.vector_norm(grad.detach().reshape(grad.shape[0], -1), dim=1)
            max_grad = max(max_grad, float(grad_norm_per.max().cpu()))
            grad_sum += float(grad_norm_per.mean().cpu())
            update_max = max(update_max, float(update_ratio_per.max().cpu()))
            update_sum += float(update_ratio_per.mean().cpu())
            step_count += 1
            likelihood_increase_steps += int(after_l > before_l)
            ade_increase_steps += int(after_ade > prev_ade)
            nonfinite_cases += int((~torch.isfinite(x_after.reshape(x_after.shape[0], -1)).all(dim=1)).sum().cpu())
            trace_rows.append(
                {
                    "condition": condition,
                    "zeta": zeta,
                    "t": t_idx,
                    "likelihood_before_step": before_l,
                    "likelihood_after_step": after_l,
                    "likelihood_increased": bool(after_l > before_l),
                    "ADE_before_step": prev_ade,
                    "ADE_after_step": after_ade,
                    "ADE_increased": bool(after_ade > prev_ade),
                    "grad_norm_mean": float(grad_norm_per.mean().cpu()),
                    "grad_norm_max": float(grad_norm_per.max().cpu()),
                    "update_to_x_ratio_mean": float(update_ratio_per.mean().cpu()),
                    "update_to_x_ratio_max": float(update_ratio_per.max().cpu()),
                    "finite": bool(torch.isfinite(x_after).all().cpu()),
                }
            )
            x_t = x_after.detach()
            prev_ade = after_ade
    pred = final_abs_from_xt(x_t, anchor, rel_mean, rel_std)
    diagnostics = {
        "finite": bool(np.isfinite(pred).all()),
        "nonfinite_cases": int(nonfinite_cases),
        "max_grad_norm": max_grad,
        "mean_grad_norm": grad_sum / max(step_count, 1),
        "max_update_to_x_ratio": update_max,
        "mean_update_to_x_ratio": update_sum / max(step_count, 1),
        "likelihood_increase_steps": likelihood_increase_steps,
        "ade_increase_steps": ade_increase_steps,
    }
    return pred.astype(np.float32), trace_rows, diagnostics


def metric_row(condition: str, zeta: float, pred: np.ndarray, arrays: dict[str, np.ndarray], diagnostics: dict) -> dict:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    fused = arrays["fused"]
    ade_dps, rmse_dps, acc_dps = per_traj_metrics(pred, clean)
    ade_e3, _, acc_e3 = per_traj_metrics(fused, clean)
    masks = bin_masks(confidence)
    high_dps = masked_ade(pred, clean, masks["high"])
    mid_dps = masked_ade(pred, clean, masks["mid"])
    low_dps = masked_ade(pred, clean, masks["low"])
    high_e3 = masked_ade(fused, clean, masks["high"])
    low_e3 = masked_ade(fused, clean, masks["low"])
    lik_before = likelihood_norm_per_traj(fused, degraded, confidence)
    lik_after = likelihood_norm_per_traj(pred, degraded, confidence)
    return {
        "condition": condition,
        "zeta": zeta,
        "finite": bool(diagnostics["finite"]),
        "ADE_E3_fused": float(ade_e3.mean()),
        "ADE_DPS": float(ade_dps.mean()),
        "ADE_delta": float(ade_dps.mean() - ade_e3.mean()),
        "ADE_improvement_percent": float((ade_e3.mean() - ade_dps.mean()) / max(float(ade_e3.mean()), EPS) * 100.0),
        "RMSE": float(rmse_dps.mean()),
        "acceleration_RMS": float(acc_dps.mean()),
        "acceleration_RMS_E3_fused": float(acc_e3.mean()),
        "acceleration_ratio_vs_E3": float(acc_dps.mean() / max(float(acc_e3.mean()), EPS)),
        "high_conf_ADE": high_dps,
        "mid_conf_ADE": mid_dps,
        "low_conf_ADE": low_dps,
        "high_conf_ADE_E3_fused": high_e3,
        "low_conf_ADE_E3_fused": low_e3,
        "high_conf_ratio_vs_E3_fused": float(high_dps / high_e3) if np.isfinite(high_dps) and high_e3 > EPS else math.nan,
        "low_conf_ratio_vs_E3_fused": float(low_dps / low_e3) if np.isfinite(low_dps) and low_e3 > EPS else math.nan,
        "win_rate_vs_E3_fused": float(np.mean(ade_dps < ade_e3)),
        "likelihood_before_DPS": float(lik_before.mean()),
        "likelihood_after_DPS": float(lik_after.mean()),
        "likelihood_delta": float(lik_after.mean() - lik_before.mean()),
        "max_grad_norm": float(diagnostics["max_grad_norm"]),
        "mean_grad_norm": float(diagnostics["mean_grad_norm"]),
        "max_update_to_x_ratio": float(diagnostics["max_update_to_x_ratio"]),
        "mean_update_to_x_ratio": float(diagnostics["mean_update_to_x_ratio"]),
        "likelihood_increase_steps": int(diagnostics["likelihood_increase_steps"]),
        "ADE_increase_steps": int(diagnostics["ade_increase_steps"]),
        "nan_inf_cases": int(diagnostics["nonfinite_cases"]),
    }


def summarize_condition(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.copy()


def summarize_zeta(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zeta, sub in metrics_df.groupby("zeta"):
        mean_e3 = float(sub["ADE_E3_fused"].mean())
        mean_dps = float(sub["ADE_DPS"].mean())
        high_ratio = float(sub["high_conf_ADE"].mean() / max(sub["high_conf_ADE_E3_fused"].mean(), EPS))
        acc_ratio = float(sub["acceleration_RMS"].mean() / max(sub["acceleration_RMS_E3_fused"].mean(), EPS))
        likelihood_decreased = bool(sub["likelihood_after_DPS"].mean() <= sub["likelihood_before_DPS"].mean())
        finite_ok = bool(sub["finite"].all() and int(sub["nan_inf_cases"].sum()) == 0)
        improvement = (mean_e3 - mean_dps) / max(mean_e3, EPS) * 100.0
        max_update = float(sub["max_update_to_x_ratio"].max())
        pass_signal = bool(
            finite_ok
            and mean_dps <= 0.99 * mean_e3
            and high_ratio <= 1.02
            and acc_ratio <= 1.10
            and likelihood_decreased
            and max_update <= 0.01
        )
        promising = bool(finite_ok and 0.0 < improvement < 1.0 and high_ratio <= 1.02 and acc_ratio <= 1.10)
        decision = "PASS_SIGNAL" if pass_signal else "PROMISING" if promising else "FAIL"
        rows.append(
            {
                "zeta": zeta,
                "mean_ADE_E3_fused": mean_e3,
                "mean_ADE_DPS": mean_dps,
                "ADE_delta": mean_dps - mean_e3,
                "ADE_improvement_percent": improvement,
                "high_conf_ratio_vs_E3": high_ratio,
                "acceleration_ratio_vs_E3": acc_ratio,
                "likelihood_before": float(sub["likelihood_before_DPS"].mean()),
                "likelihood_after": float(sub["likelihood_after_DPS"].mean()),
                "likelihood_decreased": likelihood_decreased,
                "max_update_to_x_ratio": max_update,
                "mean_update_to_x_ratio": float(sub["mean_update_to_x_ratio"].mean()),
                "finite_ok": finite_ok,
                "nan_inf_cases": int(sub["nan_inf_cases"].sum()),
                "audit_decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values("zeta")


def plot_representative(condition: str, arrays: dict[str, np.ndarray], preds: dict[float, np.ndarray]) -> None:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    fused = arrays["fused"]
    confidence = arrays["confidence"]
    e3_ade = per_traj_metrics(fused, clean)[0]
    dps_ade = per_traj_metrics(preds[1e-4], clean)[0]
    idx = int(np.argmax(np.abs(dps_ade - e3_ade)))
    frames = np.arange(20)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.plot(clean[idx, :, 0], clean[idx, :, 1], "k-", label="clean")
    ax.plot(degraded[idx, :, 0], degraded[idx, :, 1], color="tab:orange", label="degraded")
    ax.plot(fused[idx, :, 0], fused[idx, :, 1], color="tab:blue", label="E3 fused")
    for zeta, pred in preds.items():
        ax.plot(pred[idx, :, 0], pred[idx, :, 1], linewidth=1.0, label=f"DPS {zeta:g}")
    ax.set_title(f"{condition} idx={idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6)
    ax = axes[1]
    ax.plot(frames, np.linalg.norm(fused[idx] - clean[idx], axis=-1), color="tab:blue", label="E3 fused")
    for zeta, pred in preds.items():
        ax.plot(frames, np.linalg.norm(pred[idx] - clean[idx], axis=-1), linewidth=1.0, label=f"DPS {zeta:g}")
    ax.set_title("Error curve")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6)
    ax = axes[2]
    ax.plot(frames, confidence[idx], color="tab:purple")
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.3, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Confidence")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = FIG_DIR / f"representative_{condition}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print_file(path)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = df[cols].copy()
    for col in show.columns:
        show[col] = show[col].map(
            lambda value: f"{float(value):.9f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value)
        )
    lines = ["| " + " | ".join(show.columns) + " |", "| " + " | ".join(["---"] * len(show.columns)) + " |"]
    for row in show.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(zeta_df: pd.DataFrame) -> None:
    pass_rows = zeta_df[zeta_df["audit_decision"] == "PASS_SIGNAL"]
    promising_rows = zeta_df[zeta_df["audit_decision"] == "PROMISING"]
    if not pass_rows.empty:
        decision = "PASS_SIGNAL"
        locked = float(pass_rows.sort_values("mean_ADE_DPS").iloc[0]["zeta"])
        recommendation = f"Recommend Phase B confirm on seed 14500-14999 with zeta={locked:g} pre-locked."
    elif not promising_rows.empty:
        decision = "PROMISING"
        locked = math.nan
        recommendation = "Keep DPS as supplementary / optional polish. Do not enter Phase B as a main result."
    else:
        decision = "FAIL"
        locked = math.nan
        recommendation = "Stop DPS and keep E3 fusion as final stable method."
    lines = [
        "# Calibrated E3-Initialized DPS Phase A Audit",
        "",
        "This is a calibration audit, not training, not formal E4, and not final method selection.",
        "",
        "## Setup Answers",
        "",
        f"1. Calibration hold-out generated: `yes`, seed range `{SEED_START}-{SEED_END}`.",
        "2. Six degradations generated: `yes`.",
        "3. E3 fused baseline reproduced: `yes`.",
        f"4. Zeta values evaluated: `{ZETAS}`.",
        "",
        "## Zeta Summary",
        "",
        markdown_table(
            zeta_df,
            [
                "zeta",
                "mean_ADE_E3_fused",
                "mean_ADE_DPS",
                "ADE_improvement_percent",
                "high_conf_ratio_vs_E3",
                "acceleration_ratio_vs_E3",
                "likelihood_decreased",
                "max_update_to_x_ratio",
                "finite_ok",
                "audit_decision",
            ],
        ),
        "",
        "## Decision",
        "",
        f"- Overall Phase A decision: `{decision}`",
        f"- Locked zeta for Phase B, if any: `{locked}`",
        f"- Recommendation: {recommendation}",
        "",
        "## Output Files",
        "",
        f"- `{rel(METRICS_PATH)}`",
        f"- `{rel(CONDITION_SUMMARY_PATH)}`",
        f"- `{rel(ZETA_SUMMARY_PATH)}`",
        f"- `{rel(STEP_DIAG_PATH)}`",
        f"- `{rel(FIG_DIR)}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_file(SUMMARY_PATH)


def main() -> None:
    require_inputs()
    ensure_dirs()
    clean = generate_clean()
    degraded_map = generate_degradations(clean)
    model, diffusion, rel_mean, rel_std = load_model()
    arrays_by_condition: dict[str, dict[str, np.ndarray]] = {}
    metrics_rows: list[dict] = []
    step_rows: list[dict] = []
    preds_for_fig: dict[str, dict[float, np.ndarray]] = {}
    for condition in CONDITIONS:
        degraded = degraded_map[condition].astype(np.float32)
        confidence, _ = compute_confidence(degraded, clean)
        sdedit = run_sdedit_t1(condition, degraded, model, diffusion, rel_mean, rel_std)
        fused = hard_fuse(degraded, sdedit, confidence)
        arrays = {"clean": clean, "degraded": degraded, "confidence": confidence, "sdedit": sdedit, "fused": fused}
        arrays_by_condition[condition] = arrays
        np.save(ARRAY_DIR / f"{condition}_clean.npy", clean.astype(np.float32))
        np.save(ARRAY_DIR / f"{condition}_degraded.npy", degraded.astype(np.float32))
        np.save(ARRAY_DIR / f"{condition}_confidence.npy", confidence.astype(np.float32))
        np.save(ARRAY_DIR / f"{condition}_fused_t1_tau07_gamma2.npy", fused.astype(np.float32))
        print(f"[FILE] {rel(ARRAY_DIR / f'{condition}_fused_t1_tau07_gamma2.npy')} written")
        preds_for_fig[condition] = {}
        for zeta in ZETAS:
            pred, trace, diagnostics = run_dps(condition, zeta, arrays, model, diffusion, rel_mean, rel_std)
            np.save(ARRAY_DIR / f"{condition}_dps_zeta_{zeta:g}.npy", pred.astype(np.float32))
            preds_for_fig[condition][zeta] = pred
            metrics_rows.append(metric_row(condition, zeta, pred, arrays, diagnostics))
            step_rows.extend(trace)
            print(f"[DONE] condition={condition} zeta={zeta:g}")
    metrics_df = pd.DataFrame(metrics_rows)
    condition_df = summarize_condition(metrics_df)
    zeta_df = summarize_zeta(metrics_df)
    step_df = pd.DataFrame(step_rows)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print_file(METRICS_PATH)
    condition_df.to_csv(CONDITION_SUMMARY_PATH, index=False)
    print_file(CONDITION_SUMMARY_PATH)
    zeta_df.to_csv(ZETA_SUMMARY_PATH, index=False)
    print_file(ZETA_SUMMARY_PATH)
    step_df.to_csv(STEP_DIAG_PATH, index=False)
    print_file(STEP_DIAG_PATH)
    for condition in CONDITIONS:
        plot_representative(condition, arrays_by_condition[condition], preds_for_fig[condition])
    write_summary(zeta_df)
    print("E3_DPS_CALIBRATED_PHASEA_COMPLETE")
    print(zeta_df.to_string(index=False))
    print(f"output_dir={rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
