from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3_initialized_dps_aligned_audit_mpl")

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


CONDITIONS = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
SEEDS = [42, 43, 44]
ZETAS = [0.3, 1.0, 3.0]
TIMESTEPS = 100
T_DPS_START = 5
SIGMA0 = 0.05
KAPPA = 1.0
EPS = 1e-12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

E3_ARRAY_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_holdout1000_confidence_aware_sdedit" / "arrays"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CKPT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_initialized_dps_aligned_audit"
FIG_DIR = OUT_DIR / "figures"
METRICS_PATH = OUT_DIR / "e3_initialized_dps_aligned_metrics.csv"
CONDITION_SUMMARY_PATH = OUT_DIR / "e3_initialized_dps_aligned_condition_summary.csv"
SEED_ZETA_SUMMARY_PATH = OUT_DIR / "e3_initialized_dps_aligned_seed_zeta_summary.csv"
STEP_TRACE_PATH = OUT_DIR / "e3_initialized_dps_aligned_step_trace.csv"
SUMMARY_PATH = OUT_DIR / "e3_initialized_dps_aligned_decision_summary.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_file(path: Path) -> None:
    print(f"[FILE] {rel(path)} written")


def require_inputs() -> None:
    missing = [path for path in [E3_ARRAY_DIR, NORM_PATH, CKPT_PATH] if not path.exists()]
    for condition in CONDITIONS:
        for suffix in ["clean", "degraded", "confidence", "fused_t1_tau07_gamma2"]:
            path = E3_ARRAY_DIR / f"{condition}_{suffix}.npy"
            if not path.is_file():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing required E3 frozen inputs:\n" + "\n".join(str(path) for path in missing))


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_model() -> tuple[TemporalDenoiser1D, DDPMForwardProcess, np.ndarray, np.ndarray]:
    norm = np.load(NORM_PATH)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
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
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=DEVICE)
    return model, diffusion, rel_mean, rel_std


def load_arrays(condition: str) -> dict[str, np.ndarray]:
    arrays = {
        "clean": np.load(E3_ARRAY_DIR / f"{condition}_clean.npy").astype(np.float32),
        "degraded": np.load(E3_ARRAY_DIR / f"{condition}_degraded.npy").astype(np.float32),
        "confidence": np.load(E3_ARRAY_DIR / f"{condition}_confidence.npy").astype(np.float32),
        "e3_fused": np.load(E3_ARRAY_DIR / f"{condition}_fused_t1_tau07_gamma2.npy").astype(np.float32),
    }
    clean_shape = arrays["clean"].shape
    if clean_shape != (1000, 20, 2):
        raise ValueError(f"{condition}: unexpected clean shape {clean_shape}")
    for name, arr in arrays.items():
        expected = clean_shape[:2] if name == "confidence" else clean_shape
        if arr.shape != expected:
            raise ValueError(f"{condition}: {name} shape {arr.shape} != {expected}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{condition}: {name} contains non-finite values")
    return arrays


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


def sigma_obs(confidence: np.ndarray) -> np.ndarray:
    sigma2 = SIGMA0**2 * (1.0 + KAPPA * (1.0 - confidence))
    return np.sqrt(sigma2).astype(np.float32)


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


def run_dps_polish(
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    arrays: dict[str, np.ndarray],
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    seed: int,
    zeta: float,
) -> tuple[np.ndarray, list[dict], dict]:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    e3_fused = arrays["e3_fused"]
    init_rel = to_rel(e3_fused)
    init_rel_norm = normalize_rel(init_rel, rel_mean, rel_std)
    anchor = degraded[:, 0, :].astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = torch.from_numpy(init_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32)
    t = torch.full((x0.shape[0],), T_DPS_START, device=DEVICE, dtype=torch.long)
    x_t, _ = diffusion.q_sample(x0, t)

    y_t = torch.from_numpy(degraded).to(DEVICE, dtype=torch.float32)
    conf_t = torch.from_numpy(confidence).to(DEVICE, dtype=torch.float32)
    anchor_t = torch.from_numpy(anchor).to(DEVICE, dtype=torch.float32)
    rel_mean_t = torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32)

    trace_rows: list[dict] = []
    max_update_to_x_ratio = 0.0
    nonfinite_cases = 0

    for t_idx in reversed(range(T_DPS_START + 1)):
        x_t = x_t.detach().requires_grad_(True)
        t_cur = torch.full((x_t.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
        eps_pred = model(x_t, t_cur)
        abs_hat = x0_hat_abs_uncond(x_t, eps_pred, t_idx, anchor_t, rel_mean_t, rel_std_t, diffusion)
        sigma_t = torch.sqrt(SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t)))
        weighted = (abs_hat - y_t) / sigma_t[..., None]
        per_sample_norm = torch.linalg.vector_norm(weighted.reshape(weighted.shape[0], -1), dim=1)
        loss = per_sample_norm.sum()
        loss_isfinite = bool(torch.isfinite(loss).detach().cpu())
        if loss_isfinite:
            grad = torch.autograd.grad(loss, x_t, retain_graph=False)[0]
        else:
            grad = torch.full_like(x_t, float("nan"))
        grad_isfinite = bool(torch.isfinite(grad).detach().all().cpu())

        with torch.no_grad():
            x_base = ddpm_step(x_t, eps_pred, t_idx, diffusion)
            base_isfinite = bool(torch.isfinite(x_base).detach().all().cpu())
            update = float(zeta) * grad
            update_isfinite = bool(torch.isfinite(update).detach().all().cpu())
            update_norm_per = torch.linalg.vector_norm(update.detach().reshape(update.shape[0], -1), dim=1)
            xbase_norm_per = torch.linalg.vector_norm(x_base.detach().reshape(x_base.shape[0], -1), dim=1)
            update_to_x_per = update_norm_per / (xbase_norm_per + EPS)
            if bool(torch.isfinite(update_to_x_per).all().cpu()):
                max_update_to_x_ratio = max(max_update_to_x_ratio, float(update_to_x_per.max().cpu()))
            x_after = x_base - update
            after_isfinite = bool(torch.isfinite(x_after).detach().all().cpu())
            nonfinite_mask = ~torch.isfinite(x_after.detach().reshape(x_after.shape[0], -1)).all(dim=1)
            nonfinite_cases += int(nonfinite_mask.sum().cpu())

        trace_rows.append(
            {
                "seed": seed,
                "zeta": zeta,
                "t": t_idx,
                "loss_mean": float(per_sample_norm.detach().mean().cpu()) if loss_isfinite else math.nan,
                "loss_before_guidance_sum": float(loss.detach().cpu()) if loss_isfinite else math.nan,
                "grad_norm_mean": float(torch.linalg.vector_norm(grad.detach().reshape(grad.shape[0], -1), dim=1).mean().cpu()) if grad_isfinite else math.nan,
                "grad_norm_max": float(torch.linalg.vector_norm(grad.detach().reshape(grad.shape[0], -1), dim=1).max().cpu()) if grad_isfinite else math.nan,
                "update_to_x_ratio_mean": float(update_to_x_per.detach().mean().cpu()) if update_isfinite and base_isfinite else math.nan,
                "update_to_x_ratio_max": float(update_to_x_per.detach().max().cpu()) if update_isfinite and base_isfinite else math.nan,
                "x_after_isfinite": after_isfinite,
                "loss_isfinite": loss_isfinite,
                "grad_isfinite": grad_isfinite,
            }
        )
        x_t = x_after.detach()

    final_abs = final_abs_from_xt(x_t, anchor, rel_mean, rel_std)
    finite = bool(np.isfinite(final_abs).all())
    if not finite:
        nonfinite_cases += int(np.sum(~np.isfinite(final_abs).reshape(final_abs.shape[0], -1).all(axis=1)))
    diagnostics = {
        "finite": finite,
        "nonfinite_cases": int(nonfinite_cases),
        "max_update_to_x_ratio": max_update_to_x_ratio,
        "initialization": "E3_fused_t1_tau07_gamma2_q_sample_t5",
        "pure_gaussian_initialization": False,
    }
    return final_abs.astype(np.float32), trace_rows, diagnostics


def aggregate_row(
    condition: str,
    seed: int,
    zeta: float,
    pred: np.ndarray,
    arrays: dict[str, np.ndarray],
    diagnostics: dict,
) -> dict:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    e3_fused = arrays["e3_fused"]
    ade_traj, rmse_traj, acc_traj = per_traj_metrics(pred, clean)
    e3_ade_traj, _, e3_acc_traj = per_traj_metrics(e3_fused, clean)
    mask = bin_masks(confidence)
    high_ade = masked_ade(pred, clean, mask["high"])
    mid_ade = masked_ade(pred, clean, mask["mid"])
    low_ade = masked_ade(pred, clean, mask["low"])
    e3_high = masked_ade(e3_fused, clean, mask["high"])
    e3_mid = masked_ade(e3_fused, clean, mask["mid"])
    e3_low = masked_ade(e3_fused, clean, mask["low"])
    lik_before = likelihood_norm_per_traj(e3_fused, degraded, confidence)
    lik_after = likelihood_norm_per_traj(pred, degraded, confidence)
    return {
        "condition": condition,
        "seed": seed,
        "zeta": zeta,
        "N": int(clean.shape[0]),
        "ADE": float(ade_traj.mean()),
        "RMSE": float(rmse_traj.mean()),
        "acceleration_RMS": float(acc_traj.mean()),
        "high_conf_ADE": high_ade,
        "mid_conf_ADE": mid_ade,
        "low_conf_ADE": low_ade,
        "E3_fused_ADE": float(e3_ade_traj.mean()),
        "E3_fused_acceleration_RMS": float(e3_acc_traj.mean()),
        "E3_fused_high_conf_ADE": e3_high,
        "E3_fused_mid_conf_ADE": e3_mid,
        "E3_fused_low_conf_ADE": e3_low,
        "win_rate_vs_E3_fused": float(np.mean(ade_traj < e3_ade_traj)),
        "ADE_delta_vs_E3_fused": float(ade_traj.mean() - e3_ade_traj.mean()),
        "ADE_ratio_vs_E3_fused": float(ade_traj.mean() / max(float(e3_ade_traj.mean()), EPS)),
        "high_conf_delta_vs_E3_fused": float(high_ade - e3_high) if np.isfinite(high_ade) and np.isfinite(e3_high) else math.nan,
        "high_conf_ratio_vs_E3_fused": float(high_ade / e3_high) if np.isfinite(high_ade) and np.isfinite(e3_high) and e3_high > EPS else math.nan,
        "acceleration_ratio_vs_E3_fused": float(acc_traj.mean() / max(float(e3_acc_traj.mean()), EPS)),
        "likelihood_norm_before_DPS": float(lik_before.mean()),
        "likelihood_norm_after_DPS": float(lik_after.mean()),
        "likelihood_delta_after_minus_before": float(lik_after.mean() - lik_before.mean()),
        "likelihood_decreased": bool(lik_after.mean() <= lik_before.mean()),
        "update_to_x_ratio": float(diagnostics["max_update_to_x_ratio"]),
        "finite": bool(diagnostics["finite"]),
        "nan_inf_cases": int(diagnostics["nonfinite_cases"]),
    }


def summarize_condition(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, zeta), sub in metrics_df.groupby(["condition", "zeta"]):
        rows.append(
            {
                "condition": condition,
                "zeta": zeta,
                "ADE_mean_over_seeds": float(sub["ADE"].mean()),
                "ADE_std_over_seeds": float(sub["ADE"].std(ddof=0)),
                "E3_fused_ADE": float(sub["E3_fused_ADE"].mean()),
                "ADE_delta_vs_E3_fused": float(sub["ADE_delta_vs_E3_fused"].mean()),
                "ADE_improvement_percent_vs_E3": float((sub["E3_fused_ADE"].mean() - sub["ADE"].mean()) / max(sub["E3_fused_ADE"].mean(), EPS) * 100.0),
                "high_conf_ADE": float(sub["high_conf_ADE"].mean()),
                "E3_fused_high_conf_ADE": float(sub["E3_fused_high_conf_ADE"].mean()),
                "high_conf_ratio_vs_E3": float(sub["high_conf_ADE"].mean() / max(sub["E3_fused_high_conf_ADE"].mean(), EPS)),
                "acceleration_RMS": float(sub["acceleration_RMS"].mean()),
                "E3_fused_acceleration_RMS": float(sub["E3_fused_acceleration_RMS"].mean()),
                "acceleration_ratio_vs_E3": float(sub["acceleration_RMS"].mean() / max(sub["E3_fused_acceleration_RMS"].mean(), EPS)),
                "likelihood_before": float(sub["likelihood_norm_before_DPS"].mean()),
                "likelihood_after": float(sub["likelihood_norm_after_DPS"].mean()),
                "likelihood_decreased": bool(sub["likelihood_norm_after_DPS"].mean() <= sub["likelihood_norm_before_DPS"].mean()),
                "nan_inf_cases": int(sub["nan_inf_cases"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_seed_zeta(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zeta, sub in metrics_df.groupby("zeta"):
        mean_ade = float(sub["ADE"].mean())
        mean_e3 = float(sub["E3_fused_ADE"].mean())
        high_ratio = float(sub["high_conf_ADE"].mean() / max(sub["E3_fused_high_conf_ADE"].mean(), EPS))
        acc_ratio = float(sub["acceleration_RMS"].mean() / max(sub["E3_fused_acceleration_RMS"].mean(), EPS))
        likelihood_down = bool(sub["likelihood_norm_after_DPS"].mean() <= sub["likelihood_norm_before_DPS"].mean())
        finite_ok = bool(sub["finite"].all() and int(sub["nan_inf_cases"].sum()) == 0)
        ade_improvement_pct = (mean_e3 - mean_ade) / max(mean_e3, EPS) * 100.0
        pass_signal = bool(
            finite_ok
            and ade_improvement_pct >= 1.0
            and high_ratio <= 1.02
            and acc_ratio <= 1.10
            and likelihood_down
        )
        promising = bool(
            finite_ok
            and 0.0 < ade_improvement_pct < 1.0
            and high_ratio <= 1.05
            and acc_ratio <= 1.10
        )
        if pass_signal:
            decision = "PASS_SIGNAL"
        elif promising:
            decision = "PROMISING"
        else:
            decision = "FAIL"
        rows.append(
            {
                "zeta": zeta,
                "six_condition_mean_ADE": mean_ade,
                "six_condition_mean_E3_fused_ADE": mean_e3,
                "ADE_delta_vs_E3_fused": mean_ade - mean_e3,
                "ADE_improvement_percent_vs_E3": ade_improvement_pct,
                "high_conf_ratio_vs_E3": high_ratio,
                "high_conf_preserved_2pct": bool(high_ratio <= 1.02),
                "acceleration_ratio_vs_E3": acc_ratio,
                "roughening_over_10pct": bool(acc_ratio > 1.10),
                "likelihood_before": float(sub["likelihood_norm_before_DPS"].mean()),
                "likelihood_after": float(sub["likelihood_norm_after_DPS"].mean()),
                "likelihood_decreased": likelihood_down,
                "finite_ok": finite_ok,
                "nan_inf_cases": int(sub["nan_inf_cases"].sum()),
                "max_update_to_x_ratio": float(sub["update_to_x_ratio"].max()),
                "audit_decision": decision,
            }
        )
    return pd.DataFrame(rows)


def plot_representative(condition: str, arrays: dict[str, np.ndarray], pred: np.ndarray, zeta: float) -> None:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    e3 = arrays["e3_fused"]
    confidence = arrays["confidence"]
    e3_ade = frame_error(e3, clean).mean(axis=1)
    pred_ade = frame_error(pred, clean).mean(axis=1)
    idx = int(np.argmax(np.abs(pred_ade - e3_ade)))
    frames = np.arange(clean.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(clean[idx, :, 0], clean[idx, :, 1], "k-", label="clean")
    ax.plot(degraded[idx, :, 0], degraded[idx, :, 1], color="tab:orange", label="degraded")
    ax.plot(e3[idx, :, 0], e3[idx, :, 1], color="tab:blue", label="E3 fused")
    ax.plot(pred[idx, :, 0], pred[idx, :, 1], color="tab:red", label=f"E3+DPS zeta={zeta}")
    ax.set_title(f"{condition} representative idx={idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(frames, np.linalg.norm(degraded[idx] - clean[idx], axis=-1), color="tab:orange", label="degraded")
    ax.plot(frames, np.linalg.norm(e3[idx] - clean[idx], axis=-1), color="tab:blue", label="E3 fused")
    ax.plot(frames, np.linalg.norm(pred[idx] - clean[idx], axis=-1), color="tab:red", label="E3+DPS")
    ax.set_title("Per-frame error")
    ax.set_xlabel("frame")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(frames, confidence[idx], color="tab:purple")
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.3, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Oracle confidence")
    ax.set_xlabel("frame")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = FIG_DIR / f"representative_{condition}_zeta{str(zeta).replace('.', 'p')}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print_file(path)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = df[cols].copy()
    for col in show.columns:
        show[col] = show[col].map(
            lambda value: f"{float(value):.6f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value)
        )
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(seed_zeta_df: pd.DataFrame, condition_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    any_pass = bool((seed_zeta_df["audit_decision"] == "PASS_SIGNAL").any())
    any_promising = bool((seed_zeta_df["audit_decision"] == "PROMISING").any())
    all_finite = bool(seed_zeta_df["finite_ok"].all())
    best_decision = "PASS_SIGNAL" if any_pass else "PROMISING" if any_promising else "FAIL"
    if any_pass:
        recommendation = "Recommend formal E4 validation on seed 14000-14999 with the passing zeta pre-locked. Do not select zeta from this frozen set as a final result."
    elif any_promising:
        recommendation = "Keep DPS as optional future/supplementary work; do not promote it over the frozen E3 fusion baseline yet."
    else:
        recommendation = "Stop DPS for now and keep E3 confidence-aware fusion as the final stable method."
    lines = [
        "# E3-Initialized DPS Aligned Audit",
        "",
        "This is an aligned audit, not formal E4 and not hyperparameter selection. It reads frozen E3 `fused_t1_tau07_gamma2` arrays, initializes from them, and runs short t=5 norm-based DPS polish for all zeta values.",
        "",
        "## Setup Answers",
        "",
        "1. Initialized from E3 fused_t1_tau07_gamma2: `yes`.",
        "2. Pure Gaussian initialization avoided: `yes`.",
        f"3. 0 NaN / Inf: `{all_finite}`.",
        "",
        "## Zeta Summary",
        "",
        markdown_table(
            seed_zeta_df,
            [
                "zeta",
                "six_condition_mean_ADE",
                "six_condition_mean_E3_fused_ADE",
                "ADE_improvement_percent_vs_E3",
                "high_conf_ratio_vs_E3",
                "acceleration_ratio_vs_E3",
                "likelihood_decreased",
                "nan_inf_cases",
                "audit_decision",
            ],
        ),
        "",
        "## Decision Questions",
        "",
    ]
    for _, row in seed_zeta_df.iterrows():
        lines.append(
            f"- zeta={row['zeta']}: mean ADE `{row['six_condition_mean_ADE']:.6f}`, "
            f"improvement vs E3 `{row['ADE_improvement_percent_vs_E3']:.6f}%`, "
            f"high-conf ratio `{row['high_conf_ratio_vs_E3']:.6f}`, "
            f"acceleration ratio `{row['acceleration_ratio_vs_E3']:.6f}`, "
            f"likelihood decreased `{row['likelihood_decreased']}`, decision `{row['audit_decision']}`."
        )
    lines += [
        "",
        f"Overall audit decision: `{best_decision}`.",
        f"Recommendation: {recommendation}",
        "",
        "## Output Files",
        "",
        f"- `{rel(METRICS_PATH)}`",
        f"- `{rel(CONDITION_SUMMARY_PATH)}`",
        f"- `{rel(SEED_ZETA_SUMMARY_PATH)}`",
        f"- `{rel(STEP_TRACE_PATH)}`",
        f"- `{rel(FIG_DIR)}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_file(SUMMARY_PATH)


def main() -> None:
    require_inputs()
    ensure_dirs()
    model, diffusion, rel_mean, rel_std = load_model()
    print(f"[CHECK] device={DEVICE}")
    print("[CHECK] initialization=E3_fused_t1_tau07_gamma2, pure_gaussian=False")
    print(f"[CHECK] t_dps_start={T_DPS_START}, zetas={ZETAS}, seeds={SEEDS}")

    metrics_rows: list[dict] = []
    trace_rows: list[dict] = []
    representative_preds: dict[str, np.ndarray] = {}
    arrays_by_condition: dict[str, dict[str, np.ndarray]] = {}

    for condition in CONDITIONS:
        arrays = load_arrays(condition)
        arrays_by_condition[condition] = arrays
        for zeta in ZETAS:
            for seed in SEEDS:
                pred, trace, diagnostics = run_dps_polish(model, diffusion, arrays, rel_mean, rel_std, seed, zeta)
                for row in trace:
                    row["condition"] = condition
                    trace_rows.append(row)
                metrics_rows.append(aggregate_row(condition, seed, zeta, pred, arrays, diagnostics))
                if seed == 42 and zeta == 0.3:
                    representative_preds[condition] = pred
                print(f"[DONE] condition={condition} seed={seed} zeta={zeta}")

    metrics_df = pd.DataFrame(metrics_rows)
    condition_df = summarize_condition(metrics_df)
    seed_zeta_df = summarize_seed_zeta(metrics_df)
    trace_df = pd.DataFrame(trace_rows)

    metrics_df.to_csv(METRICS_PATH, index=False)
    print_file(METRICS_PATH)
    condition_df.to_csv(CONDITION_SUMMARY_PATH, index=False)
    print_file(CONDITION_SUMMARY_PATH)
    seed_zeta_df.to_csv(SEED_ZETA_SUMMARY_PATH, index=False)
    print_file(SEED_ZETA_SUMMARY_PATH)
    trace_df.to_csv(STEP_TRACE_PATH, index=False)
    print_file(STEP_TRACE_PATH)

    for condition in CONDITIONS:
        plot_representative(condition, arrays_by_condition[condition], representative_preds[condition], 0.3)

    write_summary(seed_zeta_df, condition_df, metrics_df)

    print("E3_INITIALIZED_DPS_ALIGNED_AUDIT_COMPLETE")
    print(seed_zeta_df.to_string(index=False))
    print(f"output_dir={rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
