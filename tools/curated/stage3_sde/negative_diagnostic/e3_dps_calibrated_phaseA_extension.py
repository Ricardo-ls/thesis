from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3_dps_phaseA_extension_mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
ZETAS_EXTENSION = [0.0, 3e-3]
TIMESTEPS = 100
T_DPS_START = 5
SIGMA0 = 0.05
KAPPA = 1.0
EPS = 1e-12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHASEA_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_dps_calibrated_phaseA"
ARRAY_DIR = PHASEA_DIR / "arrays"
PHASEA_ZETA_SUMMARY = PHASEA_DIR / "e3_dps_calibrated_phaseA_zeta_summary.csv"
PHASEA_METRICS = PHASEA_DIR / "e3_dps_calibrated_phaseA_metrics.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_dps_calibrated_phaseA_extension"
METRICS_PATH = OUT_DIR / "e3_dps_phaseA_extension_metrics.csv"
ZETA_SUMMARY_PATH = OUT_DIR / "e3_dps_phaseA_extension_zeta_summary.csv"
SUMMARY_PATH = OUT_DIR / "e3_dps_phaseA_extension_decision_summary.md"

NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CKPT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_inputs() -> None:
    required = [NORM_PATH, CKPT_PATH, PHASEA_ZETA_SUMMARY, PHASEA_METRICS]
    for condition in CONDITIONS:
        for suffix in ["clean", "degraded", "confidence", "fused_t1_tau07_gamma2"]:
            required.append(ARRAY_DIR / f"{condition}_{suffix}.npy")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Phase A inputs:\n" + "\n".join(str(path) for path in missing))


def print_file(path: Path) -> None:
    print(f"[FILE] {rel(path)} written")


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


def load_arrays(condition: str) -> dict[str, np.ndarray]:
    arrays = {
        "clean": np.load(ARRAY_DIR / f"{condition}_clean.npy").astype(np.float32),
        "degraded": np.load(ARRAY_DIR / f"{condition}_degraded.npy").astype(np.float32),
        "confidence": np.load(ARRAY_DIR / f"{condition}_confidence.npy").astype(np.float32),
        "fused": np.load(ARRAY_DIR / f"{condition}_fused_t1_tau07_gamma2.npy").astype(np.float32),
    }
    for name, arr in arrays.items():
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
    return np.sqrt(SIGMA0**2 * (1.0 + KAPPA * (1.0 - confidence))).astype(np.float32)


def likelihood_norm_per_traj(pred_abs: np.ndarray, degraded_abs: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    weighted = (pred_abs - degraded_abs) / sigma_obs(confidence)[..., None]
    return np.linalg.norm(weighted.reshape(weighted.shape[0], -1), axis=1).astype(np.float32)


def run_dps(
    condition: str,
    zeta: float,
    arrays: dict[str, np.ndarray],
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
) -> tuple[np.ndarray, dict]:
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    fused = arrays["fused"]
    init_rel_norm = normalize_rel(to_rel(fused), rel_mean, rel_std)
    anchor = degraded[:, 0, :].astype(np.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    x0 = torch.from_numpy(init_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32)
    t = torch.full((x0.shape[0],), 5, device=DEVICE, dtype=torch.long)
    x_t, _ = diffusion.q_sample(x0, t)
    y_t = torch.from_numpy(degraded).to(DEVICE, dtype=torch.float32)
    conf_t = torch.from_numpy(confidence).to(DEVICE, dtype=torch.float32)
    anchor_t = torch.from_numpy(anchor).to(DEVICE, dtype=torch.float32)
    rel_mean_t = torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32)
    update_max = 0.0
    update_sum = 0.0
    steps = 0
    for t_idx in reversed(range(6)):
        x_t = x_t.detach().requires_grad_(zeta != 0.0)
        t_cur = torch.full((x_t.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
        eps_pred = model(x_t, t_cur)
        with torch.set_grad_enabled(zeta != 0.0):
            abs_hat = x0_hat_abs_uncond(x_t, eps_pred, t_idx, anchor_t, rel_mean_t, rel_std_t, diffusion)
            sigma_t = torch.sqrt(SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t)))
            weighted = (abs_hat - y_t) / sigma_t[..., None]
            loss = torch.linalg.vector_norm(weighted.reshape(weighted.shape[0], -1), dim=1).sum()
            grad = torch.autograd.grad(loss, x_t, retain_graph=False)[0] if zeta != 0.0 else torch.zeros_like(x_t)
        with torch.no_grad():
            x_base = ddpm_step(x_t, eps_pred, t_idx, diffusion)
            update = float(zeta) * grad
            if zeta != 0.0:
                update_ratio = torch.linalg.vector_norm(update.reshape(update.shape[0], -1), dim=1) / (
                    torch.linalg.vector_norm(x_base.reshape(x_base.shape[0], -1), dim=1) + EPS
                )
                update_max = max(update_max, float(update_ratio.max().cpu()))
                update_sum += float(update_ratio.mean().cpu())
                steps += 1
            x_t = (x_base - update).detach()
    pred = final_abs_from_xt(x_t, anchor, rel_mean, rel_std)
    return pred.astype(np.float32), {
        "finite": bool(np.isfinite(pred).all()),
        "max_update_to_x_ratio": update_max,
        "mean_update_to_x_ratio": update_sum / max(steps, 1),
        "nan_inf_cases": int(np.sum(~np.isfinite(pred).reshape(pred.shape[0], -1).all(axis=1))),
    }


def metric_row(condition: str, zeta: float, pred: np.ndarray, arrays: dict[str, np.ndarray], diagnostics: dict) -> dict:
    clean = arrays["clean"]
    degraded = arrays["degraded"]
    confidence = arrays["confidence"]
    fused = arrays["fused"]
    ade_dps, rmse_dps, acc_dps = per_traj_metrics(pred, clean)
    ade_e3, _, acc_e3 = per_traj_metrics(fused, clean)
    masks = bin_masks(confidence)
    high_dps = masked_ade(pred, clean, masks["high"])
    high_e3 = masked_ade(fused, clean, masks["high"])
    low_dps = masked_ade(pred, clean, masks["low"])
    low_e3 = masked_ade(fused, clean, masks["low"])
    lik_before = likelihood_norm_per_traj(fused, degraded, confidence)
    lik_after = likelihood_norm_per_traj(pred, degraded, confidence)
    return {
        "condition": condition,
        "zeta": zeta,
        "ADE_E3_fused": float(ade_e3.mean()),
        "ADE_DPS": float(ade_dps.mean()),
        "ADE_delta": float(ade_dps.mean() - ade_e3.mean()),
        "ADE_improvement_percent": float((ade_e3.mean() - ade_dps.mean()) / max(float(ade_e3.mean()), EPS) * 100.0),
        "RMSE": float(rmse_dps.mean()),
        "acceleration_RMS": float(acc_dps.mean()),
        "acceleration_RMS_E3_fused": float(acc_e3.mean()),
        "acceleration_ratio_vs_E3": float(acc_dps.mean() / max(float(acc_e3.mean()), EPS)),
        "high_conf_ADE": high_dps,
        "high_conf_ADE_E3_fused": high_e3,
        "high_conf_ratio_vs_E3": float(high_dps / high_e3) if np.isfinite(high_dps) and high_e3 > EPS else math.nan,
        "low_conf_ADE": low_dps,
        "low_conf_ADE_E3_fused": low_e3,
        "low_conf_ratio_vs_E3": float(low_dps / low_e3) if np.isfinite(low_dps) and low_e3 > EPS else math.nan,
        "win_rate_vs_E3_fused": float(np.mean(ade_dps < ade_e3)),
        "likelihood_before": float(lik_before.mean()),
        "likelihood_after": float(lik_after.mean()),
        "likelihood_delta": float(lik_after.mean() - lik_before.mean()),
        "likelihood_decreased": bool(lik_after.mean() <= lik_before.mean()),
        "max_update_to_x_ratio": float(diagnostics["max_update_to_x_ratio"]),
        "mean_update_to_x_ratio": float(diagnostics["mean_update_to_x_ratio"]),
        "finite": bool(diagnostics["finite"]),
        "nan_inf_cases": int(diagnostics["nan_inf_cases"]),
    }


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zeta, sub in metrics_df.groupby("zeta"):
        mean_e3 = float(sub["ADE_E3_fused"].mean())
        mean_dps = float(sub["ADE_DPS"].mean())
        improvement = (mean_e3 - mean_dps) / max(mean_e3, EPS) * 100.0
        high_ratio = float(sub["high_conf_ADE"].mean() / max(sub["high_conf_ADE_E3_fused"].mean(), EPS))
        acc_ratio = float(sub["acceleration_RMS"].mean() / max(sub["acceleration_RMS_E3_fused"].mean(), EPS))
        finite_ok = bool(sub["finite"].all() and int(sub["nan_inf_cases"].sum()) == 0)
        pass_signal = bool(
            finite_ok
            and mean_dps <= 0.99 * mean_e3
            and high_ratio <= 1.02
            and acc_ratio <= 1.10
            and sub["likelihood_after"].mean() <= sub["likelihood_before"].mean()
            and float(sub["max_update_to_x_ratio"].max()) <= 0.01
        )
        promising = bool(finite_ok and 0.0 < improvement < 1.0 and high_ratio <= 1.02 and acc_ratio <= 1.10)
        rows.append(
            {
                "zeta": zeta,
                "mean_ADE_E3_fused": mean_e3,
                "mean_ADE_DPS": mean_dps,
                "ADE_delta": mean_dps - mean_e3,
                "ADE_improvement_percent": improvement,
                "high_conf_ratio_vs_E3": high_ratio,
                "acceleration_ratio_vs_E3": acc_ratio,
                "likelihood_before": float(sub["likelihood_before"].mean()),
                "likelihood_after": float(sub["likelihood_after"].mean()),
                "likelihood_decreased": bool(sub["likelihood_after"].mean() <= sub["likelihood_before"].mean()),
                "max_update_to_x_ratio": float(sub["max_update_to_x_ratio"].max()),
                "finite_ok": finite_ok,
                "nan_inf_cases": int(sub["nan_inf_cases"].sum()),
                "audit_decision": "PASS_SIGNAL" if pass_signal else "PROMISING" if promising else "FAIL",
            }
        )
    return pd.DataFrame(rows).sort_values("zeta")


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    for col in show.columns:
        show[col] = show[col].map(
            lambda value: f"{float(value):.9f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value)
        )
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(ext_summary: pd.DataFrame, phasea_summary: pd.DataFrame) -> None:
    z0 = ext_summary[ext_summary["zeta"] == 0.0].iloc[0]
    z3 = ext_summary[ext_summary["zeta"] == 0.003].iloc[0]
    z1e3 = phasea_summary[phasea_summary["zeta"] == 0.001].iloc[0]
    phase_b = bool(z3["audit_decision"] == "PASS_SIGNAL")
    lines = [
        "# E3-DPS Phase A Extension",
        "",
        "This extension only evaluates zeta=0 no-guidance re-diffusion and zeta=3e-3 on the existing Phase A calibration set. It does not generate new data, train, change likelihood, change t_start, or enter Phase B.",
        "",
        "## Key Questions",
        "",
        f"1. zeta=0 ADE: `{float(z0['mean_ADE_DPS']):.9f}`.",
        f"2. zeta=0 impact vs E3 fused: delta `{float(z0['ADE_delta']):.9f}`, improvement `{float(z0['ADE_improvement_percent']):.6f}%`.",
        f"3. zeta=1e-3 ADE from Phase A: `{float(z1e3['mean_ADE_DPS']):.9f}`.",
        f"4. zeta=3e-3 ADE: `{float(z3['mean_ADE_DPS']):.9f}`.",
        f"5. zeta=3e-3 better than zeta=1e-3: `{bool(float(z3['mean_ADE_DPS']) < float(z1e3['mean_ADE_DPS']))}`.",
        f"6. zeta=3e-3 beats E3 fused baseline: `{bool(float(z3['mean_ADE_DPS']) < float(z3['mean_ADE_E3_fused']))}`.",
        f"7. zeta=3e-3 likelihood decreased: `{bool(z3['likelihood_decreased'])}`.",
        f"8. zeta=3e-3 high-conf ratio vs E3: `{float(z3['high_conf_ratio_vs_E3']):.9f}`.",
        f"9. zeta=3e-3 acceleration ratio vs E3: `{float(z3['acceleration_ratio_vs_E3']):.9f}`.",
        f"10. Recommend Phase B: `{phase_b}`.",
        "",
        "## Extension Zeta Summary",
        "",
        markdown_table(ext_summary),
        "",
        "## Decision",
        "",
    ]
    if phase_b:
        lines.append("zeta=3e-3 passes Phase A extension criteria; Phase B can be considered only with zeta=3e-3 pre-locked.")
    else:
        lines.append("No Phase B is recommended. Keep E3 confidence-aware fusion as the final stable method.")
    lines += [
        "",
        "## Output Files",
        "",
        f"- `{rel(METRICS_PATH)}`",
        f"- `{rel(ZETA_SUMMARY_PATH)}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_file(SUMMARY_PATH)


def main() -> None:
    require_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, diffusion, rel_mean, rel_std = load_model()
    metrics_rows = []
    for condition in CONDITIONS:
        arrays = load_arrays(condition)
        for zeta in ZETAS_EXTENSION:
            pred, diagnostics = run_dps(condition, zeta, arrays, model, diffusion, rel_mean, rel_std)
            np.save(OUT_DIR / f"{condition}_dps_zeta_{zeta:g}.npy", pred.astype(np.float32))
            metrics_rows.append(metric_row(condition, zeta, pred, arrays, diagnostics))
            print(f"[DONE] condition={condition} zeta={zeta:g}")
    metrics_df = pd.DataFrame(metrics_rows)
    ext_summary = summarize(metrics_df)
    phasea_summary = pd.read_csv(PHASEA_ZETA_SUMMARY)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print_file(METRICS_PATH)
    ext_summary.to_csv(ZETA_SUMMARY_PATH, index=False)
    print_file(ZETA_SUMMARY_PATH)
    write_summary(ext_summary, phasea_summary)
    print("E3_DPS_PHASEA_EXTENSION_COMPLETE")
    print(ext_summary.to_string(index=False))
    print(f"output_dir={rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
