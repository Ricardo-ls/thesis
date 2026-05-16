from __future__ import annotations

from pathlib import Path
import json
import math
import os
import sys
from datetime import datetime

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
import pandas as pd
import torch

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser_conditional import ConditionalTemporalDenoiser1D

try:
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False


PRE_REG_PATH = PROJECT_ROOT / "docs" / "stage4" / "E2_DPS_hypothesis.md"
AMENDMENT_001_PATH = PROJECT_ROOT / "docs" / "stage4" / "E2_DPS_hypothesis_amendment_001.md"
AMENDMENT_002_PATH = PROJECT_ROOT / "docs" / "stage4" / "E2_DPS_hypothesis_amendment_002.md"
FORMAL_E1_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_oracle_residual_gating_6conditions"
CONF_CACHE_DIR = FORMAL_E1_DIR / "confidence_cache"
CONF_VALIDATION_PATH = CONF_CACHE_DIR / "confidence_cache_validation_summary.csv"
PROTOCOL_PATH = FORMAL_E1_DIR / "e1_protocol_validation_summary.csv"
E1_BIN_PATH = FORMAL_E1_DIR / "e1_confidence_bin_metrics.csv"
E1_PASS_FAIL_PATH = FORMAL_E1_DIR / "e1_pass_fail_summary.csv"
E2_MIN_TRAJ_PATH = PROJECT_ROOT / "outputs" / "stage4" / "e2_min_absolute_posterior_anchoring" / "e2_min_optimized_trajectories.npz"
VAL_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "val_trajs.npy"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CKPT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "stage3_indoor"
    / "conditional_residual_ddpm_gaussian"
    / "seed42"
    / "best_ema_model.pt"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_dps_pilot"
FIG_DIR = OUT_DIR / "figures"
AUDIT_FAILED_PATH = OUT_DIR / "e2_dps_pilot_audit_failed.md"
AUDIT_SUMMARY_PATH = OUT_DIR / "e2_dps_pilot_audit_summary.md"
FULL_METRICS_PATH = OUT_DIR / "e2_dps_pilot_full_metrics.csv"
BIN_METRICS_PATH = OUT_DIR / "e2_dps_pilot_confidence_bin_metrics.csv"
PER_TRAJ_PATH = OUT_DIR / "e2_dps_pilot_per_trajectory_metrics.csv"
PASS_FAIL_PATH = OUT_DIR / "e2_dps_pilot_pass_fail_summary.csv"
DIAGNOSTICS_PATH = OUT_DIR / "e2_dps_pilot_diagnostics.csv"
SUMMARY_PATH = OUT_DIR / "e2_dps_pilot_summary.md"

DEGRADATION_ORDER = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
ZETAS = [0.3, 1.0, 3.0]
SEEDS = [42, 43, 44]
N_PILOT = 50
TIMESTEPS = 100
SIGMA0 = 0.05
KAPPA = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAPSHOT_STEPS = [99, 80, 60, 40, 20, 0]
EPS = 1e-12


def print_file(path: Path) -> None:
    print(f"[FILE] {path} written")


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print_file(path)


def write_audit_failed(title: str, details: list[str], checks: list[str] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E2-DPS Stage 8A Pilot Audit Failed",
        "",
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Failure: {title}",
        "",
        "## Details",
    ]
    lines.extend(f"- {item}" for item in details)
    if checks:
        lines.extend(["", "## Completed Checks"])
        lines.extend(f"- {item}" for item in checks)
    lines.extend(
        [
            "",
            "## Action",
            "Pilot sampling was not started. Stage 8B was not started. This follows the locked E2-DPS pre-registration and the Stage 8A instruction to stop on path or data problems.",
        ]
    )
    write_text_file(AUDIT_FAILED_PATH, "\n".join(lines) + "\n")


def require_file(path: Path, message: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{message}: {path}")


def A_relative_to_absolute(rel_disp: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    abs_pos = np.zeros((rel_disp.shape[0] + 1, 2), dtype=np.float32)
    abs_pos[0] = anchor.astype(np.float32)
    abs_pos[1:] = anchor.astype(np.float32)[None, :] + np.cumsum(rel_disp.astype(np.float32), axis=0)
    return abs_pos


def roundtrip_check(y: np.ndarray) -> float:
    rel_from_y = y[1:] - y[:-1]
    y_reconstructed = A_relative_to_absolute(rel_from_y, y[0])
    return float(np.max(np.abs(y_reconstructed - y)))


def to_rel(trajs: np.ndarray) -> np.ndarray:
    return (trajs[:, 1:, :] - trajs[:, :-1, :]).astype(np.float32)


def normalize_rel(rel: np.ndarray, rel_mean: np.ndarray, rel_std: np.ndarray) -> np.ndarray:
    return ((rel - rel_mean[None, None, :]) / rel_std[None, None, :]).astype(np.float32)


def reconstruct_from_rel(start_points: np.ndarray, rel: np.ndarray) -> np.ndarray:
    abs_hat = np.zeros((rel.shape[0], 20, 2), dtype=np.float32)
    abs_hat[:, 0, :] = start_points.astype(np.float32)
    abs_hat[:, 1:, :] = start_points[:, None, :] + np.cumsum(rel, axis=1)
    return abs_hat


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def per_traj_metrics(pred: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    err = frame_error(pred, clean)
    ade = err.mean(axis=1)
    rmse = np.sqrt(np.mean(err**2, axis=1))
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    smooth = np.sqrt(np.mean(np.sum(acc**2, axis=-1), axis=1))
    return ade.astype(np.float32), rmse.astype(np.float32), smooth.astype(np.float32)


def stats(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan, "p25": math.nan, "p75": math.nan}
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


def safe_wilcoxon(delta: np.ndarray) -> float:
    if not SCIPY_AVAILABLE:
        return math.nan
    vals = np.asarray(delta, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or np.allclose(vals, 0.0):
        return 1.0
    try:
        return float(wilcoxon(vals, alternative="less").pvalue)
    except Exception:
        return math.nan


def load_state_dict_flexible(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)


def load_model() -> tuple[ConditionalTemporalDenoiser1D, DDPMForwardProcess, np.ndarray, np.ndarray]:
    norm = np.load(NORM_PATH)
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    model = ConditionalTemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=4, hidden_dim=128).to(DEVICE)
    load_state_dict_flexible(model, CKPT_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=DEVICE)
    return model, diffusion, rel_mean, rel_std


def load_inputs() -> tuple[pd.DataFrame, dict[str, dict], np.ndarray]:
    protocol = pd.read_csv(PROTOCOL_PATH)
    gt = np.load(VAL_PATH).astype(np.float32)[:N_PILOT]
    if gt.shape != (N_PILOT, 20, 2):
        raise ValueError(f"Unexpected ground truth shape from val_trajs.npy[:50]: {gt.shape}")
    e2_npz = np.load(E2_MIN_TRAJ_PATH)
    loaded: dict[str, dict] = {}
    for degradation in DEGRADATION_ORDER:
        row = protocol[protocol["degradation"] == degradation]
        if row.empty:
            raise RuntimeError(f"Missing protocol row for {degradation}")
        row = row.iloc[0]
        degraded = np.load(str(row["degraded_path"])).astype(np.float32)[:N_PILOT]
        formal_key = f"{degradation}__formal_e1"
        e2_key = f"{degradation}__V2_uniform_cond_motion"
        if formal_key not in e2_npz.files or e2_key not in e2_npz.files:
            raise RuntimeError(f"Missing Formal E1/E2-Min V2 arrays in {E2_MIN_TRAJ_PATH} for {degradation}")
        formal_e1 = e2_npz[formal_key].astype(np.float32)[:N_PILOT]
        e2_min = e2_npz[e2_key].astype(np.float32)[:N_PILOT]
        if degraded.shape != (N_PILOT, 20, 2) or formal_e1.shape != degraded.shape or e2_min.shape != degraded.shape:
            raise ValueError(f"Shape mismatch for {degradation}: degraded={degraded.shape}, e1={formal_e1.shape}, e2min={e2_min.shape}")
        loaded[degradation] = {
            "degraded": degraded,
            "formal_e1": formal_e1,
            "e2_min_v2": e2_min,
            "conditional_source": str(row["conditional_source"]),
            "degraded_path": str(row["degraded_path"]),
        }
    return protocol, loaded, gt


def load_confidence_cache() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    validation = pd.read_csv(CONF_VALIDATION_PATH)
    if set(validation["degradation"]) != set(DEGRADATION_ORDER):
        raise RuntimeError("Confidence validation summary does not cover all six conditions")
    if not bool(validation["validation_pass"].all()):
        raise RuntimeError("One or more confidence cache validation rows are false")
    confs: dict[str, np.ndarray] = {}
    for degradation in DEGRADATION_ORDER:
        path = CONF_CACHE_DIR / f"{degradation}_confidence.npy"
        require_file(path, f"Missing confidence cache for {degradation}")
        c_t_cache = np.load(path).astype(np.float32)
        c_t_now = np.load(path).astype(np.float32)
        if c_t_cache.shape != (200, 20):
            raise ValueError(f"Unexpected confidence shape for {degradation}: {c_t_cache.shape}")
        bit_identical = bool(np.array_equal(c_t_now, c_t_cache))
        print(f"[CHECK] {degradation} c_t shape = {c_t_cache.shape}")
        print(f"[CHECK] {degradation} c_t bit-identical to validated cache: {bit_identical}")
        if not bit_identical:
            raise AssertionError("c_t drift between cache and E2-DPS runtime")
        confs[degradation] = c_t_cache[:N_PILOT]
    print("[CHECK] c_t loaded from validated reconstructed Formal E1 cache: True")
    print("[CHECK] c_t bit-identical to cache: True")
    return validation, confs


def validate_confidence_counts(validation_df: pd.DataFrame, confs: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for degradation in DEGRADATION_ORDER:
        full_conf = np.load(CONF_CACHE_DIR / f"{degradation}_confidence.npy").astype(np.float32)
        row = validation_df[validation_df["degradation"] == degradation].iloc[0]
        full_high = int((full_conf > 0.7).sum())
        full_low = int((full_conf < 0.3).sum())
        full_mid = int(((full_conf >= 0.3) & (full_conf <= 0.7)).sum())
        expected_high = int(row["n_high_conf_frames"])
        expected_low = int(row["n_low_conf_frames"])
        expected_mid = int(row["n_mid_conf_frames"])
        pilot = confs[degradation]
        pilot_high = int((pilot > 0.7).sum())
        pilot_low = int((pilot < 0.3).sum())
        pilot_mid = int(((pilot >= 0.3) & (pilot <= 0.7)).sum())
        print(
            f"[CHECK] {degradation} confidence bins full_200: "
            f"high={full_high} mid={full_mid} low={full_low}; "
            f"pilot_50: high={pilot_high} mid={pilot_mid} low={pilot_low}"
        )
        if full_high != expected_high or full_low != expected_low:
            raise AssertionError(f"Confidence full_200 high/low count mismatch for {degradation}")
        rows.append(
            {
                "degradation": degradation,
                "full_high": full_high,
                "full_mid": full_mid,
                "full_low": full_low,
                "pilot_high": pilot_high,
                "pilot_mid": pilot_mid,
                "pilot_low": pilot_low,
            }
        )
    return rows


def sigma_obs_stats(confs: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for degradation, c_t in confs.items():
        sigma2 = SIGMA0**2 * (1.0 + KAPPA * (1.0 - c_t))
        row = {
            "degradation": degradation,
            "sigma_obs2_min": float(sigma2.min()),
            "sigma_obs2_mean": float(sigma2.mean()),
            "sigma_obs2_max": float(sigma2.max()),
        }
        rows.append(row)
        print(
            f"[CHECK] {degradation} sigma_obs^2 min/mean/max = "
            f"{row['sigma_obs2_min']:.6f}/{row['sigma_obs2_mean']:.6f}/{row['sigma_obs2_max']:.6f}"
        )
    return rows


def x0_hat_abs_from_xt(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t_idx: int,
    degraded_rel_norm_t: torch.Tensor,
    start_t: torch.Tensor,
    rel_mean_t: torch.Tensor,
    rel_std_t: torch.Tensor,
    diffusion: DDPMForwardProcess,
) -> torch.Tensor:
    alpha_bar = diffusion.alpha_bars[t_idx]
    x0_residual_norm = (x_t - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
    final_rel_norm = degraded_rel_norm_t + x0_residual_norm.permute(0, 2, 1)
    final_rel = final_rel_norm * rel_std_t.view(1, 1, 2) + rel_mean_t.view(1, 1, 2)
    cumsum = torch.cumsum(final_rel, dim=1)
    return torch.cat([start_t[:, None, :], start_t[:, None, :] + cumsum], dim=1)


def ddpm_step(x_t: torch.Tensor, eps_pred: torch.Tensor, t_idx: int, diffusion: DDPMForwardProcess) -> torch.Tensor:
    alpha_t = diffusion.alphas[t_idx]
    alpha_bar_t = diffusion.alpha_bars[t_idx]
    beta_t = diffusion.betas[t_idx]
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_pred)
    if t_idx > 0:
        return mean + torch.sqrt(beta_t) * torch.randn_like(x_t)
    return mean


def weighted_observation_loss(abs_hat: torch.Tensor, y_t: torch.Tensor, conf_t: torch.Tensor) -> torch.Tensor:
    sigma2 = SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t))
    sigma = torch.sqrt(sigma2)
    weighted_residual = (abs_hat - y_t) / sigma[..., None]
    return torch.linalg.vector_norm(weighted_residual)


def likelihood_sign_check(
    model: ConditionalTemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    data: dict[str, dict],
    confs: dict[str, np.ndarray],
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
) -> dict:
    degradation = "drift_medium"
    y = data[degradation]["degraded"][:1]
    conf = confs[degradation][:1]
    degraded_rel = to_rel(y)
    degraded_rel_norm = normalize_rel(degraded_rel, rel_mean, rel_std)
    rel_mean_t = torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32)
    x_cond = torch.from_numpy(degraded_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32)
    y_t = torch.from_numpy(y).to(DEVICE, dtype=torch.float32)
    c_t = torch.from_numpy(conf).to(DEVICE, dtype=torch.float32)
    start_t = y_t[:, 0, :]
    torch.manual_seed(12345)
    x_t = torch.randn((1, 2, 19), device=DEVICE, dtype=torch.float32, requires_grad=True)
    t_idx = TIMESTEPS - 1
    t_cur = torch.full((1,), t_idx, device=DEVICE, dtype=torch.long)
    eps = model(x_t, x_cond, t_cur)
    abs_hat = x0_hat_abs_from_xt(x_t, eps, t_idx, torch.from_numpy(degraded_rel_norm).to(DEVICE), start_t, rel_mean_t, rel_std_t, diffusion)
    loss = weighted_observation_loss(abs_hat, y_t, c_t)
    grad = torch.autograd.grad(loss, x_t)[0]
    with torch.no_grad():
        eps_minus = model(x_t - 1e-4 * grad, x_cond, t_cur)
        abs_minus = x0_hat_abs_from_xt(x_t - 1e-4 * grad, eps_minus, t_idx, torch.from_numpy(degraded_rel_norm).to(DEVICE), start_t, rel_mean_t, rel_std_t, diffusion)
        loss_minus = weighted_observation_loss(abs_minus, y_t, c_t)
    passed = bool(loss_minus.item() < loss.item())
    print(f"[CHECK] norm-based likelihood sign loss before/after = {loss.item():.6f}/{loss_minus.item():.6f}; pass={passed}")
    if not passed:
        raise AssertionError("Norm-based likelihood sign check failed")
    return {"degradation": degradation, "loss_before": float(loss.item()), "loss_after": float(loss_minus.item()), "passed": passed}


def isfinite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def run_dps_single(
    degraded_single: np.ndarray,
    confidence_single: np.ndarray,
    model: ConditionalTemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    zeta: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    degraded = degraded_single[None, :, :].astype(np.float32)
    confidence = confidence_single[None, :].astype(np.float32)
    degraded_rel = to_rel(degraded)
    degraded_rel_norm = normalize_rel(degraded_rel, rel_mean, rel_std)
    x_cond = torch.from_numpy(degraded_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32)
    degraded_rel_norm_t = torch.from_numpy(degraded_rel_norm).to(DEVICE, dtype=torch.float32)
    y_t = torch.from_numpy(degraded).to(DEVICE, dtype=torch.float32)
    c_t = torch.from_numpy(confidence).to(DEVICE, dtype=torch.float32)
    start_t = y_t[:, 0, :]
    rel_mean_t = torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32)
    x_t = torch.randn((1, 2, 19), device=DEVICE, dtype=torch.float32)
    max_update_to_x_ratio = 0.0
    max_grad_norm = 0.0
    snapshots = {step: math.nan for step in SNAPSHOT_STEPS}
    first_nonfinite_step = None
    first_nonfinite_stage = "none"

    for t_idx in reversed(range(TIMESTEPS)):
        x_t = x_t.detach().requires_grad_(True)
        before_ok = isfinite_tensor(x_t)
        t_cur = torch.full((1,), t_idx, device=DEVICE, dtype=torch.long)
        eps_pred = model(x_t, x_cond, t_cur)
        abs_hat = x0_hat_abs_from_xt(x_t, eps_pred, t_idx, degraded_rel_norm_t, start_t, rel_mean_t, rel_std_t, diffusion)
        loss = weighted_observation_loss(abs_hat, y_t, c_t)
        loss_ok = bool(torch.isfinite(loss.detach()).cpu())
        if loss_ok:
            grad = torch.autograd.grad(loss, x_t)[0]
        else:
            grad = torch.full_like(x_t, float("nan"))
        grad_ok = isfinite_tensor(grad)
        if grad_ok:
            max_grad_norm = max(max_grad_norm, float(torch.linalg.vector_norm(grad.detach()).cpu()))
        with torch.no_grad():
            step = ddpm_step(x_t, eps_pred, t_idx, diffusion)
            step_ok = isfinite_tensor(step)
            update = float(zeta) * grad
            update_ok = isfinite_tensor(update)
            step_norm = float(torch.linalg.vector_norm(step.detach()).cpu()) if step_ok else math.nan
            update_norm = float(torch.linalg.vector_norm(update.detach()).cpu()) if update_ok else math.nan
            update_to_x = update_norm / (step_norm + EPS) if step_ok and update_ok else math.nan
            if np.isfinite(update_to_x):
                max_update_to_x_ratio = max(max_update_to_x_ratio, float(update_to_x))
            if t_idx in snapshots:
                snapshots[t_idx] = float(update_to_x) if np.isfinite(update_to_x) else math.nan
            x_next = step - update
            after_ok = isfinite_tensor(x_next)
        if not before_ok:
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "before_ddpm"
            break
        if not loss_ok:
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "likelihood_loss"
            break
        if not grad_ok:
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "grad"
            break
        if not step_ok:
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "after_ddpm"
            break
        if not update_ok:
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "update"
            break
        if not after_ok:
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "after_guidance"
            break
        x_t = x_next

    finite = first_nonfinite_step is None
    if finite:
        residual_hat_norm = x_t.detach().permute(0, 2, 1).cpu().numpy().astype(np.float32)
        final_rel_norm = degraded_rel_norm + residual_hat_norm
        final_rel = (final_rel_norm * rel_std[None, None, :] + rel_mean[None, None, :]).astype(np.float32)
        pred = reconstruct_from_rel(degraded[:, 0, :], final_rel)[0]
    else:
        pred = np.full((20, 2), np.nan, dtype=np.float32)
    diag = {
        "finite": finite,
        "first_nonfinite_step": first_nonfinite_step,
        "first_nonfinite_stage": first_nonfinite_stage,
        "max_grad_norm": max_grad_norm,
        "max_update_to_x_ratio": max_update_to_x_ratio,
    }
    for step in SNAPSHOT_STEPS:
        diag[f"update_to_x_ratio_t{step}"] = snapshots[step]
    return pred.astype(np.float32), diag


def run_dps_batch(
    degraded: np.ndarray,
    confidence: np.ndarray,
    model: ConditionalTemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    zeta: float,
    seed: int,
    degradation: str,
) -> tuple[np.ndarray, list[dict]]:
    preds = np.full_like(degraded, np.nan, dtype=np.float32)
    rows: list[dict] = []
    for traj_idx in range(degraded.shape[0]):
        pred, diag = run_dps_single(degraded[traj_idx], confidence[traj_idx], model, diffusion, rel_mean, rel_std, zeta, seed)
        preds[traj_idx] = pred
        row = {"degradation": degradation, "zeta": zeta, "seed": seed, "trajectory_id": traj_idx}
        row.update(diag)
        rows.append(row)
        if not diag["finite"]:
            print(
                "[NONFINITE] "
                f"cond={degradation} zeta={zeta} seed={seed} traj={traj_idx} "
                f"step={diag['first_nonfinite_step']} stage={diag['first_nonfinite_stage']}"
            )
    return preds, rows


def compute_motion_usage(pred: np.ndarray, noisy: np.ndarray, formal_e1: np.ndarray) -> float:
    dp = np.diff(pred, axis=1)
    dy = np.diff(noisy, axis=1)
    de1 = np.diff(formal_e1, axis=1)
    den = float(np.linalg.norm(de1 - dy, axis=-1).mean())
    num = float(np.linalg.norm(dp - dy, axis=-1).mean())
    return num / den if den > 0 else math.nan


def evaluate_methods(
    predictions: dict[tuple[float, int, str], np.ndarray],
    dps_sampling_df: pd.DataFrame,
    data: dict[str, dict],
    gt: np.ndarray,
    confs: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_rows: list[dict] = []
    full_rows: list[dict] = []
    bin_rows: list[dict] = []
    diag_rows: list[dict] = []

    for degradation in DEGRADATION_ORDER:
        noisy = data[degradation]["degraded"]
        formal = data[degradation]["formal_e1"]
        e2min = data[degradation]["e2_min_v2"]
        conf = confs[degradation]
        high = conf > 0.7
        low = conf < 0.3
        mid = (conf >= 0.3) & (conf <= 0.7)
        masks = {"high": high, "mid": mid, "low": low}
        baseline_arrays = {
            "noisy_input": noisy,
            "Formal_E1": formal,
            "E2_Min_V2_selected": e2min,
        }
        all_arrays_by_seed: dict[str, list[np.ndarray]] = {name: [] for name in baseline_arrays}
        for seed in SEEDS:
            for name, arr in baseline_arrays.items():
                all_arrays_by_seed[name].append(arr)
            for zeta in ZETAS:
                all_arrays_by_seed.setdefault(f"E2_DPS_zeta_{zeta}", []).append(predictions[(zeta, seed, degradation)])

        ref_ade_noisy, _, _ = per_traj_metrics(noisy, gt)
        ref_ade_formal, _, _ = per_traj_metrics(formal, gt)
        ref_ade_e2min, _, _ = per_traj_metrics(e2min, gt)

        for method, arrs in all_arrays_by_seed.items():
            method_zeta = float(method.replace("E2_DPS_zeta_", "")) if method.startswith("E2_DPS_zeta_") else math.nan
            method_sampling = (
                dps_sampling_df[(dps_sampling_df["degradation"] == degradation) & (dps_sampling_df["zeta"] == method_zeta)]
                if method.startswith("E2_DPS_zeta_")
                else pd.DataFrame()
            )
            sample_ade: list[float] = []
            sample_rmse: list[float] = []
            sample_smooth: list[float] = []
            sample_high: list[float] = []
            sample_mid: list[float] = []
            sample_low: list[float] = []
            sample_motion: list[float] = []
            sample_gap: list[float] = []
            sample_max_update: list[float] = []
            sample_t99: list[float] = []
            sample_t80: list[float] = []
            sample_t60: list[float] = []
            sample_t40: list[float] = []
            sample_t20: list[float] = []
            sample_t0: list[float] = []
            wins_formal: list[bool] = []
            wins_e2min: list[bool] = []
            for seed_index, arr in enumerate(arrs):
                seed = SEEDS[seed_index]
                ade, rmse, smooth = per_traj_metrics(arr, gt)
                err = frame_error(arr, gt)
                motion_usage = compute_motion_usage(arr, noisy, formal)
                for idx in range(arr.shape[0]):
                    h = high[idx]
                    m = mid[idx]
                    l = low[idx]
                    high_ade = float(err[idx][h].mean()) if np.any(h) else math.nan
                    mid_ade = float(err[idx][m].mean()) if np.any(m) else math.nan
                    low_ade = float(err[idx][l].mean()) if np.any(l) else math.nan
                    noisy_gap = float(ade[idx] - ref_ade_noisy[idx])
                    win_formal = bool(ade[idx] <= ref_ade_formal[idx])
                    win_e2min = bool(ade[idx] <= ref_ade_e2min[idx])
                    if method.startswith("E2_DPS_zeta_"):
                        sampling_match = method_sampling[
                            (method_sampling["seed"] == seed) & (method_sampling["trajectory_id"] == idx)
                        ]
                        if sampling_match.empty:
                            max_update = math.nan
                            t99 = t80 = t60 = t40 = t20 = t0 = math.nan
                        else:
                            sampling_row = sampling_match.iloc[0]
                            max_update = float(sampling_row["max_update_to_x_ratio"])
                            t99 = float(sampling_row["update_to_x_ratio_t99"])
                            t80 = float(sampling_row["update_to_x_ratio_t80"])
                            t60 = float(sampling_row["update_to_x_ratio_t60"])
                            t40 = float(sampling_row["update_to_x_ratio_t40"])
                            t20 = float(sampling_row["update_to_x_ratio_t20"])
                            t0 = float(sampling_row["update_to_x_ratio_t0"])
                    else:
                        max_update = t99 = t80 = t60 = t40 = t20 = t0 = math.nan
                    sample_ade.append(float(ade[idx]))
                    sample_rmse.append(float(rmse[idx]))
                    sample_smooth.append(float(smooth[idx]))
                    sample_high.append(high_ade)
                    sample_mid.append(mid_ade)
                    sample_low.append(low_ade)
                    sample_motion.append(motion_usage)
                    sample_gap.append(noisy_gap)
                    sample_max_update.append(max_update)
                    sample_t99.append(t99)
                    sample_t80.append(t80)
                    sample_t60.append(t60)
                    sample_t40.append(t40)
                    sample_t20.append(t20)
                    sample_t0.append(t0)
                    wins_formal.append(win_formal)
                    wins_e2min.append(win_e2min)
                    per_rows.append(
                        {
                            "degradation": degradation,
                            "seed": seed,
                            "trajectory_id": idx,
                            "method": method,
                            "ADE": float(ade[idx]),
                            "RMSE": float(rmse[idx]),
                            "acceleration_RMS": float(smooth[idx]),
                            "ADE_high": high_ade,
                            "ADE_mid": mid_ade,
                            "ADE_low": low_ade,
                            "motion_usage_ratio": motion_usage,
                            "noisy_reversion_gap": noisy_gap,
                            "max_update_to_x_ratio": max_update,
                            "update_to_x_ratio_t99": t99,
                            "update_to_x_ratio_t80": t80,
                            "update_to_x_ratio_t60": t60,
                            "update_to_x_ratio_t40": t40,
                            "update_to_x_ratio_t20": t20,
                            "update_to_x_ratio_t0": t0,
                            "win_vs_Formal_E1": win_formal,
                            "win_vs_E2_Min_V2": win_e2min,
                        }
                    )

            ade_stats = stats(np.array(sample_ade))
            full_rows.append(
                {
                    "degradation": degradation,
                    "method": method,
                    "N_samples": len(sample_ade),
                    "ADE_mean": ade_stats["mean"],
                    "ADE_std": ade_stats["std"],
                    "ADE_median": ade_stats["median"],
                    "ADE_min": ade_stats["min"],
                    "ADE_max": ade_stats["max"],
                    "ADE_p25": ade_stats["p25"],
                    "ADE_p75": ade_stats["p75"],
                    "RMSE_mean": float(np.nanmean(sample_rmse)),
                    "acceleration_RMS_mean": float(np.nanmean(sample_smooth)),
                    "ADE_high_mean": float(np.nanmean(sample_high)),
                    "ADE_mid_mean": float(np.nanmean(sample_mid)),
                    "ADE_low_mean": float(np.nanmean(sample_low)),
                    "motion_usage_ratio": float(np.nanmean(sample_motion)),
                    "noisy_reversion_gap": float(np.nanmean(sample_gap)),
                    "max_update_to_x_ratio": float(np.nanmax(sample_max_update)) if np.any(np.isfinite(sample_max_update)) else math.nan,
                    "late_update_to_x_ratio_max": float(np.nanmax([np.nanmax(sample_t40), np.nanmax(sample_t20), np.nanmax(sample_t0)]))
                    if any(np.any(np.isfinite(vals)) for vals in [sample_t40, sample_t20, sample_t0])
                    else math.nan,
                    "nonfinite_count": int(method_sampling["finite"].eq(False).sum()) if method.startswith("E2_DPS_zeta_") else 0,
                    "win_rate_vs_Formal_E1": float(np.mean(wins_formal)),
                    "win_rate_vs_E2_Min_V2": float(np.mean(wins_e2min)),
                }
            )
            method_err_all = []
            method_conf_all = []
            for arr in arrs:
                method_err_all.append(frame_error(arr, gt))
                method_conf_all.append(conf)
            err_stack = np.concatenate(method_err_all, axis=0)
            conf_stack = np.concatenate(method_conf_all, axis=0)
            for bin_name, mask in {
                "high": conf_stack > 0.7,
                "mid": (conf_stack >= 0.3) & (conf_stack <= 0.7),
                "low": conf_stack < 0.3,
            }.items():
                vals = err_stack[mask]
                s = stats(vals)
                bin_rows.append(
                    {
                        "degradation": degradation,
                        "method": method,
                        "confidence_bin": bin_name,
                        "N_frames": int(mask.sum()),
                        "ADE_mean": s["mean"],
                        "ADE_std": s["std"],
                        "ADE_median": s["median"],
                        "ADE_min": s["min"],
                        "ADE_max": s["max"],
                        "ADE_p25": s["p25"],
                        "ADE_p75": s["p75"],
                    }
                )
            if method.startswith("E2_DPS_zeta_"):
                noisy_low = np.array([row["ADE_low"] for row in per_rows if row["degradation"] == degradation and row["method"] == "noisy_input"], dtype=np.float64)
                method_low = np.array(sample_low, dtype=np.float64)
                low_delta = method_low - noisy_low[: len(method_low)]
                high_ratio = float(np.nanmean(sample_high) / np.nanmean([row["ADE_high"] for row in per_rows if row["degradation"] == degradation and row["method"] == "noisy_input"]))
                diag_rows.append(
                    {
                        "degradation": degradation,
                        "zeta": float(method.replace("E2_DPS_zeta_", "")),
                        "method": method,
                        "ADE_mean": ade_stats["mean"],
                        "ADE_high_mean": float(np.nanmean(sample_high)),
                        "ADE_high_noisy_mean": float(np.nanmean([row["ADE_high"] for row in per_rows if row["degradation"] == degradation and row["method"] == "noisy_input"])),
                        "high_conf_no_harm_ratio": high_ratio,
                        "ADE_low_mean": float(np.nanmean(sample_low)),
                        "low_conf_wilcoxon_p_vs_noisy": safe_wilcoxon(low_delta),
                        "motion_usage_ratio": float(np.nanmean(sample_motion)),
                        "noisy_reversion_gap": float(np.nanmean(sample_gap)),
                        "acceleration_RMS_mean": float(np.nanmean(sample_smooth)),
                        "max_update_to_x_ratio": float(np.nanmax(sample_max_update)) if np.any(np.isfinite(sample_max_update)) else math.nan,
                        "late_update_to_x_ratio_max": float(np.nanmax([np.nanmax(sample_t40), np.nanmax(sample_t20), np.nanmax(sample_t0)]))
                        if any(np.any(np.isfinite(vals)) for vals in [sample_t40, sample_t20, sample_t0])
                        else math.nan,
                        "nonfinite_count": int(method_sampling["finite"].eq(False).sum()),
                    }
                )
    return pd.DataFrame(full_rows), pd.DataFrame(bin_rows), pd.DataFrame(per_rows), pd.DataFrame(diag_rows)


def build_pass_fail(full_df: pd.DataFrame, diag_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zeta in ZETAS:
        drift = diag_df[(diag_df["degradation"] == "drift_medium") & (diag_df["zeta"] == zeta)].iloc[0]
        burst = diag_df[(diag_df["degradation"] == "burst_medium") & (diag_df["zeta"] == zeta)].iloc[0]
        bias_e2 = full_df[(full_df["degradation"] == "bias_medium") & (full_df["method"] == f"E2_DPS_zeta_{zeta}")].iloc[0]
        bias_e1 = full_df[(full_df["degradation"] == "bias_medium") & (full_df["method"] == "Formal_E1")].iloc[0]
        drift_valid = bool(np.isfinite(float(drift["ADE_mean"])) and np.isfinite(float(drift["high_conf_no_harm_ratio"])))
        burst_valid = bool(np.isfinite(float(burst["ADE_mean"])) and np.isfinite(float(burst["high_conf_no_harm_ratio"])))
        bias_valid = bool(np.isfinite(float(bias_e2["ADE_mean"])) and np.isfinite(float(bias_e1["ADE_mean"])))
        red = bool(bias_valid and float(bias_e2["ADE_mean"]) / float(bias_e1["ADE_mean"]) < 0.95)
        h1_pass = bool(drift_valid and float(drift["high_conf_no_harm_ratio"]) <= 1.05)
        h2_pass = bool(burst_valid and float(burst["high_conf_no_harm_ratio"]) <= 1.05)
        rows.append(
            {
                "zeta": zeta,
                "H1_drift_high_conf_no_harm_ratio": float(drift["high_conf_no_harm_ratio"]),
                "H1_valid_finite": drift_valid,
                "H1_pass": h1_pass,
                "H2_burst_high_conf_no_harm_ratio": float(burst["high_conf_no_harm_ratio"]),
                "H2_valid_finite": burst_valid,
                "H2_pass": h2_pass,
                "bias_ADE_E2DPS": float(bias_e2["ADE_mean"]),
                "bias_ADE_E1": float(bias_e1["ADE_mean"]),
                "bias_red_flag": red,
                "any_core_signal": bool(h1_pass or h2_pass),
            }
        )
    return pd.DataFrame(rows)


def six_condition_mean_ade(full_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zeta in ZETAS:
        method = f"E2_DPS_zeta_{zeta}"
        vals = full_df[full_df["method"] == method].set_index("degradation").loc[DEGRADATION_ORDER, "ADE_mean"].to_numpy()
        rows.append({"zeta": zeta, "six_condition_mean_ADE": float(vals.mean())})
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    for col in table.columns:
        table[col] = table[col].map(
            lambda value: f"{float(value):.6f}"
            if isinstance(value, (float, np.floating)) and np.isfinite(value)
            else ("NaN" if isinstance(value, (float, np.floating)) else str(value))
        )
    lines = [
        "| " + " | ".join(str(col) for col in table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for row in table.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_audit_summary(audit_rows: list[dict], sign_row: dict, count_rows: list[dict], sigma_rows: list[dict]) -> None:
    lines = [
        "# E2-DPS Stage 8A Pilot Audit Summary",
        "",
        f"Pre-registration: {PRE_REG_PATH}",
        f"Amendment 001: {AMENDMENT_001_PATH}",
        f"Amendment 002: {AMENDMENT_002_PATH}",
        f"Confidence cache: {CONF_CACHE_DIR}",
        f"Device: {DEVICE}",
        "",
        "All audit checks passed before pilot sampling.",
        "",
        "Method: Amendment 002 canonical norm-based DPS guidance with heteroscedastic confidence weighting.",
        "",
        "## A Operator Round-Trip",
    ]
    lines.extend(f"- {row['degradation']}: max_err={row['roundtrip_max_err']:.2e}" for row in audit_rows)
    lines.extend(["", "## Confidence Cache Counts"])
    lines.extend(
        f"- {row['degradation']}: full high/mid/low={row['full_high']}/{row['full_mid']}/{row['full_low']}; "
        f"pilot high/mid/low={row['pilot_high']}/{row['pilot_mid']}/{row['pilot_low']}"
        for row in count_rows
    )
    lines.extend(["", "## Sigma Obs Squared"])
    lines.extend(
        f"- {row['degradation']}: min/mean/max={row['sigma_obs2_min']:.6f}/{row['sigma_obs2_mean']:.6f}/{row['sigma_obs2_max']:.6f}"
        for row in sigma_rows
    )
    lines.extend(
        [
            "",
            "## Norm-Based Likelihood Sign Check",
            f"- {sign_row['degradation']}: loss before={sign_row['loss_before']:.6f}, after={sign_row['loss_after']:.6f}, passed={sign_row['passed']}",
            "",
            "Ground truth val_trajs.npy[:50] is used only for metrics, not guidance, anchor, or sigma_obs.",
        ]
    )
    write_text_file(AUDIT_SUMMARY_PATH, "\n".join(lines) + "\n")


def make_figures(predictions: dict[tuple[float, int, str], np.ndarray], data: dict[str, dict], gt: np.ndarray, confs: dict[str, np.ndarray]) -> list[dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    requests = [
        ("drift", "drift_medium"),
        ("burst", "burst_medium"),
        ("bias_negative_control", "bias_medium"),
        ("jump_diagnostic", "jump_medium"),
    ]
    for label, degradation in requests:
        noisy = data[degradation]["degraded"]
        formal = data[degradation]["formal_e1"]
        e2min = data[degradation]["e2_min_v2"]
        conf = confs[degradation]
        err_formal = frame_error(formal, gt).mean(axis=1)
        idx = int(np.argmax(err_formal))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        ax = axes[0]
        ax.plot(gt[idx, :, 0], gt[idx, :, 1], "k-o", ms=3, lw=1.3, label="clean")
        ax.plot(noisy[idx, :, 0], noisy[idx, :, 1], "C1-o", ms=3, lw=1.0, label="noisy")
        ax.plot(formal[idx, :, 0], formal[idx, :, 1], "C0-o", ms=3, lw=1.0, label="Formal E1")
        ax.plot(e2min[idx, :, 0], e2min[idx, :, 1], "C2-o", ms=3, lw=1.0, label="E2-Min V2")
        colors = {0.3: "C3", 1.0: "C4", 3.0: "C5"}
        for zeta in ZETAS:
            pred = predictions[(zeta, 42, degradation)]
            ax.plot(pred[idx, :, 0], pred[idx, :, 1], marker="o", ms=3, lw=1.0, color=colors[zeta], label=f"DPS ζ={zeta}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_title(f"{label}: {degradation} traj {idx}")
        ax.legend(fontsize=7)
        t = np.arange(20)
        axes[1].plot(t, conf[idx], "k-o", ms=3, lw=1.2)
        axes[1].axhline(0.7, color="0.35", ls="--")
        axes[1].axhline(0.3, color="0.35", ls=":")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("confidence")
        axes[1].grid(alpha=0.25)
        axes[2].plot(t, frame_error(noisy, gt)[idx], "C1-o", ms=3, lw=1.0, label="noisy")
        axes[2].plot(t, frame_error(formal, gt)[idx], "C0-o", ms=3, lw=1.0, label="Formal E1")
        axes[2].plot(t, frame_error(e2min, gt)[idx], "C2-o", ms=3, lw=1.0, label="E2-Min V2")
        for zeta in ZETAS:
            pred = predictions[(zeta, 42, degradation)]
            axes[2].plot(t, frame_error(pred, gt)[idx], marker="o", ms=3, lw=1.0, color=colors[zeta], label=f"DPS ζ={zeta}")
        axes[2].set_title("per-frame error")
        axes[2].grid(alpha=0.25)
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        path = FIG_DIR / f"{label}_{degradation}_traj{idx}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        rows.append({"figure": label, "degradation": degradation, "trajectory_id": idx, "path": str(path)})
    return rows


def write_summary(full_df: pd.DataFrame, pass_df: pd.DataFrame, diag_df: pd.DataFrame, figure_rows: list[dict]) -> None:
    mean_df = six_condition_mean_ade(full_df)
    instability = full_df[full_df["method"].str.startswith("E2_DPS") & ~np.isfinite(full_df["ADE_mean"])]
    nonfinite = full_df[full_df["method"].str.startswith("E2_DPS") & (full_df["nonfinite_count"] > 0)]
    lines = [
        "# E2-DPS Stage 8A Pilot Summary",
        "",
        "This is Stage 8A only. Stage 8B was not started.",
        "",
        "Guidance: Amendment 002 canonical norm-based DPS guidance. Overshoot caution remains active because the small re-audit was Case B.",
        "",
        "## Six-Condition Mean ADE",
        markdown_table(mean_df),
        "",
        "## H1/H2 Decision Gate",
        markdown_table(pass_df),
        "",
        "## Numerical Stability",
        "DPS rows with non-finite ADE:",
        markdown_table(instability[["degradation", "method", "ADE_mean", "ADE_high_mean", "ADE_low_mean", "motion_usage_ratio", "acceleration_RMS_mean"]]),
        "",
        "DPS rows with non-finite sampling count:",
        markdown_table(nonfinite[["degradation", "method", "nonfinite_count", "max_update_to_x_ratio", "late_update_to_x_ratio_max"]]),
        "",
        "## Diagnostics",
        markdown_table(diag_df),
        "",
        "## Figures",
    ]
    lines.extend(f"- {row['figure']}: {row['path']}" for row in figure_rows)
    write_text_file(SUMMARY_PATH, "\n".join(lines) + "\n")


def print_decision_gate(pass_df: pd.DataFrame) -> None:
    print("================================================================")
    print("E2-DPS Stage 8A Pilot — Decision Gate")
    print("================================================================")
    print("")
    print("H1 (drift high-conf no-harm) by ζ:")
    for _, row in pass_df.iterrows():
        status = "PASS" if row["H1_pass"] else "FAIL"
        print(f"  ζ={row['zeta']}:  {status}  (ratio = {row['H1_drift_high_conf_no_harm_ratio']:.3f}, threshold = 1.05)")
    print("")
    print("H2 (burst high-conf no-harm) by ζ:")
    for _, row in pass_df.iterrows():
        status = "PASS" if row["H2_pass"] else "FAIL"
        print(f"  ζ={row['zeta']}:  {status}  (ratio = {row['H2_burst_high_conf_no_harm_ratio']:.3f}, threshold = 1.05)")
    print("")
    red = bool(pass_df["bias_red_flag"].any())
    print(f"NC1 bias negative control red flag: {'TRIGGERED' if red else 'OK'}")
    if red:
        print("  NC1 §4 audit:")
        print("  (a) A operator absolute-space leakage: check audit_summary")
        print("  (b) oracle confidence c_t absolute-position information: check confidence cache validation")
        print("  (c) σ_obs,t² parameterization absolute-position bias: check sigma_obs^2 audit")
        print("  (d) ground truth in guidance path: not used in guidance by implementation")
    print("")
    any_pass = bool(pass_df["any_core_signal"].any())
    print("DECISION:")
    if any_pass:
        print("  Case A: at least one (ζ, H_i) PASS")
        print("    → RECOMMEND PROCEEDING TO STAGE 8B (full N=200, 5 seeds)")
        print("      Stage 8B was not started automatically.")
    else:
        print("  Case B: no PASS anywhere")
        print("    → PAUSE FOR AUDIT (pre-registration §8):")
        print("      (a) A operator round-trip")
        print("      (b) σ_obs parameterization")
        print("      (c) guidance sign and scale")
        print("      (d) c_t bit-identical to validated cache")
        print("")
        print("    If implementation issue found:")
        print("        fix and rerun pilot with same ζ")
        print("")
        print("    If no implementation issue:")
        print("        proceed to Stage 8B as FULL NEGATIVE-RESULT evaluation")
        print("        do not expand ζ")
        print("        do not modify likelihood")
    print("================================================================")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, message in [
        (PRE_REG_PATH, "E2-DPS pre-registration file not found"),
        (AMENDMENT_001_PATH, "E2-DPS Amendment 001 file not found"),
        (AMENDMENT_002_PATH, "E2-DPS Amendment 002 file not found"),
        (CONF_VALIDATION_PATH, "Missing validated confidence cache summary"),
        (PROTOCOL_PATH, "Missing Formal E1 protocol summary"),
        (E1_BIN_PATH, "Missing Formal E1 confidence-bin metrics"),
        (E1_PASS_FAIL_PATH, "Missing Formal E1 pass/fail metrics"),
        (E2_MIN_TRAJ_PATH, "Missing E2-Min trajectory outputs"),
        (VAL_PATH, "Missing validation trajectories"),
        (NORM_PATH, "Missing relative normalization parameters"),
        (CKPT_PATH, "Missing Stage 3 prior checkpoint"),
    ]:
        require_file(path, message)

    validation_df, confs = load_confidence_cache()
    protocol, data, gt = load_inputs()

    audit_rows = []
    for degradation in DEGRADATION_ORDER:
        max_err = roundtrip_check(data[degradation]["degraded"][0])
        print(f"[CHECK] A operator round-trip max err = {max_err:.2e} ({degradation})")
        if max_err >= 1e-6:
            write_audit_failed("A operator round-trip failed", [f"{degradation}: max_err={max_err:.8e}"])
            raise SystemExit(2)
        audit_rows.append({"degradation": degradation, "roundtrip_max_err": max_err})

    count_rows = validate_confidence_counts(validation_df, confs)
    sigma_rows = sigma_obs_stats(confs)
    model, diffusion, rel_mean, rel_std = load_model()
    sign_row = likelihood_sign_check(model, diffusion, data, confs, rel_mean, rel_std)
    write_audit_summary(audit_rows, sign_row, count_rows, sigma_rows)

    predictions: dict[tuple[float, int, str], np.ndarray] = {}
    sampling_rows: list[dict] = []
    for zeta in ZETAS:
        for seed in SEEDS:
            for degradation in DEGRADATION_ORDER:
                pred, rows = run_dps_batch(
                    data[degradation]["degraded"],
                    confs[degradation],
                    model,
                    diffusion,
                    rel_mean,
                    rel_std,
                    zeta,
                    seed,
                    degradation,
                )
                predictions[(zeta, seed, degradation)] = pred
                sampling_rows.extend(rows)
                print(f"[DONE] ζ={zeta} seed={seed} cond={degradation}")

    sampling_df = pd.DataFrame(sampling_rows)
    full_df, bin_df, per_df, diag_df = evaluate_methods(predictions, sampling_df, data, gt, confs)
    pass_df = build_pass_fail(full_df, diag_df)
    figure_rows = make_figures(predictions, data, gt, confs)

    full_df.to_csv(FULL_METRICS_PATH, index=False)
    print_file(FULL_METRICS_PATH)
    bin_df.to_csv(BIN_METRICS_PATH, index=False)
    print_file(BIN_METRICS_PATH)
    per_df.to_csv(PER_TRAJ_PATH, index=False)
    print_file(PER_TRAJ_PATH)
    pass_df.to_csv(PASS_FAIL_PATH, index=False)
    print_file(PASS_FAIL_PATH)
    diag_df.to_csv(DIAGNOSTICS_PATH, index=False)
    print_file(DIAGNOSTICS_PATH)
    sampling_path = OUT_DIR / "e2_dps_pilot_sampling_diagnostics.csv"
    sampling_df.to_csv(sampling_path, index=False)
    print_file(sampling_path)
    pd.DataFrame(figure_rows).to_csv(OUT_DIR / "e2_dps_pilot_figures.csv", index=False)
    print_file(OUT_DIR / "e2_dps_pilot_figures.csv")
    write_summary(full_df, pass_df, diag_df, figure_rows)
    print_decision_gate(pass_df)


if __name__ == "__main__":
    main()
