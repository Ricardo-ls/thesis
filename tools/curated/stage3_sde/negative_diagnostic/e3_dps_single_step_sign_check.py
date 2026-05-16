from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3_dps_single_step_sign_check_mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D


CONDITION = "drift_medium"
TRAJECTORY_ID = 0
SEED = 42
T_START = 5
TIMESTEPS = 100
SIGMA0 = 0.05
KAPPA = 1.0
EPS_VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
EPS_DENOM = 1e-12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARRAY_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_holdout1000_confidence_aware_sdedit" / "arrays"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CKPT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_dps_single_step_sign_check"
CSV_PATH = OUT_DIR / "e3_dps_single_step_sign_check.csv"
SUMMARY_PATH = OUT_DIR / "e3_dps_single_step_sign_check_summary.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_inputs() -> None:
    required = [
        ARRAY_DIR / f"{CONDITION}_clean.npy",
        ARRAY_DIR / f"{CONDITION}_degraded.npy",
        ARRAY_DIR / f"{CONDITION}_confidence.npy",
        ARRAY_DIR / f"{CONDITION}_fused_t1_tau07_gamma2.npy",
        NORM_PATH,
        CKPT_PATH,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))


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


def to_rel(traj: np.ndarray) -> np.ndarray:
    return (traj[:, 1:, :] - traj[:, :-1, :]).astype(np.float32)


def normalize_rel(rel: np.ndarray, rel_mean: np.ndarray, rel_std: np.ndarray) -> np.ndarray:
    return ((rel - rel_mean[None, None, :]) / rel_std[None, None, :]).astype(np.float32)


def acceleration_rms_single(traj: np.ndarray) -> float:
    acc = traj[2:, :] - 2.0 * traj[1:-1, :] + traj[:-2, :]
    return float(np.sqrt(np.mean(np.sum(acc**2, axis=-1))))


def ade_single(pred: np.ndarray, clean: np.ndarray) -> float:
    return float(np.linalg.norm(pred - clean, axis=-1).mean())


def x0_hat_abs(
    x_t: torch.Tensor,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    anchor_t: torch.Tensor,
    rel_mean_t: torch.Tensor,
    rel_std_t: torch.Tensor,
) -> torch.Tensor:
    t_cur = torch.full((x_t.shape[0],), T_START, device=DEVICE, dtype=torch.long)
    eps_pred = model(x_t, t_cur)
    alpha_bar = diffusion.alpha_bars[T_START]
    x0_norm = (x_t - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
    rel_norm = x0_norm.permute(0, 2, 1)
    rel = rel_norm * rel_std_t.view(1, 1, 2) + rel_mean_t.view(1, 1, 2)
    cumsum = torch.cumsum(rel, dim=1)
    return torch.cat([anchor_t[:, None, :], anchor_t[:, None, :] + cumsum], dim=1)


def loss_and_abs(
    x_t: torch.Tensor,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    y_t: torch.Tensor,
    conf_t: torch.Tensor,
    anchor_t: torch.Tensor,
    rel_mean_t: torch.Tensor,
    rel_std_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    abs_hat = x0_hat_abs(x_t, model, diffusion, anchor_t, rel_mean_t, rel_std_t)
    sigma2 = SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t))
    sigma = torch.sqrt(sigma2)
    weighted_residual = (abs_hat - y_t) / sigma[..., None]
    loss = torch.linalg.vector_norm(weighted_residual)
    return loss, abs_hat


def evaluate_candidate(
    x_candidate: torch.Tensor,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    y_t: torch.Tensor,
    conf_t: torch.Tensor,
    anchor_t: torch.Tensor,
    rel_mean_t: torch.Tensor,
    rel_std_t: torch.Tensor,
    clean_np: np.ndarray,
) -> tuple[float, float, float, bool]:
    with torch.no_grad():
        loss, abs_hat = loss_and_abs(
            x_candidate.detach(),
            model,
            diffusion,
            y_t,
            conf_t,
            anchor_t,
            rel_mean_t,
            rel_std_t,
        )
    abs_np = abs_hat.detach().cpu().numpy().astype(np.float32)[0]
    finite = bool(torch.isfinite(loss).cpu()) and bool(np.isfinite(abs_np).all())
    return float(loss.detach().cpu()), ade_single(abs_np, clean_np), acceleration_rms_single(abs_np), finite


def classify(rows: pd.DataFrame) -> tuple[str, str]:
    small = rows.sort_values("eps").iloc[0]
    any_minus_down = bool((rows["delta_L_minus"] < 0).any())
    any_plus_down = bool((rows["delta_L_plus"] < 0).any())
    small_minus_down = bool(small["delta_L_minus"] < 0)
    small_plus_down = bool(small["delta_L_plus"] < 0)
    large_minus_up = bool(rows.sort_values("eps").iloc[-1]["delta_L_minus"] > 0)
    all_finite = bool(rows["finite_minus"].all() and rows["finite_plus"].all())
    tol = max(abs(float(small["L_zero"])) * 1e-10, 1e-12)
    flat = bool((rows["delta_L_minus"].abs() <= tol).all() and (rows["delta_L_plus"].abs() <= tol).all())

    if not all_finite:
        return "Case 5: numerical instability", "Non-finite likelihood or trajectory appeared in the single-step diagnostic."
    if small_minus_down and large_minus_up:
        return "Case 1: sign correct, step too large", "Minus-gradient direction lowers likelihood at small eps, but large eps overshoots."
    if small_minus_down or (any_minus_down and not any_plus_down):
        return "Case 1: sign correct", "Minus-gradient direction lowers likelihood; inspect stable eps range before any DPS rerun."
    if small_plus_down and not small_minus_down:
        return "Case 2: sign reversed", "Plus-gradient direction lowers likelihood while minus-gradient does not."
    if (rows["delta_L_minus"] > 0).all() and (rows["delta_L_plus"] > 0).all():
        return "Case 3: variable interface mismatch", "Both directions increase likelihood even at tiny eps."
    if flat:
        return "Case 4: flat / ineffective gradient", "Loss is effectively unchanged in both directions."
    if any_minus_down and any_plus_down:
        return "Mixed: local curvature / scale sensitive", "Both signs lower likelihood for some eps; inspect variable interface and scale."
    return "Inconclusive", "The single-step trend does not match a clean diagnostic case."


def write_summary(df: pd.DataFrame, case_name: str, case_note: str) -> None:
    stable_minus = df[df["delta_L_minus"] < 0]["eps"].tolist()
    stable_plus = df[df["delta_L_plus"] < 0]["eps"].tolist()
    l_zero = float(df["L_zero"].iloc[0])
    grad_norm = float(df["grad_norm"].iloc[0])
    x_norm = float(df["x_norm"].iloc[0])
    min_row = df.sort_values("eps").iloc[0]
    max_stable_minus = max(stable_minus) if stable_minus else math.nan
    zeta_03_ratio = 0.3 * grad_norm / (x_norm + EPS_DENOM)
    lines = [
        "# E3 DPS Single-Step Sign Check",
        "",
        "This diagnostic uses one case only: `drift_medium`, trajectory `0`, initialized from frozen E3 `fused_t1_tau07_gamma2`, q_sampled to `t=5` with seed `42`.",
        "",
        f"- L_zero: `{l_zero:.9f}`",
        f"- grad_norm: `{grad_norm:.9f}`",
        f"- x_norm: `{x_norm:.9f}`",
        f"- zeta=0.3 equivalent update_to_x_ratio at this x_t: `{zeta_03_ratio:.9f}`",
        "",
        "## Direction Trend",
        "",
        "| eps | L_minus | delta_L_minus | L_plus | delta_L_plus | update_to_x_ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in df.sort_values("eps").iterrows():
        lines.append(
            f"| {row['eps']:.0e} | {row['L_minus']:.9f} | {row['delta_L_minus']:.9f} | "
            f"{row['L_plus']:.9f} | {row['delta_L_plus']:.9f} | {row['update_to_x_ratio_minus']:.9f} |"
        )
    lines += [
        "",
        "## Diagnostic Answers",
        "",
        f"- Which direction lowers likelihood: `minus` eps values `{stable_minus}`, `plus` eps values `{stable_plus}`.",
        f"- At minimum eps={min_row['eps']:.0e}: minus delta `{min_row['delta_L_minus']:.9f}`, plus delta `{min_row['delta_L_plus']:.9f}`.",
        f"- Diagnostic case: `{case_name}`.",
        f"- Note: {case_note}",
        f"- Stable minus eps range max: `{max_stable_minus}`.",
        f"- Is zeta=0.3 clearly too large for this single-step scale: `{bool(zeta_03_ratio > max(df['update_to_x_ratio_minus'].max(), 1.0))}`.",
        "",
        "## Next Step Recommendation",
        "",
    ]
    if case_name.startswith("Case 2"):
        lines.append("A. Fix sign before any further DPS audit.")
    elif case_name.startswith("Case 1"):
        lines.append("B. Do not treat the previous DPS failure as final method failure. The sign is locally correct, but zeta=0.3 is many orders larger than the stable single-step eps range; any future audit would need a pre-registered log-scale zeta sweep on a new hold-out.")
    elif case_name.startswith("Case 3"):
        lines.append("C. Check variable interface before running DPS again.")
    else:
        lines.append("D. Do not run formal E4 from the current DPS form without another interface/scale audit.")
    lines += [
        "",
        "## Output",
        "",
        f"- `{rel(CSV_PATH)}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(SUMMARY_PATH)} written")


def main() -> None:
    require_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, diffusion, rel_mean, rel_std = load_model()

    clean = np.load(ARRAY_DIR / f"{CONDITION}_clean.npy").astype(np.float32)[TRAJECTORY_ID]
    degraded = np.load(ARRAY_DIR / f"{CONDITION}_degraded.npy").astype(np.float32)[TRAJECTORY_ID : TRAJECTORY_ID + 1]
    confidence = np.load(ARRAY_DIR / f"{CONDITION}_confidence.npy").astype(np.float32)[TRAJECTORY_ID : TRAJECTORY_ID + 1]
    fused = np.load(ARRAY_DIR / f"{CONDITION}_fused_t1_tau07_gamma2.npy").astype(np.float32)[TRAJECTORY_ID : TRAJECTORY_ID + 1]

    init_rel = to_rel(fused)
    init_rel_norm = normalize_rel(init_rel, rel_mean, rel_std)
    x0 = torch.from_numpy(init_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t = torch.full((1,), T_START, device=DEVICE, dtype=torch.long)
    x_t, _ = diffusion.q_sample(x0, t)

    y_t = torch.from_numpy(degraded).to(DEVICE, dtype=torch.float32)
    conf_t = torch.from_numpy(confidence).to(DEVICE, dtype=torch.float32)
    anchor_t = torch.from_numpy(degraded[:, 0, :]).to(DEVICE, dtype=torch.float32)
    rel_mean_t = torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32)

    x_t = x_t.detach().requires_grad_(True)
    loss_zero, abs_zero = loss_and_abs(x_t, model, diffusion, y_t, conf_t, anchor_t, rel_mean_t, rel_std_t)
    grad = torch.autograd.grad(loss_zero, x_t, retain_graph=False)[0]
    grad_norm = float(torch.linalg.vector_norm(grad.detach()).cpu())
    x_norm = float(torch.linalg.vector_norm(x_t.detach()).cpu())
    zero_np = abs_zero.detach().cpu().numpy().astype(np.float32)[0]
    ade_zero = ade_single(zero_np, clean)
    accel_zero = acceleration_rms_single(zero_np)

    rows = []
    for eps in EPS_VALUES:
        x_minus = (x_t.detach() - float(eps) * grad.detach()).detach()
        x_plus = (x_t.detach() + float(eps) * grad.detach()).detach()
        l_minus, ade_minus, accel_minus, finite_minus = evaluate_candidate(
            x_minus, model, diffusion, y_t, conf_t, anchor_t, rel_mean_t, rel_std_t, clean
        )
        l_plus, ade_plus, accel_plus, finite_plus = evaluate_candidate(
            x_plus, model, diffusion, y_t, conf_t, anchor_t, rel_mean_t, rel_std_t, clean
        )
        update_norm = float(eps) * grad_norm
        rows.append(
            {
                "condition": CONDITION,
                "trajectory_id": TRAJECTORY_ID,
                "eps": eps,
                "L_zero": float(loss_zero.detach().cpu()),
                "L_minus": l_minus,
                "L_plus": l_plus,
                "delta_L_minus": l_minus - float(loss_zero.detach().cpu()),
                "delta_L_plus": l_plus - float(loss_zero.detach().cpu()),
                "ADE_zero": ade_zero,
                "ADE_minus": ade_minus,
                "ADE_plus": ade_plus,
                "accel_zero": accel_zero,
                "accel_minus": accel_minus,
                "accel_plus": accel_plus,
                "grad_norm": grad_norm,
                "x_norm": x_norm,
                "update_norm_minus": update_norm,
                "update_norm_plus": update_norm,
                "update_to_x_ratio_minus": update_norm / (x_norm + EPS_DENOM),
                "update_to_x_ratio_plus": update_norm / (x_norm + EPS_DENOM),
                "finite_minus": finite_minus,
                "finite_plus": finite_plus,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"[FILE] {rel(CSV_PATH)} written")
    case_name, case_note = classify(df)
    write_summary(df, case_name, case_note)

    print("E3_DPS_SINGLE_STEP_SIGN_CHECK_COMPLETE")
    print(f"L_zero={float(loss_zero.detach().cpu()):.9f}")
    print(df[["eps", "L_minus", "L_plus", "delta_L_minus", "delta_L_plus", "update_to_x_ratio_minus"]].to_string(index=False))
    print(f"diagnostic_case={case_name}")
    print(f"output_dir={rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
