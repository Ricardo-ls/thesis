from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3_dps_extended_sign_check_mpl")

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
TRAJECTORY_ID = 0
SEED = 42
T_STEPS = [1, 3, 5]
TIMESTEPS = 100
SIGMA0 = 0.05
KAPPA = 1.0
EPS_VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
SMALL_EPS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
EPS_DENOM = 1e-12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARRAY_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_holdout1000_confidence_aware_sdedit" / "arrays"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CKPT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e3_dps_extended_sign_check"
CSV_PATH = OUT_DIR / "e3_dps_extended_sign_check.csv"
CASE_SUMMARY_PATH = OUT_DIR / "e3_dps_extended_sign_check_case_summary.csv"
SUMMARY_PATH = OUT_DIR / "e3_dps_extended_sign_check_summary.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_inputs() -> None:
    required = [NORM_PATH, CKPT_PATH]
    for condition in CONDITIONS:
        required.extend(
            [
                ARRAY_DIR / f"{condition}_clean.npy",
                ARRAY_DIR / f"{condition}_degraded.npy",
                ARRAY_DIR / f"{condition}_confidence.npy",
                ARRAY_DIR / f"{condition}_fused_t1_tau07_gamma2.npy",
            ]
        )
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
    t_step: int,
    anchor_t: torch.Tensor,
    rel_mean_t: torch.Tensor,
    rel_std_t: torch.Tensor,
) -> torch.Tensor:
    t_cur = torch.full((x_t.shape[0],), t_step, device=DEVICE, dtype=torch.long)
    eps_pred = model(x_t, t_cur)
    alpha_bar = diffusion.alpha_bars[t_step]
    x0_norm = (x_t - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
    rel_norm = x0_norm.permute(0, 2, 1)
    rel = rel_norm * rel_std_t.view(1, 1, 2) + rel_mean_t.view(1, 1, 2)
    cumsum = torch.cumsum(rel, dim=1)
    return torch.cat([anchor_t[:, None, :], anchor_t[:, None, :] + cumsum], dim=1)


def loss_and_abs(
    x_t: torch.Tensor,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    t_step: int,
    y_t: torch.Tensor,
    conf_t: torch.Tensor,
    anchor_t: torch.Tensor,
    rel_mean_t: torch.Tensor,
    rel_std_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    abs_hat = x0_hat_abs(x_t, model, diffusion, t_step, anchor_t, rel_mean_t, rel_std_t)
    sigma2 = SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t))
    sigma = torch.sqrt(sigma2)
    weighted_residual = (abs_hat - y_t) / sigma[..., None]
    return torch.linalg.vector_norm(weighted_residual), abs_hat


def evaluate_candidate(
    x_candidate: torch.Tensor,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    t_step: int,
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
            t_step,
            y_t,
            conf_t,
            anchor_t,
            rel_mean_t,
            rel_std_t,
        )
    abs_np = abs_hat.detach().cpu().numpy().astype(np.float32)[0]
    finite = bool(torch.isfinite(loss).cpu()) and bool(np.isfinite(abs_np).all())
    return float(loss.detach().cpu()), ade_single(abs_np, clean_np), acceleration_rms_single(abs_np), finite


def load_case(condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clean = np.load(ARRAY_DIR / f"{condition}_clean.npy").astype(np.float32)[TRAJECTORY_ID]
    degraded = np.load(ARRAY_DIR / f"{condition}_degraded.npy").astype(np.float32)[TRAJECTORY_ID : TRAJECTORY_ID + 1]
    confidence = np.load(ARRAY_DIR / f"{condition}_confidence.npy").astype(np.float32)[TRAJECTORY_ID : TRAJECTORY_ID + 1]
    fused = np.load(ARRAY_DIR / f"{condition}_fused_t1_tau07_gamma2.npy").astype(np.float32)[TRAJECTORY_ID : TRAJECTORY_ID + 1]
    return clean, degraded, confidence, fused


def run_one_case(
    condition: str,
    t_step: int,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
) -> list[dict]:
    clean, degraded, confidence, fused = load_case(condition)
    init_rel = to_rel(fused)
    init_rel_norm = normalize_rel(init_rel, rel_mean, rel_std)
    x0 = torch.from_numpy(init_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_tensor = torch.full((1,), t_step, device=DEVICE, dtype=torch.long)
    x_t, _ = diffusion.q_sample(x0, t_tensor)

    y_t = torch.from_numpy(degraded).to(DEVICE, dtype=torch.float32)
    conf_t = torch.from_numpy(confidence).to(DEVICE, dtype=torch.float32)
    anchor_t = torch.from_numpy(degraded[:, 0, :]).to(DEVICE, dtype=torch.float32)
    rel_mean_t = torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32)
    rel_std_t = torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32)

    x_t = x_t.detach().requires_grad_(True)
    loss_zero, abs_zero = loss_and_abs(x_t, model, diffusion, t_step, y_t, conf_t, anchor_t, rel_mean_t, rel_std_t)
    grad = torch.autograd.grad(loss_zero, x_t, retain_graph=False)[0]
    grad_norm = float(torch.linalg.vector_norm(grad.detach()).cpu())
    x_norm = float(torch.linalg.vector_norm(x_t.detach()).cpu())
    zero_np = abs_zero.detach().cpu().numpy().astype(np.float32)[0]
    ade_zero = ade_single(zero_np, clean)
    accel_zero = acceleration_rms_single(zero_np)

    rows = []
    for eps in EPS_VALUES:
        update_norm = float(eps) * grad_norm
        x_minus = (x_t.detach() - float(eps) * grad.detach()).detach()
        x_plus = (x_t.detach() + float(eps) * grad.detach()).detach()
        l_minus, ade_minus, accel_minus, finite_minus = evaluate_candidate(
            x_minus, model, diffusion, t_step, y_t, conf_t, anchor_t, rel_mean_t, rel_std_t, clean
        )
        l_plus, ade_plus, accel_plus, finite_plus = evaluate_candidate(
            x_plus, model, diffusion, t_step, y_t, conf_t, anchor_t, rel_mean_t, rel_std_t, clean
        )
        rows.append(
            {
                "condition": condition,
                "trajectory_id": TRAJECTORY_ID,
                "t": t_step,
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
    return rows


def summarize_case(df: pd.DataFrame) -> dict:
    small = df[df["eps"].isin(SMALL_EPS)].sort_values("eps")
    stable = df[df["delta_L_minus"] < 0].sort_values("eps")
    overshoot = df[df["delta_L_minus"] > 0].sort_values("eps")
    first = small.iloc[0]
    sign_correct = bool(((small["delta_L_minus"] < 0) & (small["delta_L_plus"] > 0)).any())
    sign_reversed = bool(first["delta_L_plus"] < 0 and first["delta_L_minus"] >= 0)
    interface_suspect = bool((df["delta_L_minus"] >= 0).all() and (df["delta_L_plus"] >= 0).all())
    if sign_reversed:
        classification = "SIGN_REVERSED"
    elif interface_suspect:
        classification = "INTERFACE_SUSPECT"
    elif sign_correct:
        classification = "SIGN_CORRECT_STABLE"
    else:
        classification = "INCONCLUSIVE"
    if not bool(df["finite_minus"].all() and df["finite_plus"].all()):
        classification = "NONFINITE"

    best = df.sort_values("delta_L_minus").iloc[0]
    largest_stable_eps = float(stable["eps"].max()) if not stable.empty else math.nan
    largest_stable_update = (
        float(stable.sort_values("eps").iloc[-1]["update_to_x_ratio_minus"]) if not stable.empty else math.nan
    )
    eps_overshoot = float(overshoot["eps"].min()) if not overshoot.empty else math.nan
    notes = []
    if sign_correct:
        notes.append("minus lowers L and plus raises L for at least one small eps")
    if eps_overshoot == eps_overshoot:
        notes.append(f"overshoot begins by eps={eps_overshoot:g}")
    else:
        notes.append("no overshoot within tested eps")
    return {
        "condition": str(df["condition"].iloc[0]),
        "t": int(df["t"].iloc[0]),
        "classification": classification,
        "largest_stable_eps": largest_stable_eps,
        "largest_stable_update_to_x_ratio": largest_stable_update,
        "best_eps_by_likelihood_drop": float(best["eps"]),
        "eps_at_overshoot_if_any": eps_overshoot,
        "sign_correct": sign_correct,
        "interface_suspect": interface_suspect,
        "notes": "; ".join(notes),
    }


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = df[cols].copy()
    for col in show.columns:
        show[col] = show[col].map(
            lambda value: f"{float(value):.6g}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value)
        )
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(case_df: pd.DataFrame) -> None:
    all_sign_correct = bool(case_df["sign_correct"].all())
    any_reversed = bool((case_df["classification"] == "SIGN_REVERSED").any())
    any_interface = bool(case_df["interface_suspect"].any())
    eps_1e4_stable = bool((case_df["largest_stable_eps"] >= 1e-4).all())
    eps_1e3_stable = bool((case_df["largest_stable_eps"] >= 1e-3).all())
    eps_1e2_overshoot_broad = bool((case_df["eps_at_overshoot_if_any"] <= 1e-2).sum() >= len(case_df) * 0.75)
    global_largest_stable = float(case_df["largest_stable_eps"].max())
    conservative_stable = float(case_df["largest_stable_eps"].min())
    by_condition = case_df.groupby("condition")["largest_stable_eps"].agg(["min", "max"]).reset_index()
    by_step = case_df.groupby("t")["largest_stable_eps"].agg(["min", "max"]).reset_index()
    condition_spread = float((by_condition["max"] / by_condition["min"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).max())
    step_spread = float((by_step["max"] / by_step["min"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).max())
    condition_dependent = bool(condition_spread > 10.0)
    step_dependent = bool(step_spread > 10.0)

    if any_reversed or any_interface:
        decision_case = "Case 5"
        zeta_range = "STOP; audit implementation first"
        scale_mode = "no calibrated audit"
    elif all_sign_correct and eps_1e4_stable and eps_1e3_stable and not condition_dependent and not step_dependent:
        decision_case = "Case 1"
        zeta_range = "{1e-5, 3e-5, 1e-4, 3e-4, 1e-3}"
        scale_mode = "global constant zeta"
    elif all_sign_correct and eps_1e4_stable and not condition_dependent and not step_dependent:
        decision_case = "Case 2"
        zeta_range = "{1e-5, 3e-5, 1e-4, 3e-4}"
        scale_mode = "conservative global constant zeta"
    elif condition_dependent:
        decision_case = "Case 3"
        zeta_range = "do not run global calibrated audit yet"
        scale_mode = "condition-specific scale diagnosis"
    elif step_dependent:
        decision_case = "Case 4"
        zeta_range = "do not run constant zeta audit yet"
        scale_mode = "step-wise schedule diagnosis"
    else:
        decision_case = "Mixed"
        zeta_range = "{1e-5, 3e-5, 1e-4}"
        scale_mode = "very conservative global constant zeta only if pre-registered"

    lines = [
        "# E3-DPS Extended Sign Check",
        "",
        "This is a single-step numerical diagnostic only. It does not run full DPS, calibrated audit, training, new hold-out generation, or method tuning.",
        "",
        "## Summary Answers",
        "",
        f"- Sign correct across all 6 conditions and t in {{1,3,5}}: `{all_sign_correct}`",
        f"- Any sign-reversed case: `{any_reversed}`",
        f"- Any interface-suspect case: `{any_interface}`",
        f"- Global largest stable eps: `{global_largest_stable}`",
        f"- Most conservative stable eps: `{conservative_stable}`",
        f"- eps=1e-4 stable everywhere: `{eps_1e4_stable}`",
        f"- eps=1e-3 stable everywhere: `{eps_1e3_stable}`",
        f"- eps=1e-2 overshoots broadly: `{eps_1e2_overshoot_broad}`",
        f"- Stable eps strongly condition-dependent: `{condition_dependent}`",
        f"- Stable eps strongly step-dependent: `{step_dependent}`",
        f"- Decision case: `{decision_case}`",
        f"- Recommended Phase A zeta range: `{zeta_range}`",
        f"- Recommended scaling mode: `{scale_mode}`",
        "",
        "## Per-Case Stable Range",
        "",
        markdown_table(
            case_df,
            [
                "condition",
                "t",
                "classification",
                "largest_stable_eps",
                "largest_stable_update_to_x_ratio",
                "best_eps_by_likelihood_drop",
                "eps_at_overshoot_if_any",
            ],
        ),
        "",
        "## Condition Dependence",
        "",
        markdown_table(by_condition, ["condition", "min", "max"]),
        "",
        "## Step Dependence",
        "",
        markdown_table(by_step, ["t", "min", "max"]),
        "",
        "## Output Files",
        "",
        f"- `{rel(CSV_PATH)}`",
        f"- `{rel(CASE_SUMMARY_PATH)}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {rel(SUMMARY_PATH)} written")


def main() -> None:
    require_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, diffusion, rel_mean, rel_std = load_model()
    rows: list[dict] = []
    for condition in CONDITIONS:
        for t_step in T_STEPS:
            rows.extend(run_one_case(condition, t_step, model, diffusion, rel_mean, rel_std))
            print(f"[DONE] condition={condition} t={t_step}")
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"[FILE] {rel(CSV_PATH)} written")
    case_rows = []
    for (_, _), sub in df.groupby(["condition", "t"]):
        case_rows.append(summarize_case(sub))
    case_df = pd.DataFrame(case_rows).sort_values(["condition", "t"])
    case_df.to_csv(CASE_SUMMARY_PATH, index=False)
    print(f"[FILE] {rel(CASE_SUMMARY_PATH)} written")
    write_summary(case_df)
    print("E3_DPS_EXTENDED_SIGN_CHECK_COMPLETE")
    print(case_df.to_string(index=False))
    print(f"output_dir={rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
