from __future__ import annotations

from pathlib import Path
import json
import math
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CACHE_DIR = PROJECT_ROOT / "outputs" / "stage4" / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import numpy as np
import pandas as pd
import torch

from tools.stage4.e2_dps_pilot import (
    AMENDMENT_001_PATH,
    AMENDMENT_002_PATH,
    DEVICE,
    PRE_REG_PATH,
    TIMESTEPS,
    ddpm_step,
    frame_error,
    load_confidence_cache,
    load_inputs,
    load_model,
    normalize_rel,
    to_rel,
    weighted_observation_loss,
    x0_hat_abs_from_xt,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_dps_interface_audit"
CASE_SUMMARY_PATH = OUT_DIR / "e2_dps_interface_audit_case_summary.csv"
STEP_TRACE_PATH = OUT_DIR / "e2_dps_interface_audit_step_trace.csv"
INIT_SUMMARY_PATH = OUT_DIR / "e2_dps_interface_audit_initialization_summary.json"
SUMMARY_PATH = OUT_DIR / "e2_dps_interface_audit_summary.md"

CONDITIONS = ["drift_medium", "burst_medium", "gaussian_medium"]
ZETAS = [0.3, 1.0]
SEED = 42
TRAJECTORIES = [0, 1, 2]
SELECTED_STEPS = [99, 80, 60, 40, 20, 0]
VERSIONS = ["A_current_post_base_update", "B_pre_step_guidance", "C_post_step_consistent"]
EPS = 1e-12


def print_file(path: Path) -> None:
    print(f"[FILE] {path} written")


def tensor_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach()).cpu())


def isfinite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def acceleration_rms_single(pred: np.ndarray) -> float:
    acc = pred[2:, :] - 2.0 * pred[1:-1, :] + pred[:-2, :]
    return float(np.sqrt(np.mean(np.sum(acc**2, axis=-1))))


def rmse_single(pred: np.ndarray, clean: np.ndarray) -> float:
    err = np.linalg.norm(pred - clean, axis=-1)
    return float(np.sqrt(np.mean(err**2)))


def motion_usage_single(pred: np.ndarray, noisy: np.ndarray, formal_e1: np.ndarray) -> float:
    dp = np.diff(pred, axis=0)
    dy = np.diff(noisy, axis=0)
    de1 = np.diff(formal_e1, axis=0)
    den = float(np.linalg.norm(de1 - dy, axis=-1).mean())
    num = float(np.linalg.norm(dp - dy, axis=-1).mean())
    return num / den if den > 0 else math.nan


def setup_case(
    data: dict[str, dict],
    confs: dict[str, np.ndarray],
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    condition: str,
    traj_idx: int,
) -> dict:
    y = data[condition]["degraded"][traj_idx : traj_idx + 1]
    conf = confs[condition][traj_idx : traj_idx + 1]
    degraded_rel = to_rel(y)
    degraded_rel_norm = normalize_rel(degraded_rel, rel_mean, rel_std)
    return {
        "y_np": y.astype(np.float32),
        "conf_np": conf.astype(np.float32),
        "formal_np": data[condition]["formal_e1"][traj_idx].astype(np.float32),
        "e2min_np": data[condition]["e2_min_v2"][traj_idx].astype(np.float32),
        "degraded_rel_norm_np": degraded_rel_norm.astype(np.float32),
        "x_cond": torch.from_numpy(degraded_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32),
        "degraded_rel_norm_t": torch.from_numpy(degraded_rel_norm).to(DEVICE, dtype=torch.float32),
        "y_t": torch.from_numpy(y).to(DEVICE, dtype=torch.float32),
        "conf_t": torch.from_numpy(conf).to(DEVICE, dtype=torch.float32),
        "start_t": torch.from_numpy(y[:, 0, :]).to(DEVICE, dtype=torch.float32),
        "rel_mean_t": torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32),
        "rel_std_t": torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32),
    }


def measurement_abs_from_state(model, diffusion, state: torch.Tensor, t_idx: int, tensors: dict) -> tuple[torch.Tensor, torch.Tensor]:
    t_eval = int(max(0, min(TIMESTEPS - 1, t_idx)))
    t_cur = torch.full((state.shape[0],), t_eval, device=DEVICE, dtype=torch.long)
    eps_pred = model(state, tensors["x_cond"], t_cur)
    abs_hat = x0_hat_abs_from_xt(
        state,
        eps_pred,
        t_eval,
        tensors["degraded_rel_norm_t"],
        tensors["start_t"],
        tensors["rel_mean_t"],
        tensors["rel_std_t"],
        diffusion,
    )
    return abs_hat, eps_pred


def abs_metrics(abs_hat: torch.Tensor, clean: np.ndarray, tensors: dict) -> dict:
    arr = abs_hat.detach().cpu().numpy().astype(np.float32)[0]
    err = np.linalg.norm(arr - clean, axis=-1)
    likelihood = weighted_observation_loss(abs_hat, tensors["y_t"], tensors["conf_t"])
    return {
        "likelihood_norm": float(likelihood.detach().cpu()) if bool(torch.isfinite(likelihood).detach().cpu()) else math.nan,
        "ADE": float(err.mean()),
        "acceleration_RMS": acceleration_rms_single(arr),
    }


def final_abs_from_latent(x_t: torch.Tensor, tensors: dict, rel_mean: np.ndarray, rel_std: np.ndarray) -> np.ndarray:
    residual_hat_norm = x_t.detach().permute(0, 2, 1).cpu().numpy().astype(np.float32)[0]
    degraded_rel_norm = tensors["degraded_rel_norm_np"]
    y = tensors["y_np"]
    final_rel_norm = degraded_rel_norm[0] + residual_hat_norm
    final_rel = (final_rel_norm * rel_std[None, :] + rel_mean[None, :]).astype(np.float32)
    out = np.zeros((20, 2), dtype=np.float32)
    out[0] = y[0, 0, :]
    out[1:] = y[0, 0, :][None, :] + np.cumsum(final_rel, axis=0)
    return out


def final_metrics(final_abs: np.ndarray, clean: np.ndarray, tensors: dict, model, diffusion, x_t: torch.Tensor) -> dict:
    err = np.linalg.norm(final_abs - clean, axis=-1)
    conf = tensors["conf_np"][0]
    high = conf > 0.7
    low = conf < 0.3
    abs_final_measure, _ = measurement_abs_from_state(model, diffusion, x_t.detach(), 0, tensors)
    return {
        "final_ADE": float(err.mean()),
        "final_RMSE": rmse_single(final_abs, clean),
        "final_acceleration_RMS": acceleration_rms_single(final_abs),
        "final_likelihood_norm": float(weighted_observation_loss(abs_final_measure, tensors["y_t"], tensors["conf_t"]).detach().cpu()),
        "motion_usage_ratio": motion_usage_single(final_abs, tensors["y_np"][0], tensors["formal_np"]),
        "noisy_reversion_gap": float(err.mean() - frame_error(tensors["y_np"], clean[None, :, :])[0].mean()),
        "ADE_high": float(err[high].mean()) if np.any(high) else math.nan,
        "ADE_low": float(err[low].mean()) if np.any(low) else math.nan,
    }


def compute_grad_on_state(model, diffusion, state: torch.Tensor, t_idx: int, tensors: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = state.detach().requires_grad_(True)
    abs_hat, eps_pred = measurement_abs_from_state(model, diffusion, state, t_idx, tensors)
    loss = weighted_observation_loss(abs_hat, tensors["y_t"], tensors["conf_t"])
    if bool(torch.isfinite(loss.detach()).cpu()):
        grad = torch.autograd.grad(loss, state, retain_graph=False)[0]
    else:
        grad = torch.full_like(state, float("nan"))
    return state, loss, grad


def run_baseline_no_guidance(model, diffusion, tensors: dict, rel_mean: np.ndarray, rel_std: np.ndarray, clean: np.ndarray) -> tuple[dict, torch.Tensor]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_t = torch.randn((1, 2, 19), device=DEVICE, dtype=torch.float32)
    for t_idx in reversed(range(TIMESTEPS)):
        with torch.no_grad():
            abs_hat, eps_pred = measurement_abs_from_state(model, diffusion, x_t, t_idx, tensors)
            x_t = ddpm_step(x_t, eps_pred, t_idx, diffusion)
    final_abs = final_abs_from_latent(x_t, tensors, rel_mean, rel_std)
    metrics = final_metrics(final_abs, clean, tensors, model, diffusion, x_t)
    metrics.update({"finite": bool(np.isfinite(final_abs).all()), "first_nonfinite_step": None, "first_nonfinite_stage": "none"})
    return metrics, x_t


def run_version(
    model,
    diffusion,
    data: dict[str, dict],
    confs: dict[str, np.ndarray],
    gt: np.ndarray,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    condition: str,
    traj_idx: int,
    zeta: float,
    version: str,
    baseline_metrics: dict,
) -> tuple[list[dict], dict]:
    tensors = setup_case(data, confs, rel_mean, rel_std, condition, traj_idx)
    clean = gt[traj_idx].astype(np.float32)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_t = torch.randn((1, 2, 19), device=DEVICE, dtype=torch.float32)
    rows: list[dict] = []
    finite = True
    first_nonfinite_step = None
    first_nonfinite_stage = "none"
    version_available = True
    max_update_to_x_ratio = 0.0

    for t_idx in reversed(range(TIMESTEPS)):
        if version == "A_current_post_base_update":
            x_grad, loss, grad = compute_grad_on_state(model, diffusion, x_t, t_idx, tensors)
            grad_ok = isfinite_tensor(grad)
            with torch.no_grad():
                abs_hat, eps_pred = measurement_abs_from_state(model, diffusion, x_t.detach(), t_idx, tensors)
                x_base = ddpm_step(x_t.detach(), eps_pred, t_idx, diffusion)
                x_before = x_base
                x_after = x_base - float(zeta) * grad
                eval_t = max(t_idx - 1, 0)
        elif version == "B_pre_step_guidance":
            x_grad, loss, grad = compute_grad_on_state(model, diffusion, x_t, t_idx, tensors)
            grad_ok = isfinite_tensor(grad)
            x_t_guided = x_t.detach() - float(zeta) * grad
            with torch.no_grad():
                abs_before, _ = measurement_abs_from_state(model, diffusion, x_t.detach(), t_idx, tensors)
                x_before = x_t.detach()
                abs_hat_guided, eps_guided = measurement_abs_from_state(model, diffusion, x_t_guided.detach(), t_idx, tensors)
                x_after = ddpm_step(x_t_guided.detach(), eps_guided, t_idx, diffusion)
                eval_t = max(t_idx - 1, 0)
        elif version == "C_post_step_consistent":
            with torch.no_grad():
                abs_hat, eps_pred = measurement_abs_from_state(model, diffusion, x_t.detach(), t_idx, tensors)
                x_base_detached = ddpm_step(x_t.detach(), eps_pred, t_idx, diffusion)
            eval_t = max(t_idx - 1, 0)
            x_base_for_grad, loss, grad = compute_grad_on_state(model, diffusion, x_base_detached, eval_t, tensors)
            grad_ok = isfinite_tensor(grad)
            x_before = x_base_detached
            with torch.no_grad():
                x_after = x_base_detached - float(zeta) * grad
        else:
            raise ValueError(version)

        loss_ok = bool(torch.isfinite(loss.detach()).cpu())
        before_ok = isfinite_tensor(x_before)
        update = float(zeta) * grad
        update_ok = isfinite_tensor(update)
        after_ok = isfinite_tensor(x_after)
        before_norm = tensor_norm(x_before) if before_ok else math.nan
        update_norm = tensor_norm(update) if update_ok else math.nan
        update_to_x = update_norm / (before_norm + EPS) if before_ok and update_ok else math.nan
        if np.isfinite(update_to_x):
            max_update_to_x_ratio = max(max_update_to_x_ratio, float(update_to_x))

        if t_idx in SELECTED_STEPS:
            with torch.no_grad():
                if version == "B_pre_step_guidance":
                    abs_before_eval, _ = measurement_abs_from_state(model, diffusion, x_before.detach(), t_idx, tensors)
                    abs_after_eval, _ = measurement_abs_from_state(model, diffusion, x_after.detach(), eval_t, tensors)
                else:
                    abs_before_eval, _ = measurement_abs_from_state(model, diffusion, x_before.detach(), eval_t, tensors)
                    abs_after_eval, _ = measurement_abs_from_state(model, diffusion, x_after.detach(), eval_t, tensors)
            before_metrics = abs_metrics(abs_before_eval, clean, tensors) if before_ok else {"likelihood_norm": math.nan, "ADE": math.nan, "acceleration_RMS": math.nan}
            after_metrics = abs_metrics(abs_after_eval, clean, tensors) if after_ok else {"likelihood_norm": math.nan, "ADE": math.nan, "acceleration_RMS": math.nan}
            rows.append(
                {
                    "condition": condition,
                    "version": version,
                    "version_available": version_available,
                    "zeta": zeta,
                    "seed": SEED,
                    "trajectory_id": traj_idx,
                    "t": t_idx,
                    "likelihood_before_guidance": before_metrics["likelihood_norm"],
                    "likelihood_after_guidance": after_metrics["likelihood_norm"],
                    "likelihood_delta_after_minus_before": after_metrics["likelihood_norm"] - before_metrics["likelihood_norm"],
                    "ADE_before_guidance": before_metrics["ADE"],
                    "ADE_after_guidance": after_metrics["ADE"],
                    "ADE_delta_after_minus_before": after_metrics["ADE"] - before_metrics["ADE"],
                    "acceleration_RMS_before_guidance": before_metrics["acceleration_RMS"],
                    "acceleration_RMS_after_guidance": after_metrics["acceleration_RMS"],
                    "acceleration_delta_after_minus_before": after_metrics["acceleration_RMS"] - before_metrics["acceleration_RMS"],
                    "update_to_x_ratio": update_to_x,
                    "grad_norm": tensor_norm(grad) if grad_ok else math.nan,
                    "loss_isfinite": loss_ok,
                    "grad_isfinite": grad_ok,
                    "before_isfinite": before_ok,
                    "after_isfinite": after_ok,
                }
            )

        if not loss_ok:
            finite = False
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "likelihood_loss"
            break
        if not grad_ok:
            finite = False
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "grad"
            break
        if not before_ok:
            finite = False
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "before_guidance"
            break
        if not update_ok:
            finite = False
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "update"
            break
        if not after_ok:
            finite = False
            first_nonfinite_step = t_idx
            first_nonfinite_stage = "after_guidance"
            break
        x_t = x_after.detach()

    if finite:
        final_abs = final_abs_from_latent(x_t, tensors, rel_mean, rel_std)
        metrics = final_metrics(final_abs, clean, tensors, model, diffusion, x_t)
    else:
        metrics = {
            "final_ADE": math.nan,
            "final_RMSE": math.nan,
            "final_acceleration_RMS": math.nan,
            "final_likelihood_norm": math.nan,
            "motion_usage_ratio": math.nan,
            "noisy_reversion_gap": math.nan,
            "ADE_high": math.nan,
            "ADE_low": math.nan,
        }
    metrics.update(
        {
            "condition": condition,
            "version": version,
            "version_available": version_available,
            "zeta": zeta,
            "seed": SEED,
            "trajectory_id": traj_idx,
            "finite": finite,
            "first_nonfinite_step": first_nonfinite_step,
            "first_nonfinite_stage": first_nonfinite_stage,
            "max_update_to_x_ratio": max_update_to_x_ratio,
            "baseline_zeta0_ADE": baseline_metrics["final_ADE"],
            "baseline_zeta0_likelihood_norm": baseline_metrics["final_likelihood_norm"],
            "baseline_zeta0_acceleration_RMS": baseline_metrics["final_acceleration_RMS"],
            "likelihood_improves_vs_zeta0": bool(metrics["final_likelihood_norm"] < baseline_metrics["final_likelihood_norm"])
            if np.isfinite(metrics["final_likelihood_norm"])
            else False,
            "ADE_improves_vs_zeta0": bool(metrics["final_ADE"] < baseline_metrics["final_ADE"])
            if np.isfinite(metrics["final_ADE"])
            else False,
            "acceleration_increases_vs_zeta0": bool(metrics["final_acceleration_RMS"] > baseline_metrics["final_acceleration_RMS"])
            if np.isfinite(metrics["final_acceleration_RMS"])
            else False,
        }
    )
    return rows, metrics


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


def write_initialization_summary(path: Path) -> dict:
    summary = {
        "initial_state_distribution": "pure Gaussian noise in normalized residual latent space",
        "initial_state_shape": [1, 2, 19],
        "uses_degraded_y_noised_initialization": False,
        "uses_stage3_conditional_output_initialization": False,
        "uses_e2_min_output_initialization": False,
        "conditioning_signal": "degraded trajectory relative displacement, normalized, passed as x_cond to ConditionalTemporalDenoiser1D",
        "zeta0_no_guidance_equivalent_to_unconditional_prior_generation": False,
        "zeta0_no_guidance_description": "conditional residual DDPM sampling from pure Gaussian residual latent, conditioned on degraded y, without DPS observation guidance",
        "is_true_refinement_from_observation_initial_state": False,
        "refinement_note": "observation y enters as conditioning and likelihood guidance, but the reverse state is initialized from noise rather than from y, Stage 3 output, or E2-Min output",
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_file(path)
    return summary


def write_summary(case_df: pd.DataFrame, step_df: pd.DataFrame, init_summary: dict) -> None:
    grouped = (
        case_df.groupby(["version", "zeta"])[
            [
                "final_ADE",
                "final_likelihood_norm",
                "final_acceleration_RMS",
                "motion_usage_ratio",
                "noisy_reversion_gap",
                "ADE_high",
                "ADE_low",
                "max_update_to_x_ratio",
            ]
        ]
        .mean()
        .reset_index()
    )
    baseline = (
        case_df[case_df["version"] == "Z_no_guidance"]
        .groupby("condition")[["final_ADE", "final_likelihood_norm", "final_acceleration_RMS"]]
        .mean()
        .reset_index()
    )
    dps = case_df[case_df["version"].isin(VERSIONS)]
    step_dps = step_df[step_df["version"].isin(VERSIONS)]
    version_availability = dps.groupby("version")["version_available"].all().to_dict()
    any_version_improves_ade = bool(dps["ADE_improves_vs_zeta0"].any())
    any_version_improves_likelihood_and_not_ade_bad = bool(
        ((dps["likelihood_improves_vs_zeta0"]) & (dps["final_ADE"] <= dps["baseline_zeta0_ADE"])).any()
    )
    version_means = grouped[grouped["version"].isin(VERSIONS)]
    best_ade_row = version_means.sort_values("final_ADE").iloc[0]
    a_means = version_means[version_means["version"] == "A_current_post_base_update"]
    b_means = version_means[version_means["version"] == "B_pre_step_guidance"]
    c_means = version_means[version_means["version"] == "C_post_step_consistent"]
    b_or_c_better_than_a = False
    if not a_means.empty:
        a_min = float(a_means["final_ADE"].min())
        bc_min = float(pd.concat([b_means, c_means])["final_ADE"].min())
        b_or_c_better_than_a = bool(bc_min < a_min)
    step_rates = (
        step_dps.groupby("version")
        .apply(
            lambda x: pd.Series(
                {
                    "likelihood_decrease_rate": float((x["likelihood_delta_after_minus_before"] < 0).mean()),
                    "ADE_increase_rate": float((x["ADE_delta_after_minus_before"] > 0).mean()),
                    "acceleration_increase_rate": float((x["acceleration_delta_after_minus_before"] > 0).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    if b_or_c_better_than_a and any_version_improves_ade:
        recommendation = "Write Amendment 003 to correct guidance placement before any further Stage 8A/8B run."
        stop_or_amend = "Amendment 003 recommended."
    elif not any_version_improves_ade:
        recommendation = "Do not enter Stage 8B. The audited placements do not improve over zeta=0; E2-DPS appears incompatible with this residual-latent refinement interface unless a new pre-registered interface is proposed."
        stop_or_amend = "Stop E2-DPS or write a more fundamental amendment; do not run Stage 8B."
    else:
        recommendation = "Do not enter Stage 8B yet; a targeted amendment is required."
        stop_or_amend = "Amendment required."

    lines = [
        "# E2-DPS Guidance Interface Audit Summary",
        "",
        "This is a very small interface audit only. Stage 8B was not started, zeta was not expanded, and the formal method was not modified.",
        "",
        f"Pre-registration: {PRE_REG_PATH}",
        f"Amendment 001: {AMENDMENT_001_PATH}",
        f"Amendment 002: {AMENDMENT_002_PATH}",
        "",
        "## 1. Current Initialization",
        f"- initial state distribution: {init_summary['initial_state_distribution']}",
        f"- uses degraded-y noised initialization: {init_summary['uses_degraded_y_noised_initialization']}",
        f"- uses Stage 3 conditional output initialization: {init_summary['uses_stage3_conditional_output_initialization']}",
        f"- uses E2-Min output initialization: {init_summary['uses_e2_min_output_initialization']}",
        f"- conditioning signal: {init_summary['conditioning_signal']}",
        "",
        "## 2. zeta=0 Base Sampler",
        f"- zeta=0 is unconditional prior generation: {init_summary['zeta0_no_guidance_equivalent_to_unconditional_prior_generation']}",
        f"- description: {init_summary['zeta0_no_guidance_description']}",
        f"- true refinement from observation initial state: {init_summary['is_true_refinement_from_observation_initial_state']}",
        "",
        "## 3. Version Availability",
        f"- Version A available: {version_availability.get('A_current_post_base_update', False)}",
        f"- Version B available: {version_availability.get('B_pre_step_guidance', False)}",
        f"- Version C available: {version_availability.get('C_post_step_consistent', False)}",
        "- Version C implementation note: x_base can be detached/requires_grad and evaluated at timestep t-1 through the existing x0_hat_abs_from_xt interface, so it is available for audit.",
        "",
        "## 4. Aggregate Metrics",
        markdown_table(grouped),
        "",
        "## 5. zeta=0 Baseline by Condition",
        markdown_table(baseline),
        "",
        "## 6. Step-Level Direction",
        markdown_table(step_rates),
        "",
        "## 7. Audit Answers",
        "- Most reasonable variable interface: Version C is the most internally consistent placement, because its loss and gradient are computed on the post-DDPM state that is actually updated.",
        "- Current Version A has a gradient/update variable mismatch: True. It computes grad with respect to pre-step x_t but applies that gradient after the stochastic DDPM base step.",
        f"- Any version makes likelihood improve while ADE does not worsen vs zeta=0: {any_version_improves_likelihood_and_not_ade_bad}",
        f"- Any version clearly improves over current Version A by ADE: {b_or_c_better_than_a}",
        f"- Best mean ADE audited row: version={best_ade_row['version']}, zeta={best_ade_row['zeta']}, ADE={best_ade_row['final_ADE']:.6f}",
        "",
        "## 8. Decision",
        f"- {recommendation}",
        f"- concise recommendation: {stop_or_amend}",
        "- Stage 8B allowed now: False.",
        "",
        "## Output Files",
        f"- case summary: {CASE_SUMMARY_PATH}",
        f"- step trace: {STEP_TRACE_PATH}",
        f"- initialization summary: {INIT_SUMMARY_PATH}",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_file(SUMMARY_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [PRE_REG_PATH, AMENDMENT_001_PATH, AMENDMENT_002_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)
    init_summary = write_initialization_summary(INIT_SUMMARY_PATH)
    _, confs = load_confidence_cache()
    _, data, gt = load_inputs()
    model, diffusion, rel_mean, rel_std = load_model()

    case_rows: list[dict] = []
    step_rows: list[dict] = []
    for condition in CONDITIONS:
        for traj_idx in TRAJECTORIES:
            tensors = setup_case(data, confs, rel_mean, rel_std, condition, traj_idx)
            clean = gt[traj_idx].astype(np.float32)
            baseline_metrics, _ = run_baseline_no_guidance(model, diffusion, tensors, rel_mean, rel_std, clean)
            baseline_row = {
                "condition": condition,
                "version": "Z_no_guidance",
                "version_available": True,
                "zeta": 0.0,
                "seed": SEED,
                "trajectory_id": traj_idx,
                **baseline_metrics,
                "max_update_to_x_ratio": 0.0,
                "baseline_zeta0_ADE": baseline_metrics["final_ADE"],
                "baseline_zeta0_likelihood_norm": baseline_metrics["final_likelihood_norm"],
                "baseline_zeta0_acceleration_RMS": baseline_metrics["final_acceleration_RMS"],
                "likelihood_improves_vs_zeta0": False,
                "ADE_improves_vs_zeta0": False,
                "acceleration_increases_vs_zeta0": False,
            }
            case_rows.append(baseline_row)
            print(f"[DONE] interface baseline condition={condition} traj={traj_idx} finite={baseline_metrics['finite']}")
            for version in VERSIONS:
                for zeta in ZETAS:
                    rows, summary = run_version(
                        model,
                        diffusion,
                        data,
                        confs,
                        gt,
                        rel_mean,
                        rel_std,
                        condition,
                        traj_idx,
                        zeta,
                        version,
                        baseline_metrics,
                    )
                    step_rows.extend(rows)
                    case_rows.append(summary)
                    print(
                        f"[DONE] interface audit condition={condition} version={version} "
                        f"zeta={zeta} traj={traj_idx} finite={summary['finite']}"
                    )

    case_df = pd.DataFrame(case_rows)
    step_df = pd.DataFrame(step_rows)
    case_df.to_csv(CASE_SUMMARY_PATH, index=False)
    print_file(CASE_SUMMARY_PATH)
    step_df.to_csv(STEP_TRACE_PATH, index=False)
    print_file(STEP_TRACE_PATH)
    write_summary(case_df, step_df, init_summary)

    grouped = case_df.groupby(["version", "zeta"])[["final_ADE", "final_likelihood_norm", "final_acceleration_RMS"]].mean().reset_index()
    print("E2_DPS_INTERFACE_AUDIT_COMPLETE")
    print(f"initialization_type: {init_summary['initial_state_distribution']}")
    print("version_availability: A=True B=True C=True")
    print(grouped.to_string(index=False))
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
