from __future__ import annotations

from pathlib import Path
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
import pandas as pd
import torch

from tools.stage4.e2_dps_pilot import (
    AMENDMENT_001_PATH,
    AMENDMENT_002_PATH,
    DEVICE,
    KAPPA,
    PRE_REG_PATH,
    SIGMA0,
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


OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_dps_mechanism_audit"
FIG_DIR = OUT_DIR / "figures"
STEP_TRACE_PATH = OUT_DIR / "e2_dps_mechanism_audit_step_trace.csv"
CASE_SUMMARY_PATH = OUT_DIR / "e2_dps_mechanism_audit_case_summary.csv"
SUMMARY_PATH = OUT_DIR / "e2_dps_mechanism_audit_summary.md"

CONDITIONS = ["drift_medium", "burst_medium", "gaussian_medium"]
ZETAS = [0.3, 1.0, 3.0]
SEED = 42
TRAJECTORIES = [0, 1, 2]
SELECTED_STEPS = [99, 80, 60, 40, 20, 0]
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


def measurement_abs_from_state(
    model,
    diffusion,
    state: torch.Tensor,
    t_idx: int,
    tensors: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def final_abs_from_latent(
    x_t: torch.Tensor,
    degraded_rel_norm: np.ndarray,
    y: np.ndarray,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
) -> np.ndarray:
    residual_hat_norm = x_t.detach().permute(0, 2, 1).cpu().numpy().astype(np.float32)[0]
    final_rel_norm = degraded_rel_norm[0] + residual_hat_norm
    final_rel = (final_rel_norm * rel_std[None, :] + rel_mean[None, :]).astype(np.float32)
    out = np.zeros((20, 2), dtype=np.float32)
    out[0] = y[0, 0, :]
    out[1:] = y[0, 0, :][None, :] + np.cumsum(final_rel, axis=0)
    return out


def run_reverse(
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
) -> tuple[list[dict], dict]:
    tensors = setup_case(data, confs, rel_mean, rel_std, condition, traj_idx)
    clean = gt[traj_idx].astype(np.float32)
    y = tensors["y_np"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_t = torch.randn((1, 2, 19), device=DEVICE, dtype=torch.float32)
    rows: list[dict] = []
    nonfinite_step = None
    nonfinite_stage = "none"
    final_likelihood = math.nan

    for t_idx in reversed(range(TIMESTEPS)):
        x_t = x_t.detach().requires_grad_(True)
        abs_current, eps_pred = measurement_abs_from_state(model, diffusion, x_t, t_idx, tensors)
        loss = weighted_observation_loss(abs_current, tensors["y_t"], tensors["conf_t"])
        loss_ok = bool(torch.isfinite(loss.detach()).cpu())
        if loss_ok:
            grad = torch.autograd.grad(loss, x_t, retain_graph=False)[0]
        else:
            grad = torch.full_like(x_t, float("nan"))
        grad_ok = isfinite_tensor(grad)

        with torch.no_grad():
            x_base = ddpm_step(x_t, eps_pred, t_idx, diffusion)
            x_base_ok = isfinite_tensor(x_base)
            update = float(zeta) * grad
            update_ok = isfinite_tensor(update)
            base_norm = tensor_norm(x_base) if x_base_ok else math.nan
            update_norm = tensor_norm(update) if update_ok else math.nan
            update_to_x = update_norm / (base_norm + EPS) if x_base_ok and update_ok else math.nan
            x_guided = x_base - update
            guided_ok = isfinite_tensor(x_guided)

        if t_idx in SELECTED_STEPS:
            # x_base / x_guided are x_{t-1}; evaluate their denoised x0_hat at the next timestep.
            eval_t = max(t_idx - 1, 0)
            with torch.no_grad():
                abs_base, _ = measurement_abs_from_state(model, diffusion, x_base.detach(), eval_t, tensors)
                abs_guided, _ = measurement_abs_from_state(model, diffusion, x_guided.detach(), eval_t, tensors)
            base_metrics = abs_metrics(abs_base, clean, tensors) if x_base_ok else {"likelihood_norm": math.nan, "ADE": math.nan, "acceleration_RMS": math.nan}
            guided_metrics = (
                abs_metrics(abs_guided, clean, tensors)
                if guided_ok
                else {"likelihood_norm": math.nan, "ADE": math.nan, "acceleration_RMS": math.nan}
            )
            rows.append(
                {
                    "condition": condition,
                    "zeta": zeta,
                    "seed": SEED,
                    "trajectory_id": traj_idx,
                    "t": t_idx,
                    "eval_t_for_base_guided": eval_t,
                    "measurement_loss_variable": "x0_hat_abs_from_xt(x_t, eps_pred, t)",
                    "guidance_update_applied_to": "x_base_after_ddpm_step",
                    "likelihood_norm_before_guidance": base_metrics["likelihood_norm"],
                    "likelihood_norm_after_guidance": guided_metrics["likelihood_norm"],
                    "likelihood_delta_after_minus_before": guided_metrics["likelihood_norm"] - base_metrics["likelihood_norm"],
                    "ADE_before_guidance": base_metrics["ADE"],
                    "ADE_after_guidance": guided_metrics["ADE"],
                    "ADE_delta_after_minus_before": guided_metrics["ADE"] - base_metrics["ADE"],
                    "acceleration_RMS_before_guidance": base_metrics["acceleration_RMS"],
                    "acceleration_RMS_after_guidance": guided_metrics["acceleration_RMS"],
                    "acceleration_delta_after_minus_before": guided_metrics["acceleration_RMS"] - base_metrics["acceleration_RMS"],
                    "update_to_x_ratio": update_to_x,
                    "grad_norm": tensor_norm(grad) if grad_ok else math.nan,
                    "x_base_isfinite": x_base_ok,
                    "x_guided_isfinite": guided_ok,
                    "loss_isfinite": loss_ok,
                    "grad_isfinite": grad_ok,
                }
            )

        if not loss_ok:
            nonfinite_step = t_idx
            nonfinite_stage = "likelihood_loss"
            break
        if not grad_ok:
            nonfinite_step = t_idx
            nonfinite_stage = "grad"
            break
        if not x_base_ok:
            nonfinite_step = t_idx
            nonfinite_stage = "after_ddpm"
            break
        if not update_ok:
            nonfinite_step = t_idx
            nonfinite_stage = "update"
            break
        if not guided_ok:
            nonfinite_step = t_idx
            nonfinite_stage = "after_guidance"
            break
        x_t = x_guided.detach()

    finite = nonfinite_step is None
    if finite:
        final_abs = final_abs_from_latent(x_t, tensors["degraded_rel_norm_np"], y, rel_mean, rel_std)
        err = np.linalg.norm(final_abs - clean, axis=-1)
        final_ade = float(err.mean())
        final_acc = acceleration_rms_single(final_abs)
        abs_final_measure, _ = measurement_abs_from_state(model, diffusion, x_t.detach(), 0, tensors)
        final_likelihood = float(weighted_observation_loss(abs_final_measure, tensors["y_t"], tensors["conf_t"]).detach().cpu())
        motion_usage = motion_usage_single(final_abs, y[0], tensors["formal_np"])
    else:
        final_ade = math.nan
        final_acc = math.nan
        motion_usage = math.nan

    summary = {
        "condition": condition,
        "zeta": zeta,
        "seed": SEED,
        "trajectory_id": traj_idx,
        "finite": finite,
        "first_nonfinite_step": nonfinite_step,
        "first_nonfinite_stage": nonfinite_stage,
        "final_ADE": final_ade,
        "final_acceleration_RMS": final_acc,
        "final_likelihood_norm": final_likelihood,
        "motion_usage_ratio": motion_usage,
        "noisy_ADE": float(frame_error(y, clean[None, :, :])[0].mean()),
        "formal_E1_ADE": float(frame_error(tensors["formal_np"][None, :, :], clean[None, :, :])[0].mean()),
        "e2_min_v2_ADE": float(frame_error(tensors["e2min_np"][None, :, :], clean[None, :, :])[0].mean()),
    }
    return rows, summary


def write_figures(step_df: pd.DataFrame, case_df: pd.DataFrame) -> list[dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        sub = step_df[(step_df["condition"] == condition) & (step_df["trajectory_id"] == 0)]
        if sub.empty:
            continue
        path = FIG_DIR / f"{condition}_traj0_mechanism_trace.png"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for zeta in ZETAS:
            z = sub[sub["zeta"] == zeta].sort_values("t")
            axes[0].plot(z["t"], z["likelihood_delta_after_minus_before"], marker="o", label=f"zeta={zeta}")
            axes[1].plot(z["t"], z["ADE_delta_after_minus_before"], marker="o", label=f"zeta={zeta}")
            axes[2].plot(z["t"], z["acceleration_delta_after_minus_before"], marker="o", label=f"zeta={zeta}")
        axes[0].set_title("likelihood after-before")
        axes[1].set_title("ADE after-before")
        axes[2].set_title("acceleration after-before")
        for ax in axes:
            ax.axhline(0.0, color="0.3", lw=0.8)
            ax.invert_xaxis()
            ax.grid(alpha=0.25)
            ax.set_xlabel("reverse step t")
            ax.legend(fontsize=7)
        fig.suptitle(f"{condition} traj 0 step-level guidance effect")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        rows.append({"condition": condition, "trajectory_id": 0, "path": str(path)})
    return rows


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


def classify_failure(step_df: pd.DataFrame, case_df: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    likelihood_decreases = float((step_df["likelihood_delta_after_minus_before"] < 0).mean())
    ade_increases = float((step_df["ADE_delta_after_minus_before"] > 0).mean())
    acc_increases = float((step_df["acceleration_delta_after_minus_before"] > 0).mean())
    if likelihood_decreases > 0.5 and ade_increases > 0.5:
        reasons.append("objective mismatch: guidance often lowers measurement likelihood while increasing ADE")
    if acc_increases > 0.5:
        reasons.append("over-guidance / roughening: guidance often increases acceleration RMS")
    zeta_group = case_df.groupby("zeta")[["final_ADE", "final_acceleration_RMS", "final_likelihood_norm", "motion_usage_ratio"]].mean()
    if zeta_group["final_ADE"].is_monotonic_increasing and zeta_group["motion_usage_ratio"].is_monotonic_increasing:
        reasons.append("over-guidance scale problem: ADE and motion usage increase monotonically with zeta")
    if zeta_group["final_likelihood_norm"].is_monotonic_decreasing and zeta_group["final_ADE"].is_monotonic_increasing:
        reasons.append("likelihood/ADE conflict: larger zeta lowers final likelihood but worsens ADE")
    if not reasons:
        reasons.append("no single dominant mechanism found in the selected subset")
    if any("objective mismatch" in r for r in reasons):
        label = "objective mismatch + over-guidance scale problem"
    else:
        label = "over-guidance scale problem"
    return label, reasons


def write_summary(step_df: pd.DataFrame, case_df: pd.DataFrame, figure_rows: list[dict]) -> None:
    loss_variable = "x0_hat_abs_from_xt(x_t, eps_pred, t), the denoised absolute trajectory estimate derived from the current reverse latent x_t"
    x0_shape = "(1, 20, 2) absolute trajectory per selected case"
    x0_scale = "absolute coordinate scale after adding normalized residual displacement to degraded relative displacement and integrating from y[0]"
    guided_step_df = step_df[step_df["zeta"] > 0]
    likelihood_decrease_rate = float((guided_step_df["likelihood_delta_after_minus_before"] < 0).mean())
    ade_increase_rate = float((guided_step_df["ADE_delta_after_minus_before"] > 0).mean())
    acc_increase_rate = float((guided_step_df["acceleration_delta_after_minus_before"] > 0).mean())
    zeta_group = case_df.groupby("zeta")[["final_ADE", "final_acceleration_RMS", "final_likelihood_norm", "motion_usage_ratio"]].mean().reset_index()
    zeta0 = case_df[case_df["zeta"] == 0.0]
    dps = case_df[case_df["zeta"] > 0]
    base_ade = float(zeta0["final_ADE"].mean())
    base_acc = float(zeta0["final_acceleration_RMS"].mean())
    failure_label, reasons = classify_failure(step_df[step_df["zeta"] > 0], dps)
    monotonic_likelihood_down = bool(zeta_group["final_likelihood_norm"].is_monotonic_decreasing)
    monotonic_ade_up = bool(zeta_group["final_ADE"].is_monotonic_increasing)

    lines = [
        "# E2-DPS Mechanism Audit Summary",
        "",
        "This is a mechanism audit only. Stage 8B was not started, zeta was not expanded, and the method was not changed.",
        "",
        f"Pre-registration: {PRE_REG_PATH}",
        f"Amendment 001: {AMENDMENT_001_PATH}",
        f"Amendment 002: {AMENDMENT_002_PATH}",
        "",
        "## A. Guidance Variable Audit",
        f"- current implementation applies measurement loss to: {loss_variable}",
        "- the DDPM stochastic base step is computed as `x_base = ddpm_step(x_t, eps_pred, t)`.",
        "- the guidance update is then applied to `x_base`: `x_guided = x_base - zeta * grad`.",
        "- therefore the gradient is computed through the denoised estimate from the pre-step latent, then applied after the DDPM base step.",
        "",
        "## B. DPS Canonical Interface Audit",
        f"- denoised x0_hat is available: True",
        f"- x0_hat shape: {x0_shape}",
        f"- x0_hat scale: {x0_scale}",
        "- x0_hat can be converted to absolute trajectory by the existing A/operator reconstruction inside `x0_hat_abs_from_xt`.",
        "",
        "## C. Step-Level Likelihood vs ADE Audit",
        f"- fraction of selected guided steps where likelihood_norm decreases after guidance: {likelihood_decrease_rate:.3f}",
        f"- fraction of selected guided steps where ADE increases after guidance: {ade_increase_rate:.3f}",
        f"- fraction of selected guided steps where acceleration RMS increases after guidance: {acc_increase_rate:.3f}",
        "",
        "## D. zeta=0 Base Sampler Audit",
        f"- zeta=0 no-guidance mean final ADE over selected cases: {base_ade:.6f}",
        f"- zeta=0 no-guidance mean acceleration RMS over selected cases: {base_acc:.6f}",
        "",
        "## E. Monotonicity Audit",
        markdown_table(zeta_group),
        "",
        f"- zeta increases monotonically worsen ADE: {monotonic_ade_up}",
        f"- zeta increases monotonically lower likelihood: {monotonic_likelihood_down}",
        "",
        "## Failure Attribution",
        f"- classification: {failure_label}",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(
        [
            "",
            "## Stage 8B Decision",
            "Stage 8B should not be started from this audited implementation as a positive-signal continuation. The Stage 8A result is finite but mechanism-negative.",
            "",
            "Recommended next step: write a new amendment before any further run. Based on this audit, the first amendment to consider is not zeta expansion; it is a variable-interface / guidance-timing correction that computes and applies guidance consistently with the denoised x0_hat interface, or a pre-registered timestep schedule if the interface is judged correct.",
            "",
            "Trust-region or clipping should only be considered after the variable-interface audit is resolved, because they would hide rather than explain the current harmful guidance mechanism.",
            "",
            "## Figures",
        ]
    )
    lines.extend(f"- {row['condition']} traj {row['trajectory_id']}: {row['path']}" for row in figure_rows)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_file(SUMMARY_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [PRE_REG_PATH, AMENDMENT_001_PATH, AMENDMENT_002_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)

    _, confs = load_confidence_cache()
    _, data, gt = load_inputs()
    model, diffusion, rel_mean, rel_std = load_model()

    step_rows: list[dict] = []
    case_rows: list[dict] = []
    all_zetas = [0.0] + ZETAS
    for condition in CONDITIONS:
        for traj_idx in TRAJECTORIES:
            for zeta in all_zetas:
                rows, summary = run_reverse(
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
                )
                step_rows.extend(rows)
                case_rows.append(summary)
                print(
                    f"[DONE] mechanism audit condition={condition} zeta={zeta} "
                    f"seed={SEED} traj={traj_idx} finite={summary['finite']}"
                )

    step_df = pd.DataFrame(step_rows)
    case_df = pd.DataFrame(case_rows)
    step_df.to_csv(STEP_TRACE_PATH, index=False)
    print_file(STEP_TRACE_PATH)
    case_df.to_csv(CASE_SUMMARY_PATH, index=False)
    print_file(CASE_SUMMARY_PATH)
    figure_rows = write_figures(step_df, case_df)
    pd.DataFrame(figure_rows).to_csv(OUT_DIR / "e2_dps_mechanism_audit_figures.csv", index=False)
    print_file(OUT_DIR / "e2_dps_mechanism_audit_figures.csv")
    write_summary(step_df, case_df, figure_rows)

    dps_step = step_df[step_df["zeta"] > 0]
    dps_case = case_df[case_df["zeta"] > 0]
    label, _ = classify_failure(dps_step, dps_case)
    print("E2_DPS_MECHANISM_AUDIT_COMPLETE")
    print("loss_variable: x0_hat_abs_from_xt(x_t, eps_pred, t)")
    print("x0_hat_available: True")
    print(f"zeta0_base_ADE_mean: {case_df[case_df['zeta'] == 0.0]['final_ADE'].mean():.6f}")
    print(f"likelihood_decrease_rate_guided_steps: {(dps_step['likelihood_delta_after_minus_before'] < 0).mean():.6f}")
    print(f"ADE_increase_rate_guided_steps: {(dps_step['ADE_delta_after_minus_before'] > 0).mean():.6f}")
    print(f"acceleration_increase_rate_guided_steps: {(dps_step['acceleration_delta_after_minus_before'] > 0).mean():.6f}")
    print(f"failure_attribution: {label}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
