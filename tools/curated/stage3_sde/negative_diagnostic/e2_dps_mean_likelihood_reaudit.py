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

import numpy as np
import pandas as pd
import torch

from tools.stage4.e2_dps_pilot import (
    DEVICE,
    KAPPA,
    PRE_REG_PATH,
    SIGMA0,
    TIMESTEPS,
    ddpm_step,
    load_confidence_cache,
    load_inputs,
    load_model,
    normalize_rel,
    per_traj_metrics,
    to_rel,
    x0_hat_abs_from_xt,
)


AMENDMENT_PATH = PROJECT_ROOT / "docs" / "stage4" / "E2_DPS_hypothesis_amendment_001.md"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_dps_mean_likelihood_reaudit"
TRACE_PATH = OUT_DIR / "e2_dps_mean_likelihood_reaudit_trace.csv"
CASE_SUMMARY_PATH = OUT_DIR / "e2_dps_mean_likelihood_reaudit_case_summary.csv"
SUMMARY_PATH = OUT_DIR / "e2_dps_mean_likelihood_reaudit_summary.md"

CONDITIONS = ["drift_medium", "burst_medium", "gaussian_medium", "bias_medium"]
ZETAS = [0.3, 1.0, 3.0]
SEED = 42
TRAJECTORIES = [0, 1, 2]
EPS = 1e-12


def print_file(path: Path) -> None:
    print(f"[FILE] {path} written")


def tensor_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach()).cpu())


def tensor_max_abs(x: torch.Tensor) -> float:
    return float(torch.max(torch.abs(x.detach())).cpu())


def tensor_isfinite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def weighted_observation_loss_mean(abs_hat: torch.Tensor, y_t: torch.Tensor, conf_t: torch.Tensor) -> torch.Tensor:
    sigma2 = SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t))
    err2 = torch.sum((abs_hat - y_t) ** 2, dim=-1)
    return torch.mean(err2 / sigma2)


def acceleration_rms(pred: np.ndarray) -> float:
    acc = pred[2:, :] - 2.0 * pred[1:-1, :] + pred[:-2, :]
    return float(np.sqrt(np.mean(np.sum(acc**2, axis=-1))))


def reconstruct_from_rel(start_point: np.ndarray, rel: np.ndarray) -> np.ndarray:
    abs_hat = np.zeros((20, 2), dtype=np.float32)
    abs_hat[0] = start_point.astype(np.float32)
    abs_hat[1:] = start_point.astype(np.float32)[None, :] + np.cumsum(rel.astype(np.float32), axis=0)
    return abs_hat


def setup_case(
    data: dict[str, dict],
    confs: dict[str, np.ndarray],
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    condition: str,
    traj_idx: int,
) -> dict[str, torch.Tensor | np.ndarray]:
    y = data[condition]["degraded"][traj_idx : traj_idx + 1]
    conf = confs[condition][traj_idx : traj_idx + 1]
    degraded_rel = to_rel(y)
    degraded_rel_norm = normalize_rel(degraded_rel, rel_mean, rel_std)
    return {
        "y_np": y,
        "conf_np": conf,
        "degraded_rel_norm_np": degraded_rel_norm,
        "x_cond": torch.from_numpy(degraded_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32),
        "degraded_rel_norm_t": torch.from_numpy(degraded_rel_norm).to(DEVICE, dtype=torch.float32),
        "y_t": torch.from_numpy(y).to(DEVICE, dtype=torch.float32),
        "conf_t": torch.from_numpy(conf).to(DEVICE, dtype=torch.float32),
        "start_t": torch.from_numpy(y[:, 0, :]).to(DEVICE, dtype=torch.float32),
        "rel_mean_t": torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32),
        "rel_std_t": torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32),
    }


def stage_name(
    before_ok: bool,
    loss_ok: bool,
    grad_ok: bool,
    base_ok: bool,
    update_ok: bool,
    after_ok: bool,
) -> str:
    if not before_ok:
        return "before_ddpm"
    if not loss_ok:
        return "likelihood_loss"
    if not grad_ok:
        return "grad"
    if not base_ok:
        return "after_ddpm"
    if not update_ok:
        return "update"
    if not after_ok:
        return "after_guidance"
    return "none"


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
    return reconstruct_from_rel(y[0, 0, :], final_rel)


def run_case(
    model,
    diffusion,
    data: dict[str, dict],
    confs: dict[str, np.ndarray],
    gt: np.ndarray,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    condition: str,
    zeta: float,
    traj_idx: int,
) -> tuple[list[dict], dict]:
    tensors = setup_case(data, confs, rel_mean, rel_std, condition, traj_idx)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_t = torch.randn((1, 2, 19), device=DEVICE, dtype=torch.float32)
    rows: list[dict] = []
    first_nonfinite_step = None
    first_nonfinite_stage = "none"
    max_grad_norm = 0.0
    max_update_to_x_ratio = 0.0
    max_x_after_max_abs = 0.0
    final_pred = None

    for t_idx in reversed(range(TIMESTEPS)):
        x_t = x_t.detach().requires_grad_(True)
        before_ok = tensor_isfinite(x_t)
        t_cur = torch.full((1,), t_idx, device=DEVICE, dtype=torch.long)
        eps_pred = model(x_t, tensors["x_cond"], t_cur)
        abs_hat = x0_hat_abs_from_xt(
            x_t,
            eps_pred,
            t_idx,
            tensors["degraded_rel_norm_t"],
            tensors["start_t"],
            tensors["rel_mean_t"],
            tensors["rel_std_t"],
            diffusion,
        )
        loss = weighted_observation_loss_mean(abs_hat, tensors["y_t"], tensors["conf_t"])
        loss_ok = bool(torch.isfinite(loss).detach().cpu())
        if loss_ok:
            grad = torch.autograd.grad(loss, x_t, retain_graph=False)[0]
        else:
            grad = torch.full_like(x_t, float("nan"))
        grad_ok = tensor_isfinite(grad)
        grad_norm = tensor_norm(grad) if grad_ok else math.nan
        grad_max = tensor_max_abs(grad) if grad_ok else math.nan
        if grad_ok:
            max_grad_norm = max(max_grad_norm, grad_norm)

        with torch.no_grad():
            x_base = ddpm_step(x_t, eps_pred, t_idx, diffusion)
            base_ok = tensor_isfinite(x_base)
            x_base_norm = tensor_norm(x_base) if base_ok else math.nan
            x_base_max = tensor_max_abs(x_base) if base_ok else math.nan
            update = float(zeta) * grad
            update_ok = tensor_isfinite(update)
            update_norm = tensor_norm(update) if update_ok else math.nan
            update_max = tensor_max_abs(update) if update_ok else math.nan
            update_to_x = update_norm / (x_base_norm + EPS) if update_ok and base_ok else math.nan
            if np.isfinite(update_to_x):
                max_update_to_x_ratio = max(max_update_to_x_ratio, float(update_to_x))
            x_after = x_base - update
            after_ok = tensor_isfinite(x_after)
            x_after_max = tensor_max_abs(x_after) if after_ok else math.nan
            x_after_norm = tensor_norm(x_after) if after_ok else math.nan
            if np.isfinite(x_after_max):
                max_x_after_max_abs = max(max_x_after_max_abs, float(x_after_max))

        stage = stage_name(before_ok, loss_ok, grad_ok, base_ok, update_ok, after_ok)
        rows.append(
            {
                "condition": condition,
                "zeta": zeta,
                "seed": SEED,
                "trajectory_id": traj_idx,
                "t": t_idx,
                "loss_mean": float(loss.detach().cpu()) if loss_ok else math.nan,
                "loss_isfinite": loss_ok,
                "grad_norm": grad_norm,
                "grad_max_abs": grad_max,
                "grad_isfinite": grad_ok,
                "x_base_norm_after_ddpm": x_base_norm,
                "x_base_max_abs_after_ddpm": x_base_max,
                "x_base_isfinite_after_ddpm": base_ok,
                "update_norm": update_norm,
                "update_max_abs": update_max,
                "update_to_x_ratio": update_to_x,
                "x_after_norm": x_after_norm,
                "x_after_max_abs": x_after_max,
                "x_after_isfinite": after_ok,
                "first_nonfinite_stage": stage,
            }
        )
        if stage != "none":
            first_nonfinite_step = t_idx
            first_nonfinite_stage = stage
            final_pred = None
            break
        x_t = x_after.detach()

    finite = first_nonfinite_step is None
    ade = math.nan
    acc_rms = math.nan
    if finite:
        final_pred = final_abs_from_latent(
            x_t,
            tensors["degraded_rel_norm_np"],
            tensors["y_np"],
            rel_mean,
            rel_std,
        )
        clean = gt[traj_idx]
        err = np.linalg.norm(final_pred - clean, axis=-1)
        ade = float(err.mean())
        acc_rms = acceleration_rms(final_pred)

    summary = {
        "condition": condition,
        "zeta": zeta,
        "seed": SEED,
        "trajectory_id": traj_idx,
        "finite": finite,
        "first_nonfinite_step": first_nonfinite_step,
        "first_nonfinite_stage": first_nonfinite_stage,
        "max_grad_norm": max_grad_norm,
        "max_update_to_x_ratio": max_update_to_x_ratio,
        "max_x_after_max_abs": max_x_after_max_abs,
        "ADE": ade,
        "acceleration_RMS": acc_rms,
    }
    return rows, summary


def write_summary(trace_df: pd.DataFrame, case_df: pd.DataFrame) -> None:
    all_finite = bool(case_df["finite"].all())
    max_grad = float(case_df["max_grad_norm"].replace([np.inf, -np.inf], np.nan).max())
    max_update = float(case_df["max_update_to_x_ratio"].replace([np.inf, -np.inf], np.nan).max())
    max_x = float(case_df["max_x_after_max_abs"].replace([np.inf, -np.inf], np.nan).max())
    by_zeta = case_df.groupby("zeta")["finite"].all().to_dict()
    nonfinite = case_df[case_df["finite"] == False]
    if nonfinite.empty:
        first_text = "No non-finite event observed in the mean-form re-audit subset."
    else:
        first = nonfinite.sort_values("first_nonfinite_step", ascending=False).iloc[0]
        first_text = (
            f"First non-finite observed at condition={first['condition']}, "
            f"zeta={first['zeta']}, trajectory={int(first['trajectory_id'])}, "
            f"step={int(first['first_nonfinite_step'])}, stage={first['first_nonfinite_stage']}."
        )
    reasonable = bool(np.isfinite(max_update) and max_update < 10.0)
    clipping_needed = "No clipping or adaptive guidance is needed for this small re-audit before rerunning Stage 8A." if all_finite else (
        "Mean-form alone did not eliminate non-finite behavior; any clipping or adaptive guidance would require another amendment."
    )
    recommendation = (
        "Yes. Request user approval to rerun the full Stage 8A pilot under Amendment 001."
        if all_finite
        else "No. Do not rerun full Stage 8A until the remaining numerical issue is audited."
    )

    lines = [
        "# E2-DPS Mean-Form Likelihood Re-Audit Summary",
        "",
        f"Pre-registration: {PRE_REG_PATH}",
        f"Amendment: {AMENDMENT_PATH}",
        "",
        "## Scope",
        "This is a very small numerical re-audit, not a formal Stage 8A pilot result and not Stage 8B.",
        "",
        "## Re-Audit Grid",
        f"- conditions: {', '.join(CONDITIONS)}",
        f"- zeta: {ZETAS}",
        f"- seed: {SEED}",
        f"- trajectories: {TRAJECTORIES}",
        "",
        "## Main Results",
        f"- mean-form likelihood eliminated NaN in this subset: {all_finite}",
        f"- finite by zeta: {by_zeta}",
        f"- {first_text}",
        f"- max grad_norm: {max_grad:.6g}",
        f"- max update_to_x_ratio: {max_update:.6g}",
        f"- max x_after_max_abs: {max_x:.6g}",
        f"- update_to_x_ratio in a reasonable range: {reasonable}",
        f"- clipping / adaptive guidance assessment: {clipping_needed}",
        f"- full Stage 8A recommendation: {recommendation}",
        "",
        "## Required Answers",
        f"1. Mean-form likelihood eliminated NaN: {all_finite}",
        f"2. zeta=0.3/1.0/3.0 finite: {by_zeta}",
        f"3. update_to_x_ratio returned to a bounded range: {reasonable} (max={max_update:.6g})",
        f"4. Need clipping / adaptive guidance now: {'no' if all_finite else 'not allowed without another amendment'}",
        f"5. Can request approval to rerun full Stage 8A under Amendment 001: {all_finite}",
        "",
        "## Output Files",
        f"- trace: {TRACE_PATH}",
        f"- case summary: {CASE_SUMMARY_PATH}",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print_file(SUMMARY_PATH)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PRE_REG_PATH.is_file():
        raise FileNotFoundError(PRE_REG_PATH)
    if not AMENDMENT_PATH.is_file():
        raise FileNotFoundError(AMENDMENT_PATH)

    _, confs = load_confidence_cache()
    _, data, gt = load_inputs()
    model, diffusion, rel_mean, rel_std = load_model()

    trace_rows: list[dict] = []
    case_rows: list[dict] = []
    for condition in CONDITIONS:
        for zeta in ZETAS:
            for traj_idx in TRAJECTORIES:
                rows, summary = run_case(
                    model,
                    diffusion,
                    data,
                    confs,
                    gt,
                    rel_mean,
                    rel_std,
                    condition,
                    zeta,
                    traj_idx,
                )
                trace_rows.extend(rows)
                case_rows.append(summary)
                print(
                    f"[DONE] mean-form reaudit condition={condition} zeta={zeta} "
                    f"seed={SEED} traj={traj_idx} finite={summary['finite']}"
                )

    trace_df = pd.DataFrame(trace_rows)
    case_df = pd.DataFrame(case_rows)
    trace_df.to_csv(TRACE_PATH, index=False)
    print_file(TRACE_PATH)
    case_df.to_csv(CASE_SUMMARY_PATH, index=False)
    print_file(CASE_SUMMARY_PATH)
    write_summary(trace_df, case_df)

    print("E2_DPS_MEAN_LIKELIHOOD_REAUDIT_COMPLETE")
    print(f"all_finite: {bool(case_df['finite'].all())}")
    print(f"max_grad_norm: {case_df['max_grad_norm'].replace([np.inf, -np.inf], np.nan).max():.6g}")
    print(f"max_update_to_x_ratio: {case_df['max_update_to_x_ratio'].replace([np.inf, -np.inf], np.nan).max():.6g}")
    print(f"amendment: {AMENDMENT_PATH}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
