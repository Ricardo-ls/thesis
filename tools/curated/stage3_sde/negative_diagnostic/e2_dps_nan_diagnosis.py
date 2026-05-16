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
    DEVICE,
    KAPPA,
    PRE_REG_PATH,
    SIGMA0,
    TIMESTEPS,
    ZETAS,
    ddpm_step,
    load_confidence_cache,
    load_inputs,
    load_model,
    normalize_rel,
    to_rel,
    x0_hat_abs_from_xt,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_dps_nan_diagnosis"
FIG_DIR = OUT_DIR / "figures"
TRACE_PATH = OUT_DIR / "e2_dps_nan_trace.csv"
CASE_SUMMARY_PATH = OUT_DIR / "e2_dps_nan_case_summary.csv"
SUMMARY_PATH = OUT_DIR / "e2_dps_nan_diagnosis_summary.md"

CONDITIONS = ["drift_medium", "burst_medium", "gaussian_medium", "bias_medium"]
DIAG_ZETAS = [0.3, 1.0, 3.0]
SEED = 42
TRAJ_CANDIDATES = [0, 1, 2, 3, 4]
EPS = 1e-12


def tensor_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach()).cpu())


def tensor_max_abs(x: torch.Tensor) -> float:
    return float(torch.max(torch.abs(x.detach())).cpu())


def tensor_isfinite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def loss_sum_and_mean(abs_hat: torch.Tensor, y_t: torch.Tensor, conf_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sigma2 = SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t))
    err2 = torch.sum((abs_hat - y_t) ** 2, dim=-1)
    terms = err2 / sigma2
    return torch.sum(terms), torch.mean(terms)


def first_stage(
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


def setup_case(
    data: dict[str, dict],
    confs: dict[str, np.ndarray],
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    condition: str,
    traj_idx: int,
):
    y = data[condition]["degraded"][traj_idx : traj_idx + 1]
    conf = confs[condition][traj_idx : traj_idx + 1]
    degraded_rel = to_rel(y)
    degraded_rel_norm = normalize_rel(degraded_rel, rel_mean, rel_std)
    tensors = {
        "x_cond": torch.from_numpy(degraded_rel_norm.transpose(0, 2, 1)).to(DEVICE, dtype=torch.float32),
        "degraded_rel_norm_t": torch.from_numpy(degraded_rel_norm).to(DEVICE, dtype=torch.float32),
        "y_t": torch.from_numpy(y).to(DEVICE, dtype=torch.float32),
        "conf_t": torch.from_numpy(conf).to(DEVICE, dtype=torch.float32),
        "start_t": torch.from_numpy(y[:, 0, :]).to(DEVICE, dtype=torch.float32),
        "rel_mean_t": torch.from_numpy(rel_mean).to(DEVICE, dtype=torch.float32),
        "rel_std_t": torch.from_numpy(rel_std).to(DEVICE, dtype=torch.float32),
    }
    return tensors


def run_trace(
    model,
    diffusion,
    data: dict[str, dict],
    confs: dict[str, np.ndarray],
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    condition: str,
    zeta: float,
    traj_idx: int,
    no_guidance: bool = False,
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
    zeta_effective = 0.0 if no_guidance else float(zeta)

    for t_idx in reversed(range(TIMESTEPS)):
        x_t = x_t.detach().requires_grad_(True)
        before_ok = tensor_isfinite(x_t)
        x_before_norm = tensor_norm(x_t)
        x_before_max = tensor_max_abs(x_t)
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
        loss_sum, loss_mean = loss_sum_and_mean(abs_hat, tensors["y_t"], tensors["conf_t"])
        loss_ok = bool(torch.isfinite(loss_sum).detach().cpu())
        if loss_ok:
            grad = torch.autograd.grad(loss_sum, x_t, retain_graph=False)[0]
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
            update = zeta_effective * grad
            update_ok = tensor_isfinite(update)
            update_norm = tensor_norm(update) if update_ok else math.nan
            update_max = tensor_max_abs(update) if update_ok else math.nan
            update_to_x = update_norm / (x_base_norm + EPS) if update_ok and base_ok else math.nan
            if np.isfinite(update_to_x):
                max_update_to_x_ratio = max(max_update_to_x_ratio, float(update_to_x))
            x_after = x_base - update
            after_ok = tensor_isfinite(x_after)
            x_after_norm = tensor_norm(x_after) if after_ok else math.nan
            x_after_max = tensor_max_abs(x_after) if after_ok else math.nan

        stage = first_stage(before_ok, loss_ok, grad_ok, base_ok, update_ok, after_ok)
        rows.append(
            {
                "condition": condition,
                "zeta": zeta,
                "zeta_effective": zeta_effective,
                "seed": SEED,
                "trajectory_id": traj_idx,
                "no_guidance": bool(no_guidance),
                "t": t_idx,
                "loss_likelihood": float(loss_sum.detach().cpu()) if loss_ok else math.nan,
                "loss_mean_form": float(loss_mean.detach().cpu()) if bool(torch.isfinite(loss_mean).detach().cpu()) else math.nan,
                "loss_sum_form": float(loss_sum.detach().cpu()) if loss_ok else math.nan,
                "loss_sum_over_mean_ratio": float((loss_sum / (loss_mean + EPS)).detach().cpu())
                if loss_ok and bool(torch.isfinite(loss_mean).detach().cpu())
                else math.nan,
                "loss_isfinite": loss_ok,
                "grad_norm": grad_norm,
                "grad_max_abs": grad_max,
                "grad_isfinite": grad_ok,
                "x_t_norm_before_ddpm": x_before_norm,
                "x_t_max_abs_before_ddpm": x_before_max,
                "x_t_isfinite_before_ddpm": before_ok,
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
            break
        x_t = x_after.detach()

    summary = {
        "condition": condition,
        "zeta": zeta,
        "zeta_effective": zeta_effective,
        "seed": SEED,
        "trajectory_id": traj_idx,
        "no_guidance": bool(no_guidance),
        "first_nonfinite_step": first_nonfinite_step,
        "first_nonfinite_stage": first_nonfinite_stage,
        "max_grad_norm": max_grad_norm,
        "max_update_to_x_ratio": max_update_to_x_ratio,
        "finite": first_nonfinite_step is None,
    }
    return rows, summary


def make_figures(trace_df: pd.DataFrame) -> list[dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    finite = trace_df[trace_df["no_guidance"] == False]
    for condition in CONDITIONS:
        sub = finite[(finite["condition"] == condition) & (finite["trajectory_id"] == 0) & (finite["zeta"] == 0.3)]
        if sub.empty:
            continue
        path = FIG_DIR / f"{condition}_traj0_zeta0p3_trace.png"
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        x = sub["t"].to_numpy()
        axes[0].plot(x, sub["grad_norm"], "C0-")
        axes[0].set_title("grad_norm")
        axes[1].plot(x, sub["update_to_x_ratio"], "C1-")
        axes[1].set_title("update_to_x_ratio")
        axes[2].plot(x, sub["x_after_max_abs"], "C2-")
        axes[2].set_title("x_after_max_abs")
        for ax in axes:
            ax.invert_xaxis()
            ax.grid(alpha=0.25)
            ax.set_xlabel("reverse step t")
        fig.suptitle(f"{condition} traj0 zeta=0.3")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        rows.append({"condition": condition, "trajectory_id": 0, "zeta": 0.3, "path": str(path)})
    return rows


def write_summary(trace_df: pd.DataFrame, case_df: pd.DataFrame, figure_rows: list[dict]) -> None:
    guided = case_df[case_df["no_guidance"] == False]
    no_guided = case_df[case_df["no_guidance"] == True]
    nonfinite = guided[guided["finite"] == False]
    if nonfinite.empty:
        first_text = "No non-finite event observed in diagnostic cases."
        earliest = None
    else:
        # Larger t happens earlier in the reverse loop.
        earliest = nonfinite.sort_values("first_nonfinite_step", ascending=False).iloc[0]
        first_text = (
            f"First non-finite observed at condition={earliest['condition']}, "
            f"zeta={earliest['zeta']}, trajectory={int(earliest['trajectory_id'])}, "
            f"step={int(earliest['first_nonfinite_step'])}, stage={earliest['first_nonfinite_stage']}."
        )
    zeta0_finite_all = bool(no_guided["finite"].all())
    guided_finite_all = bool(guided["finite"].all())
    max_update = float(guided["max_update_to_x_ratio"].replace([np.inf, -np.inf], np.nan).max())
    max_grad = float(guided["max_grad_norm"].replace([np.inf, -np.inf], np.nan).max())
    t99 = trace_df[(trace_df["no_guidance"] == False) & (trace_df["t"] == 99)]
    loss_ratio = float(t99["loss_sum_over_mean_ratio"].dropna().median()) if not t99.empty else math.nan
    stage_counts = guided["first_nonfinite_stage"].value_counts().to_dict()

    if not zeta0_finite_all:
        problem = "DDPM reverse step / input representation / timestep schedule"
        recommendation = "Do not continue Stage 8A until the no-guidance DDPM path is finite."
    elif not guided_finite_all:
        problem = "guidance update scale / pre-registered hyperparameter instability"
        recommendation = "Do not continue Stage 8A as interpretable. If changing normalization, scaling, or clipping is needed, write a pre-registration amendment or audit finding first."
    else:
        problem = "no numerical instability observed in diagnostic subset"
        recommendation = "Stage 8A may be rerun only if this diagnostic subset is considered sufficient."

    lines = [
        "# E2-DPS NaN Diagnosis Summary",
        "",
        f"Pre-registration: {PRE_REG_PATH}",
        "",
        "## Main Findings",
        f"- {first_text}",
        f"- zeta=0 no-guidance finite for all checked cases: {zeta0_finite_all}",
        f"- guided finite for all checked cases: {guided_finite_all}",
        f"- max grad_norm observed: {max_grad:.6g}",
        f"- max update_to_x_ratio observed: {max_update:.6g}",
        f"- median t=99 loss_sum / loss_mean ratio: {loss_ratio:.6g}",
        f"- first_nonfinite_stage counts: {stage_counts}",
        f"- diagnosis: {problem}",
        f"- recommendation: {recommendation}",
        "",
        "## Interpretation",
        "This diagnostic does not modify the formal E2-DPS method. No clipping, robust likelihood, altered sigma, altered kappa, or expanded zeta was introduced.",
        "",
        "If the diagnosis points to scale mismatch, Stage 8A should not be treated as a PASS/FAIL scientific result. Any method change such as loss normalization, guidance scaling, or clipping requires a pre-registration amendment / audit finding before rerunning.",
        "",
        "## Figures",
    ]
    for row in figure_rows:
        lines.append(f"- {row['condition']} traj {row['trajectory_id']} zeta={row['zeta']}: {row['path']}")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {SUMMARY_PATH} written")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PRE_REG_PATH.is_file():
        raise FileNotFoundError(PRE_REG_PATH)
    _, confs = load_confidence_cache()
    _, data, _ = load_inputs()
    model, diffusion, rel_mean, rel_std = load_model()

    all_rows: list[dict] = []
    case_rows: list[dict] = []
    for condition in CONDITIONS:
        for traj_idx in TRAJ_CANDIDATES:
            ng_rows, ng_summary = run_trace(
                model, diffusion, data, confs, rel_mean, rel_std, condition, zeta=0.0, traj_idx=traj_idx, no_guidance=True
            )
            all_rows.extend(ng_rows)
            zeta0_finite = bool(ng_summary["finite"])
            for zeta in DIAG_ZETAS:
                rows, summary = run_trace(
                    model, diffusion, data, confs, rel_mean, rel_std, condition, zeta=zeta, traj_idx=traj_idx, no_guidance=False
                )
                all_rows.extend(rows)
                summary["zeta0_finite"] = zeta0_finite
                summary["zeta_guided_finite"] = bool(summary["finite"])
                case_rows.append(summary)
            # If traj0 already shows non-finite for this condition, move on.
            if traj_idx == 0 and any(not row["zeta_guided_finite"] for row in case_rows if row["condition"] == condition and row["trajectory_id"] == traj_idx):
                break

    trace_df = pd.DataFrame(all_rows)
    case_df = pd.DataFrame(case_rows)
    trace_df.to_csv(TRACE_PATH, index=False)
    print(f"[FILE] {TRACE_PATH} written")
    case_df.to_csv(CASE_SUMMARY_PATH, index=False)
    print(f"[FILE] {CASE_SUMMARY_PATH} written")
    figure_rows = make_figures(trace_df)
    pd.DataFrame(figure_rows).to_csv(OUT_DIR / "e2_dps_nan_diagnosis_figures.csv", index=False)
    print(f"[FILE] {OUT_DIR / 'e2_dps_nan_diagnosis_figures.csv'} written")
    write_summary(trace_df, case_df, figure_rows)

    guided = case_df[case_df["no_guidance"] == False]
    no_guided_all = bool(case_df["zeta0_finite"].all())
    nonfinite = guided[guided["zeta_guided_finite"] == False]
    print("E2_DPS_NAN_DIAGNOSIS_COMPLETE")
    print(f"zeta0_no_guidance_all_finite: {no_guided_all}")
    if nonfinite.empty:
        print("first_nonfinite: NONE")
    else:
        first = nonfinite.sort_values("first_nonfinite_step", ascending=False).iloc[0]
        print(
            "first_nonfinite: "
            f"condition={first['condition']} zeta={first['zeta']} traj={int(first['trajectory_id'])} "
            f"step={int(first['first_nonfinite_step'])} stage={first['first_nonfinite_stage']}"
        )
    print(f"max_grad_norm: {guided['max_grad_norm'].max():.6g}")
    print(f"max_update_to_x_ratio: {guided['max_update_to_x_ratio'].max():.6g}")
    print(f"trace: {TRACE_PATH}")
    print(f"case_summary: {CASE_SUMMARY_PATH}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
