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
    to_rel,
    x0_hat_abs_from_xt,
)


AMENDMENT_001_PATH = PROJECT_ROOT / "docs" / "stage4" / "E2_DPS_hypothesis_amendment_001.md"
AMENDMENT_002_PATH = PROJECT_ROOT / "docs" / "stage4" / "E2_DPS_hypothesis_amendment_002.md"
OUT_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e2_dps_norm_guidance_reaudit"
TRACE_PATH = OUT_DIR / "e2_dps_norm_guidance_reaudit_trace.csv"
CASE_SUMMARY_PATH = OUT_DIR / "e2_dps_norm_guidance_reaudit_case_summary.csv"
SUMMARY_PATH = OUT_DIR / "e2_dps_norm_guidance_reaudit_summary.md"

CONDITIONS = ["drift_medium", "burst_medium", "gaussian_medium", "bias_medium"]
ZETAS = [0.3, 1.0, 3.0]
SEED = 42
TRAJECTORIES = [0, 1, 2]
SNAPSHOT_STEPS = [99, 80, 60, 40, 20, 0]
EPS = 1e-12


def print_file(path: Path) -> None:
    print(f"[FILE] {path} written")


def tensor_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach()).cpu())


def tensor_max_abs(x: torch.Tensor) -> float:
    return float(torch.max(torch.abs(x.detach())).cpu())


def tensor_isfinite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def canonical_weighted_norm_loss(
    abs_hat: torch.Tensor,
    y_t: torch.Tensor,
    conf_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma2 = SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_t))
    sigma = torch.sqrt(sigma2)
    residual = abs_hat - y_t
    weighted = residual / sigma[..., None]
    weighted_norm = torch.linalg.vector_norm(weighted)
    unweighted_norm = torch.linalg.vector_norm(residual)
    return weighted_norm, weighted_norm, unweighted_norm, sigma


def acceleration_rms(pred: np.ndarray) -> float:
    acc = pred[2:, :] - 2.0 * pred[1:-1, :] + pred[:-2, :]
    return float(np.sqrt(np.mean(np.sum(acc**2, axis=-1))))


def reconstruct_from_rel(start_point: np.ndarray, rel: np.ndarray) -> np.ndarray:
    abs_hat = np.zeros((20, 2), dtype=np.float32)
    abs_hat[0] = start_point.astype(np.float32)
    abs_hat[1:] = start_point.astype(np.float32)[None, :] + np.cumsum(rel.astype(np.float32), axis=0)
    return abs_hat


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


def anchor_thresholds(data: dict[str, dict], gt: np.ndarray) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for condition in CONDITIONS:
        errors = np.linalg.norm(data[condition]["degraded"][:, 0, :] - gt[:, 0, :], axis=-1)
        q25, q75 = np.percentile(errors, [25, 75])
        iqr = q75 - q25
        thresholds[condition] = float(max(q75 + 1.5 * iqr, 3.0 * np.median(errors), 1e-6))
    return thresholds


def run_case(
    model,
    diffusion,
    data: dict[str, dict],
    confs: dict[str, np.ndarray],
    gt: np.ndarray,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    anchor_threshold_by_condition: dict[str, float],
    condition: str,
    zeta: float,
    traj_idx: int,
) -> tuple[list[dict], dict]:
    tensors = setup_case(data, confs, rel_mean, rel_std, condition, traj_idx)
    y_np = tensors["y_np"]
    conf_np = tensors["conf_np"]
    clean = gt[traj_idx]
    anchor_y0 = y_np[0, 0, :].astype(np.float32)
    anchor_error = float(np.linalg.norm(anchor_y0 - clean[0]))
    anchor_threshold = anchor_threshold_by_condition[condition]
    anchor_obvious_anomaly = bool(anchor_error > anchor_threshold)
    sigma_np = np.sqrt(SIGMA0**2 * (1.0 + KAPPA * (1.0 - conf_np)))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_t = torch.randn((1, 2, 19), device=DEVICE, dtype=torch.float32)
    rows: list[dict] = []
    snapshots = {step: math.nan for step in SNAPSHOT_STEPS}
    first_nonfinite_step = None
    first_nonfinite_stage = "none"
    max_grad_norm = 0.0
    max_update_to_x_ratio = 0.0
    max_x_after_max_abs = 0.0

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
        loss, weighted_residual_norm, unweighted_residual_norm, sigma_t = canonical_weighted_norm_loss(
            abs_hat,
            tensors["y_t"],
            tensors["conf_t"],
        )
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
            if t_idx in snapshots:
                snapshots[t_idx] = float(update_to_x) if np.isfinite(update_to_x) else math.nan
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
                "finite": stage == "none",
                "first_nonfinite_stage": stage,
                "L_norm": float(loss.detach().cpu()) if loss_ok else math.nan,
                "weighted_residual_norm": float(weighted_residual_norm.detach().cpu())
                if bool(torch.isfinite(weighted_residual_norm).detach().cpu())
                else math.nan,
                "unweighted_residual_norm": float(unweighted_residual_norm.detach().cpu())
                if bool(torch.isfinite(unweighted_residual_norm).detach().cpu())
                else math.nan,
                "sigma_obs_min": float(sigma_t.detach().min().cpu()) if tensor_isfinite(sigma_t) else math.nan,
                "sigma_obs_mean": float(sigma_t.detach().mean().cpu()) if tensor_isfinite(sigma_t) else math.nan,
                "sigma_obs_max": float(sigma_t.detach().max().cpu()) if tensor_isfinite(sigma_t) else math.nan,
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
                "anchor_y0_x": float(anchor_y0[0]),
                "anchor_y0_y": float(anchor_y0[1]),
                "anchor_error": anchor_error,
                "anchor_anomaly_threshold": anchor_threshold,
                "anchor_obvious_anomaly": anchor_obvious_anomaly,
            }
        )
        if stage != "none":
            first_nonfinite_step = t_idx
            first_nonfinite_stage = stage
            break
        x_t = x_after.detach()

    finite = first_nonfinite_step is None
    ade = math.nan
    acc_rms = math.nan
    if finite:
        final_pred = final_abs_from_latent(
            x_t,
            tensors["degraded_rel_norm_np"],
            y_np,
            rel_mean,
            rel_std,
        )
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
        "weighted_residual_norm_last": rows[-1]["weighted_residual_norm"] if rows else math.nan,
        "unweighted_residual_norm_last": rows[-1]["unweighted_residual_norm"] if rows else math.nan,
        "sigma_obs_min": float(sigma_np.min()),
        "sigma_obs_mean": float(sigma_np.mean()),
        "sigma_obs_max": float(sigma_np.max()),
        "anchor_y0_x": float(anchor_y0[0]),
        "anchor_y0_y": float(anchor_y0[1]),
        "anchor_error": anchor_error,
        "anchor_anomaly_threshold": anchor_threshold,
        "anchor_obvious_anomaly": anchor_obvious_anomaly,
    }
    for step in SNAPSHOT_STEPS:
        summary[f"update_to_x_ratio_t{step}"] = snapshots[step]
    return rows, summary


def classify_result(case_df: pd.DataFrame) -> tuple[str, str]:
    all_finite = bool(case_df["finite"].all())
    max_update = float(case_df["max_update_to_x_ratio"].replace([np.inf, -np.inf], np.nan).max())
    nonfinite = case_df[case_df["finite"] == False]
    if all_finite and max_update < 5.0:
        return "Case A", "All cases are finite and max update_to_x_ratio < 5. Recommend approval to rerun full Stage 8A pilot."
    if all_finite and max_update <= 50.0:
        return "Case B", "All cases are finite but update_to_x_ratio is 5-50. Stage 8A may proceed with overshoot caution; if ADE fails, consider timestep schedule rather than trust-region."
    if not nonfinite.empty:
        concentrated = set(nonfinite["condition"]).issubset({"burst_medium", "jump_medium"})
        anchor_related = bool(nonfinite["anchor_obvious_anomaly"].any())
        if concentrated and anchor_related:
            return "Case C", "Partial finite behavior with NaN concentrated in burst/jump and anchor anomalies. Consider Amendment 003 robust anchor; do not implement it now."
    return "Case D", "Norm-based form still has substantial NaN or excessive update scale. Consider trust-region/adaptive guidance or stopping E2-DPS, but only via a later amendment."


def write_summary(trace_df: pd.DataFrame, case_df: pd.DataFrame) -> None:
    all_finite = bool(case_df["finite"].all())
    by_zeta = case_df.groupby("zeta")["finite"].agg(["sum", "count", "all"]).reset_index()
    max_grad = float(case_df["max_grad_norm"].replace([np.inf, -np.inf], np.nan).max())
    max_update = float(case_df["max_update_to_x_ratio"].replace([np.inf, -np.inf], np.nan).max())
    late_cols = [f"update_to_x_ratio_t{step}" for step in [40, 20, 0]]
    late_max = float(case_df[late_cols].replace([np.inf, -np.inf], np.nan).max().max())
    late_overshoot = bool(np.isfinite(late_max) and late_max > 5.0)
    anchor_anomalies = case_df[case_df["anchor_obvious_anomaly"] == True]
    nonfinite = case_df[case_df["finite"] == False]
    if nonfinite.empty:
        first_text = "No non-finite event observed in the norm-based re-audit subset."
    else:
        first = nonfinite.sort_values("first_nonfinite_step", ascending=False).iloc[0]
        first_text = (
            f"First non-finite observed at condition={first['condition']}, "
            f"zeta={first['zeta']}, trajectory={int(first['trajectory_id'])}, "
            f"step={int(first['first_nonfinite_step'])}, stage={first['first_nonfinite_stage']}."
        )
    case_label, case_text = classify_result(case_df)
    can_request_stage8a = case_label in {"Case A", "Case B"}

    lines = [
        "# E2-DPS Norm-Based Guidance Re-Audit Summary",
        "",
        f"Pre-registration: {PRE_REG_PATH}",
        f"Amendment 001: {AMENDMENT_001_PATH}",
        f"Amendment 002: {AMENDMENT_002_PATH}",
        "",
        "## Scope",
        "This is a very small numerical re-audit for Amendment 002. It is not a full Stage 8A pilot and not Stage 8B.",
        "",
        "## Re-Audit Grid",
        f"- conditions: {', '.join(CONDITIONS)}",
        f"- zeta: {ZETAS}",
        f"- seed: {SEED}",
        f"- trajectories: {TRAJECTORIES}",
        "",
        "## Main Results",
        f"- norm-based DPS guidance eliminated NaN in this subset: {all_finite}",
        f"- finite counts by zeta:\n{by_zeta.to_string(index=False)}",
        f"- {first_text}",
        f"- max grad_norm: {max_grad:.6g}",
        f"- max update_to_x_ratio: {max_update:.6g}",
        f"- late-step max update_to_x_ratio over t=40/20/0 snapshots: {late_max:.6g}",
        f"- late-step overshoot risk: {late_overshoot}",
        f"- anchor_y0 obvious anomaly count: {len(anchor_anomalies)} / {len(case_df)}",
        f"- non-finite cases: {len(nonfinite)} / {len(case_df)}",
        f"- result class: {case_label}",
        f"- classification rationale: {case_text}",
        "",
        "## Required Answers",
        f"1. Norm-based DPS guidance eliminated NaN: {all_finite}",
        f"2. zeta=0.3/1.0/3.0 finite counts: {by_zeta.to_dict(orient='records')}",
        f"3. update_to_x_ratio returned to a reasonable range: {bool(np.isfinite(max_update) and max_update < 5.0)}",
        f"4. Late-step overshoot risk: {late_overshoot}",
        f"5. NaN concentrated in burst/jump or anchor anomaly cases: {bool((not nonfinite.empty) and set(nonfinite['condition']).issubset({'burst_medium', 'jump_medium'}) and nonfinite['anchor_obvious_anomaly'].any())}",
        f"6. Still need trust-region / clipping: {'yes, only via a later amendment' if case_label == 'Case D' else 'not for this re-audit classification'}",
        f"7. Can request user approval to rerun full Stage 8A under Amendment 002: {can_request_stage8a}",
        "",
        "## Anchor Diagnostics",
    ]
    if anchor_anomalies.empty:
        lines.append("- No obvious anchor_y0 anomaly was detected under the condition-wise IQR/median diagnostic threshold.")
    else:
        for _, row in anchor_anomalies.iterrows():
            lines.append(
                f"- {row['condition']} traj={int(row['trajectory_id'])}: "
                f"anchor_error={row['anchor_error']:.6g}, threshold={row['anchor_anomaly_threshold']:.6g}"
            )
    lines.extend(
        [
            "",
            "## Output Files",
            f"- trace: {TRACE_PATH}",
            f"- case summary: {CASE_SUMMARY_PATH}",
        ]
    )
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
    anchor_threshold_by_condition = anchor_thresholds(data, gt)

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
                    anchor_threshold_by_condition,
                    condition,
                    zeta,
                    traj_idx,
                )
                trace_rows.extend(rows)
                case_rows.append(summary)
                print(
                    f"[DONE] norm-guidance reaudit condition={condition} zeta={zeta} "
                    f"seed={SEED} traj={traj_idx} finite={summary['finite']}"
                )

    trace_df = pd.DataFrame(trace_rows)
    case_df = pd.DataFrame(case_rows)
    trace_df.to_csv(TRACE_PATH, index=False)
    print_file(TRACE_PATH)
    case_df.to_csv(CASE_SUMMARY_PATH, index=False)
    print_file(CASE_SUMMARY_PATH)
    write_summary(trace_df, case_df)

    case_label, _ = classify_result(case_df)
    finite_by_zeta = case_df.groupby("zeta")["finite"].agg(["sum", "count", "all"]).reset_index()
    print("E2_DPS_NORM_GUIDANCE_REAUDIT_COMPLETE")
    print(f"all_finite: {bool(case_df['finite'].all())}")
    print(f"finite_by_zeta: {finite_by_zeta.to_dict(orient='records')}")
    print(f"max_grad_norm: {case_df['max_grad_norm'].replace([np.inf, -np.inf], np.nan).max():.6g}")
    print(f"max_update_to_x_ratio: {case_df['max_update_to_x_ratio'].replace([np.inf, -np.inf], np.nan).max():.6g}")
    print(f"case: {case_label}")
    print(f"amendment: {AMENDMENT_002_PATH}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
