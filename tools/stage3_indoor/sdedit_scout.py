from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D


CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42"
CSV_PATH = OUTPUT_DIR / "sdedit_scout_results.csv"
FIG_PATH = OUTPUT_DIR / "sdedit_scout.png"

N = 100
TIMESTEPS = 100
T_LIST = [1, 2, 3, 5]
SDEDIT_SEEDS = [42, 43, 44]
METHODS = ["noisy_input", "sdedit_t1", "sdedit_t2", "sdedit_t3", "sdedit_t5"]
DEGRADATIONS = ["gaussian_medium", "burst_medium"]


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def compute_metrics(pred: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errors = np.linalg.norm(pred - clean, axis=-1)
    ade = errors.mean(axis=1).astype(np.float32)
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    smooth = np.linalg.norm(acc, axis=-1).mean(axis=1).astype(np.float32)
    return ade, smooth


def generate_gaussian(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        rng = np.random.default_rng(42 + idx)
        noise = rng.normal(0.0, 0.05, size=(20, 2)).astype(np.float32)
        degraded[idx] = clean[idx] + noise
    return degraded


def generate_burst(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        rng = np.random.default_rng(42 + idx)
        burst_len = int(rng.choice([3, 4, 5]))
        burst_start = int(rng.integers(0, 20 - burst_len + 1))
        noise = rng.normal(0.0, 0.01, size=(20, 2)).astype(np.float32)
        noise[burst_start:burst_start + burst_len] = rng.normal(
            0.0, 0.25, size=(burst_len, 2)
        ).astype(np.float32)
        degraded[idx] = clean[idx] + noise
    return degraded


def reconstruct_from_rel(start_points: np.ndarray, rel: np.ndarray) -> np.ndarray:
    abs_hat = np.zeros((rel.shape[0], 20, 2), dtype=np.float32)
    abs_hat[:, 0, :] = start_points.astype(np.float32)
    abs_hat[:, 1:, :] = start_points[:, None, :] + np.cumsum(rel, axis=1)
    return abs_hat


def run_sdedit(
    degraded_abs: np.ndarray,
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
    rel_mean: np.ndarray,
    rel_std: np.ndarray,
    t_start: int,
    sdedit_seed: int,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(sdedit_seed)
    np.random.seed(sdedit_seed)

    degraded_rel = (degraded_abs[:, 1:, :] - degraded_abs[:, :-1, :]).astype(np.float32)
    rel_norm = ((degraded_rel - rel_mean[None, None, :]) / rel_std[None, None, :]).astype(np.float32)
    x0 = torch.from_numpy(rel_norm.transpose(0, 2, 1)).to(device=device, dtype=torch.float32)

    t = torch.full((x0.shape[0],), t_start, device=device, dtype=torch.long)
    x_t, _ = diffusion.q_sample(x0, t)

    with torch.no_grad():
        for t_idx in reversed(range(t_start + 1)):
            t_cur = torch.full((x_t.shape[0],), t_idx, device=device, dtype=torch.long)
            eps_pred = model(x_t, t_cur)
            alpha_t = diffusion.alphas[t_idx]
            alpha_bar_t = diffusion.alpha_bars[t_idx]
            beta_t = diffusion.betas[t_idx]
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
            )
            if t_idx > 0:
                z = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * z
            else:
                x_t = mean

    rel_hat_norm = x_t.permute(0, 2, 1).cpu().numpy().astype(np.float32)
    rel_hat = (rel_hat_norm * rel_std[None, None, :] + rel_mean[None, None, :]).astype(np.float32)
    return reconstruct_from_rel(degraded_abs[:, 0, :], rel_hat)


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {CHECKPOINT_PATH}")
    if not NORM_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {NORM_PATH}")
    if not CLEAN_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {CLEAN_PATH}")

    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    clean = clean_all[:N]
    if clean.shape != (100, 20, 2):
        raise ValueError(f"Expected clean shape (100, 20, 2), got {clean.shape}")

    norm = np.load(NORM_PATH)
    if "rel_mean" not in norm or "rel_std" not in norm:
        raise KeyError("rel_norm_params_v2.npz must contain rel_mean and rel_std")
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    if rel_mean.shape != (2,) or rel_std.shape != (2,):
        raise ValueError(f"Expected rel_mean/rel_std shape (2,), got {rel_mean.shape} and {rel_std.shape}")

    degraded_map = {
        "gaussian_medium": generate_gaussian(clean),
        "burst_medium": generate_burst(clean),
    }

    device = torch.device("cpu")
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)
    model = TemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=2, hidden_dim=128).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    rows: list[dict] = []
    summary = {}
    best_predictions = {}

    for degradation in DEGRADATIONS:
        degraded = degraded_map[degradation]
        deg_summary = {}

        noisy_ade, noisy_smooth = compute_metrics(degraded, clean)
        noisy_ade_mean = float(np.mean(noisy_ade))
        noisy_ade_std = float(np.std(noisy_ade, ddof=0))
        noisy_smooth_mean = float(np.mean(noisy_smooth))
        rows.append(
            {
                "degradation": degradation,
                "method": "noisy_input",
                "t_start": -1,
                "N": N,
                "ADE_mean": noisy_ade_mean,
                "ADE_std": noisy_ade_std,
                "smooth_mean": noisy_smooth_mean,
                "ADE_vs_noisy_percent": 0.0,
                "wilcoxon_p_vs_noisy": np.nan,
            }
        )
        deg_summary["noisy_input"] = {
            "ADE_mean": noisy_ade_mean,
            "ADE_std": noisy_ade_std,
            "smooth_mean": noisy_smooth_mean,
            "ADE_per_traj": noisy_ade,
            "prediction": degraded,
        }

        best_method = "noisy_input"
        best_ade = noisy_ade_mean
        best_pred = degraded

        for t_start in T_LIST:
            preds = []
            for sdedit_seed in SDEDIT_SEEDS:
                preds.append(
                    run_sdedit(
                        degraded_abs=degraded,
                        model=model,
                        diffusion=diffusion,
                        rel_mean=rel_mean,
                        rel_std=rel_std,
                        t_start=t_start,
                        sdedit_seed=sdedit_seed,
                        device=device,
                    )
                )
            final_pred = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)
            ade, smooth = compute_metrics(final_pred, clean)
            ade_mean = float(np.mean(ade))
            ade_std = float(np.std(ade, ddof=0))
            smooth_mean = float(np.mean(smooth))
            improve_pct = (noisy_ade_mean - ade_mean) / noisy_ade_mean * 100.0
            delta_ade_vs_noisy = ade - noisy_ade
            if SCIPY_AVAILABLE:
                wilcoxon_result = wilcoxon(delta_ade_vs_noisy, alternative="less")
                wilcoxon_p = float(wilcoxon_result.pvalue)
            else:
                wilcoxon_p = np.nan

            method = f"sdedit_t{t_start}"
            rows.append(
                {
                    "degradation": degradation,
                    "method": method,
                    "t_start": t_start,
                    "N": N,
                    "ADE_mean": ade_mean,
                    "ADE_std": ade_std,
                    "smooth_mean": smooth_mean,
                    "ADE_vs_noisy_percent": improve_pct,
                    "wilcoxon_p_vs_noisy": wilcoxon_p,
                }
            )
            deg_summary[method] = {
                "ADE_mean": ade_mean,
                "ADE_std": ade_std,
                "smooth_mean": smooth_mean,
                "ADE_per_traj": ade,
                "wilcoxon_p_vs_noisy": wilcoxon_p,
                "prediction": final_pred,
            }

            if ade_mean < best_ade:
                best_ade = ade_mean
                best_method = method
                best_pred = final_pred

        summary[degradation] = deg_summary
        best_predictions[degradation] = {
            "method": best_method,
            "ade": best_ade,
            "prediction": best_pred,
        }

    save_csv(
        CSV_PATH,
        rows,
        [
            "degradation",
            "method",
            "t_start",
            "N",
            "ADE_mean",
            "ADE_std",
            "smooth_mean",
            "ADE_vs_noisy_percent",
            "wilcoxon_p_vs_noisy",
        ],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for col, degradation in enumerate(DEGRADATIONS):
        noisy_ade = summary[degradation]["noisy_input"]["ADE_mean"]
        sdedit_ades = [summary[degradation][f"sdedit_t{t}"]["ADE_mean"] for t in T_LIST]
        axes[0, col].plot(T_LIST, sdedit_ades, color="black", marker="o", linewidth=1.8)
        axes[0, col].axhline(noisy_ade, color="red", linestyle="--", linewidth=1.2)
        axes[0, col].set_xticks(T_LIST)
        axes[0, col].set_title(f"{degradation} ADE vs t_start")
        axes[0, col].set_xlabel("t_start")
        axes[0, col].set_ylabel("ADE_mean")
        axes[0, col].grid(True, alpha=0.3)

        ax = axes[1, col]
        degraded = degraded_map[degradation]
        best_method = best_predictions[degradation]["method"]
        best_pred = best_predictions[degradation]["prediction"]
        for idx, linestyle in zip([0, 1], ["-", "--"]):
            ax.plot(clean[idx, :, 0], clean[idx, :, 1], color="blue", linewidth=1.6, linestyle=linestyle)
            ax.plot(degraded[idx, :, 0], degraded[idx, :, 1], color="gray", linewidth=1.4, linestyle=linestyle)
            ax.plot(best_pred[idx, :, 0], best_pred[idx, :, 1], color="red", linewidth=1.6, linestyle=linestyle)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{degradation} best={best_method} (idx 0,1)")

    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")

    print("=== Step 3e Scout: SDEdit Refinement 诊断 ===")
    print(f"N trajectories: {N}")
    print(f"SDEdit seeds: {SDEDIT_SEEDS}")
    print(f"t_start list: {T_LIST}")
    if not SCIPY_AVAILABLE:
        print('warning: "scipy not available; Wilcoxon test skipped"')
    print("")

    any_improvement = False
    for degradation in DEGRADATIONS:
        print(f"--- {degradation} ---")
        print(f"{'method':15s}  {'ADE_mean':>10s}  {'ADE_std':>10s}  {'smooth_mean':>12s}  {'ADE_vs_noisy':>14s}")
        noisy_ade = summary[degradation]["noisy_input"]["ADE_mean"]
        for method in METHODS:
            stats = summary[degradation][method]
            improve = (noisy_ade - stats["ADE_mean"]) / noisy_ade * 100.0
            print(
                f"{method:15s}  {stats['ADE_mean']:10.4f}  {stats['ADE_std']:10.4f}  "
                f"{stats['smooth_mean']:12.4f}  {improve:13.1f}%"
            )
        print("")

    print("=== Scout 诊断结论 ===")
    for degradation in DEGRADATIONS:
        noisy_ade = summary[degradation]["noisy_input"]["ADE_mean"]
        best_method = best_predictions[degradation]["method"]
        best_sdedit_ade = best_predictions[degradation]["ade"]
        if best_method != "noisy_input" and best_sdedit_ade < noisy_ade:
            any_improvement = True
            print(f"[{degradation}] ✅ SDEdit 有初步 refinement 信号")
            print(f"    best method = {best_method}")
            print(f"    noisy ADE = {noisy_ade:.4f}")
            print(f"    best SDEdit ADE = {best_sdedit_ade:.4f}")
            print(f"    improvement = {(noisy_ade - best_sdedit_ade) / noisy_ade * 100:.1f}%")
        else:
            print(f"[{degradation}] ❌ SDEdit scout 未显示 refinement 信号")
            print(f"    所有 SDEdit ADE >= noisy_input ADE")

    if any_improvement:
        print("结论：可以考虑进入 full SDEdit diagnostic。")
    else:
        print("结论：暂时不要跑 full SDEdit，优先检查 SDEdit 接口或继续修 prior。")


if __name__ == "__main__":
    main()
