from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D
from tools.stage3.canonical_v1.utils import CanonicalConfig, run_kalman

try:
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False


CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42" / "best_ema_model.pt"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params_v2.npz"
CLEAN_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42"

DEGRADED_PATH = OUTPUT_DIR / "gaussian_full_degraded.npy"
SUMMARY_PATH = OUTPUT_DIR / "sdedit_gaussian_full_summary.csv"
PER_TRAJ_PATH = OUTPUT_DIR / "sdedit_gaussian_full_per_traj.csv"
FIG_PATH = OUTPUT_DIR / "sdedit_gaussian_full_diagnostic.png"
CONCLUSION_PATH = OUTPUT_DIR / "sdedit_gaussian_full_conclusion.json"

N = 200
TIMESTEPS = 100
T_LIST = [1, 2, 3, 5]
SDEDIT_SEEDS = [42, 43, 44, 45, 46]
METHODS = ["noisy_input", "kalman_cv", "sdedit_t1", "sdedit_t2", "sdedit_t3", "sdedit_t5"]
DEGRADATION_NAME = "gaussian_medium"


def save_numpy(path: Path, array: np.ndarray) -> None:
    np.save(path, array.astype(np.float32))
    print(f"Saved: {path}")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def format_pvalue(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4g}"


def generate_gaussian(clean: np.ndarray) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for idx in range(clean.shape[0]):
        rng = np.random.default_rng(42 + idx)
        noise = rng.normal(0.0, 0.05, size=(20, 2)).astype(np.float32)
        degraded[idx] = clean[idx] + noise
    return degraded


def kalman_wrapper(degraded_abs: np.ndarray) -> np.ndarray:
    obs_mask = np.ones((degraded_abs.shape[0], degraded_abs.shape[1]), dtype=np.uint8)
    config = CanonicalConfig()
    return run_kalman(
        degraded_abs.astype(np.float32),
        obs_mask,
        dt=config.kalman_dt,
        process_var=config.kalman_process_var,
        measure_var=config.kalman_measure_var,
    ).astype(np.float32)


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


def compute_per_traj_metrics(pred: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = np.linalg.norm(pred - clean, axis=-1)
    ade = errors.mean(axis=1).astype(np.float32)
    rmse = np.sqrt(np.mean(errors**2, axis=1)).astype(np.float32)
    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    smooth = np.linalg.norm(acc, axis=-1).mean(axis=1).astype(np.float32)
    return ade, rmse, smooth


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {CHECKPOINT_PATH}")
    if not NORM_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {NORM_PATH}")
    if not CLEAN_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {CLEAN_PATH}")

    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    clean = clean_all[:N]
    if clean.shape != (200, 20, 2):
        raise ValueError(f"Expected clean shape (200, 20, 2), got {clean.shape}")

    norm = np.load(NORM_PATH)
    if "rel_mean" not in norm or "rel_std" not in norm:
        raise KeyError("rel_norm_params_v2.npz must contain rel_mean and rel_std")
    rel_mean = norm["rel_mean"].astype(np.float32)
    rel_std = norm["rel_std"].astype(np.float32)
    if rel_mean.shape != (2,) or rel_std.shape != (2,):
        raise ValueError(f"Expected rel_mean/rel_std shape (2,), got {rel_mean.shape} and {rel_std.shape}")

    degraded = generate_gaussian(clean)
    save_numpy(DEGRADED_PATH, degraded)

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

    if not SCIPY_AVAILABLE:
        print("WARNING: scipy not available; Wilcoxon tests skipped")

    method_preds: dict[str, np.ndarray] = {"noisy_input": degraded}
    kalman_pred = kalman_wrapper(degraded)
    if kalman_pred.shape != (200, 20, 2):
        raise ValueError(f"Expected kalman_pred shape (200, 20, 2), got {kalman_pred.shape}")
    method_preds["kalman_cv"] = kalman_pred

    for t_start in T_LIST:
        seed_preds = []
        for sdedit_seed in SDEDIT_SEEDS:
            seed_preds.append(
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
        method_preds[f"sdedit_t{t_start}"] = np.mean(np.stack(seed_preds, axis=0), axis=0).astype(np.float32)

    metrics = {}
    for method, pred in method_preds.items():
        metrics[method] = {}
        ade, rmse, smooth = compute_per_traj_metrics(pred, clean)
        metrics[method]["ADE"] = ade
        metrics[method]["RMSE"] = rmse
        metrics[method]["smooth"] = smooth

    noisy_ade = metrics["noisy_input"]["ADE"]
    noisy_rmse = metrics["noisy_input"]["RMSE"]
    noisy_smooth = metrics["noisy_input"]["smooth"]
    kalman_ade = metrics["kalman_cv"]["ADE"]
    kalman_rmse = metrics["kalman_cv"]["RMSE"]
    kalman_smooth = metrics["kalman_cv"]["smooth"]

    summary_rows: list[dict] = []
    per_traj_rows: list[dict] = []

    for method in METHODS:
        ade = metrics[method]["ADE"]
        rmse = metrics[method]["RMSE"]
        smooth = metrics[method]["smooth"]

        delta_ade_vs_noisy = ade - noisy_ade
        delta_rmse_vs_noisy = rmse - noisy_rmse
        delta_smooth_vs_noisy = smooth - noisy_smooth
        delta_ade_vs_kalman = ade - kalman_ade
        delta_rmse_vs_kalman = rmse - kalman_rmse
        delta_smooth_vs_kalman = smooth - kalman_smooth

        improved_fraction_ade = float(np.mean(delta_ade_vs_noisy < 0))
        improved_fraction_rmse = float(np.mean(delta_rmse_vs_noisy < 0))
        smooth_improved_fraction = float(np.mean(delta_smooth_vs_noisy < 0))
        improved_fraction_ade_vs_kalman = float(np.mean(delta_ade_vs_kalman < 0))

        if method == "noisy_input":
            p_vs_noisy = np.nan
            p_vs_kalman = np.nan
            t_start = -1
            ade_vs_noisy_percent = 0.0
            delta_ade_mean = 0.0
            delta_ade_median = 0.0
            delta_ade_std = 0.0
            delta_rmse_mean = 0.0
            delta_smooth_mean = 0.0
            ade_vs_kalman_percent = (float(np.mean(kalman_ade)) - float(np.mean(ade))) / float(np.mean(kalman_ade)) * 100.0
            delta_ade_vs_kalman_mean = float(np.mean(delta_ade_vs_kalman))
            delta_ade_vs_kalman_median = float(np.median(delta_ade_vs_kalman))
        elif method == "kalman_cv":
            p_vs_noisy = float(wilcoxon(delta_ade_vs_noisy, alternative="less").pvalue) if SCIPY_AVAILABLE else np.nan
            p_vs_kalman = np.nan
            t_start = -1
            ade_vs_noisy_percent = (float(np.mean(noisy_ade)) - float(np.mean(ade))) / float(np.mean(noisy_ade)) * 100.0
            delta_ade_mean = float(np.mean(delta_ade_vs_noisy))
            delta_ade_median = float(np.median(delta_ade_vs_noisy))
            delta_ade_std = float(np.std(delta_ade_vs_noisy, ddof=0))
            delta_rmse_mean = float(np.mean(delta_rmse_vs_noisy))
            delta_smooth_mean = float(np.mean(delta_smooth_vs_noisy))
            ade_vs_kalman_percent = 0.0
            delta_ade_vs_kalman_mean = 0.0
            delta_ade_vs_kalman_median = 0.0
            improved_fraction_ade_vs_kalman = 0.0
        else:
            p_vs_noisy = float(wilcoxon(delta_ade_vs_noisy, alternative="less").pvalue) if SCIPY_AVAILABLE else np.nan
            p_vs_kalman = float(wilcoxon(delta_ade_vs_kalman, alternative="less").pvalue) if SCIPY_AVAILABLE else np.nan
            t_start = int(method.replace("sdedit_t", ""))
            ade_vs_noisy_percent = (float(np.mean(noisy_ade)) - float(np.mean(ade))) / float(np.mean(noisy_ade)) * 100.0
            delta_ade_mean = float(np.mean(delta_ade_vs_noisy))
            delta_ade_median = float(np.median(delta_ade_vs_noisy))
            delta_ade_std = float(np.std(delta_ade_vs_noisy, ddof=0))
            delta_rmse_mean = float(np.mean(delta_rmse_vs_noisy))
            delta_smooth_mean = float(np.mean(delta_smooth_vs_noisy))
            ade_vs_kalman_percent = (float(np.mean(kalman_ade)) - float(np.mean(ade))) / float(np.mean(kalman_ade)) * 100.0
            delta_ade_vs_kalman_mean = float(np.mean(delta_ade_vs_kalman))
            delta_ade_vs_kalman_median = float(np.median(delta_ade_vs_kalman))

        summary_rows.append(
            {
                "degradation": DEGRADATION_NAME,
                "method": method,
                "t_start": t_start,
                "N": N,
                "ADE_mean": float(np.mean(ade)),
                "ADE_std": float(np.std(ade, ddof=0)),
                "ADE_median": float(np.median(ade)),
                "ADE_p25": float(np.percentile(ade, 25)),
                "ADE_p75": float(np.percentile(ade, 75)),
                "RMSE_mean": float(np.mean(rmse)),
                "RMSE_std": float(np.std(rmse, ddof=0)),
                "smooth_mean": float(np.mean(smooth)),
                "smooth_std": float(np.std(smooth, ddof=0)),
                "ADE_vs_noisy_percent": ade_vs_noisy_percent,
                "delta_ADE_mean": delta_ade_mean,
                "delta_ADE_median": delta_ade_median,
                "delta_ADE_std": delta_ade_std,
                "improved_fraction_ADE": improved_fraction_ade if method != "noisy_input" else 0.0,
                "wilcoxon_p_vs_noisy": p_vs_noisy,
                "delta_RMSE_mean": delta_rmse_mean,
                "improved_fraction_RMSE": improved_fraction_rmse if method != "noisy_input" else 0.0,
                "delta_smooth_mean": delta_smooth_mean,
                "smooth_improved_fraction": smooth_improved_fraction if method != "noisy_input" else 0.0,
                "ADE_vs_kalman_percent": ade_vs_kalman_percent,
                "delta_ADE_vs_kalman_mean": delta_ade_vs_kalman_mean,
                "delta_ADE_vs_kalman_median": delta_ade_vs_kalman_median,
                "improved_fraction_ADE_vs_kalman": improved_fraction_ade_vs_kalman,
                "wilcoxon_p_vs_kalman": p_vs_kalman,
            }
        )

        for traj_idx in range(N):
            per_traj_rows.append(
                {
                    "degradation": DEGRADATION_NAME,
                    "traj_idx": traj_idx,
                    "method": method,
                    "t_start": t_start,
                    "ADE": float(ade[traj_idx]),
                    "RMSE": float(rmse[traj_idx]),
                    "smooth": float(smooth[traj_idx]),
                    "noisy_ADE": float(noisy_ade[traj_idx]),
                    "noisy_RMSE": float(noisy_rmse[traj_idx]),
                    "noisy_smooth": float(noisy_smooth[traj_idx]),
                    "kalman_ADE": float(kalman_ade[traj_idx]),
                    "kalman_RMSE": float(kalman_rmse[traj_idx]),
                    "kalman_smooth": float(kalman_smooth[traj_idx]),
                    "delta_ADE_vs_noisy": float(delta_ade_vs_noisy[traj_idx]),
                    "delta_RMSE_vs_noisy": float(delta_rmse_vs_noisy[traj_idx]),
                    "delta_smooth_vs_noisy": float(delta_smooth_vs_noisy[traj_idx]),
                    "delta_ADE_vs_kalman": float(delta_ade_vs_kalman[traj_idx]),
                    "delta_RMSE_vs_kalman": float(delta_rmse_vs_kalman[traj_idx]),
                    "delta_smooth_vs_kalman": float(delta_smooth_vs_kalman[traj_idx]),
                    "improved_ADE_vs_noisy": bool(delta_ade_vs_noisy[traj_idx] < 0),
                    "improved_RMSE_vs_noisy": bool(delta_rmse_vs_noisy[traj_idx] < 0),
                    "improved_smooth_vs_noisy": bool(delta_smooth_vs_noisy[traj_idx] < 0),
                    "improved_ADE_vs_kalman": bool(delta_ade_vs_kalman[traj_idx] < 0),
                    "improved_RMSE_vs_kalman": bool(delta_rmse_vs_kalman[traj_idx] < 0),
                    "improved_smooth_vs_kalman": bool(delta_smooth_vs_kalman[traj_idx] < 0),
                }
            )

    save_csv(
        SUMMARY_PATH,
        summary_rows,
        [
            "degradation",
            "method",
            "t_start",
            "N",
            "ADE_mean",
            "ADE_std",
            "ADE_median",
            "ADE_p25",
            "ADE_p75",
            "RMSE_mean",
            "RMSE_std",
            "smooth_mean",
            "smooth_std",
            "ADE_vs_noisy_percent",
            "delta_ADE_mean",
            "delta_ADE_median",
            "delta_ADE_std",
            "improved_fraction_ADE",
            "wilcoxon_p_vs_noisy",
            "delta_RMSE_mean",
            "improved_fraction_RMSE",
            "delta_smooth_mean",
            "smooth_improved_fraction",
            "ADE_vs_kalman_percent",
            "delta_ADE_vs_kalman_mean",
            "delta_ADE_vs_kalman_median",
            "improved_fraction_ADE_vs_kalman",
            "wilcoxon_p_vs_kalman",
        ],
    )
    save_csv(
        PER_TRAJ_PATH,
        per_traj_rows,
        [
            "degradation",
            "traj_idx",
            "method",
            "t_start",
            "ADE",
            "RMSE",
            "smooth",
            "noisy_ADE",
            "noisy_RMSE",
            "noisy_smooth",
            "kalman_ADE",
            "kalman_RMSE",
            "kalman_smooth",
            "delta_ADE_vs_noisy",
            "delta_RMSE_vs_noisy",
            "delta_smooth_vs_noisy",
            "delta_ADE_vs_kalman",
            "delta_RMSE_vs_kalman",
            "delta_smooth_vs_kalman",
            "improved_ADE_vs_noisy",
            "improved_RMSE_vs_noisy",
            "improved_smooth_vs_noisy",
            "improved_ADE_vs_kalman",
            "improved_RMSE_vs_kalman",
            "improved_smooth_vs_kalman",
        ],
    )

    summary_map = {row["method"]: row for row in summary_rows}
    sdedit_methods = [f"sdedit_t{t}" for t in T_LIST]
    best_sdedit_method = min(sdedit_methods, key=lambda m: summary_map[m]["ADE_mean"])
    best_row = summary_map[best_sdedit_method]
    noisy_row = summary_map["noisy_input"]
    kalman_row = summary_map["kalman_cv"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    sdedit_ades = [summary_map[f"sdedit_t{t}"]["ADE_mean"] for t in T_LIST]
    axes[0, 0].plot(T_LIST, sdedit_ades, color="black", marker="o", linewidth=1.8)
    axes[0, 0].axhline(noisy_row["ADE_mean"], color="red", linestyle="--", linewidth=1.2)
    axes[0, 0].axhline(kalman_row["ADE_mean"], color="blue", linestyle="--", linewidth=1.2)
    axes[0, 0].set_title("Gaussian medium: ADE vs t_start")
    axes[0, 0].set_xticks(T_LIST)
    axes[0, 0].grid(True, alpha=0.3)

    improved_fracs = [summary_map[f"sdedit_t{t}"]["improved_fraction_ADE"] for t in T_LIST]
    axes[0, 1].plot(T_LIST, improved_fracs, color="purple", marker="o", linewidth=1.8)
    axes[0, 1].axhline(0.5, color="gray", linestyle="--", linewidth=1.2)
    axes[0, 1].set_title("Improved fraction vs noisy_input")
    axes[0, 1].set_xticks(T_LIST)
    axes[0, 1].grid(True, alpha=0.3)

    box_data = [metrics[f"sdedit_t{t}"]["ADE"] - noisy_ade for t in T_LIST]
    axes[0, 2].boxplot(box_data, labels=[f"t{t}" for t in T_LIST])
    axes[0, 2].axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
    axes[0, 2].set_title("Paired ΔADE vs noisy_input")
    axes[0, 2].grid(True, alpha=0.3)

    sdedit_smooth = [summary_map[f"sdedit_t{t}"]["smooth_mean"] for t in T_LIST]
    axes[1, 0].plot(T_LIST, sdedit_smooth, color="black", marker="o", linewidth=1.8)
    axes[1, 0].axhline(noisy_row["smooth_mean"], color="red", linestyle="--", linewidth=1.2)
    axes[1, 0].axhline(kalman_row["smooth_mean"], color="blue", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("Smoothness vs t_start")
    axes[1, 0].set_xticks(T_LIST)
    axes[1, 0].grid(True, alpha=0.3)

    ax = axes[1, 1]
    best_pred = method_preds[best_sdedit_method]
    for idx, linestyle in zip([0, 1], ["-", "--"]):
        ax.plot(clean[idx, :, 0], clean[idx, :, 1], color="blue", linewidth=1.6, linestyle=linestyle)
        ax.plot(degraded[idx, :, 0], degraded[idx, :, 1], color="gray", linewidth=1.4, linestyle=linestyle)
        ax.plot(best_pred[idx, :, 0], best_pred[idx, :, 1], color="red", linewidth=1.6, linestyle=linestyle)
        ax.plot(kalman_pred[idx, :, 0], kalman_pred[idx, :, 1], color="green", linewidth=1.4, linestyle=linestyle)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Representative trajectories")

    axes[1, 2].scatter(noisy_ade, metrics[best_sdedit_method]["ADE"], s=16, alpha=0.6, color="black")
    lim_min = min(float(np.min(noisy_ade)), float(np.min(metrics[best_sdedit_method]["ADE"])))
    lim_max = max(float(np.max(noisy_ade)), float(np.max(metrics[best_sdedit_method]["ADE"])))
    axes[1, 2].plot([lim_min, lim_max], [lim_min, lim_max], color="gray", linestyle="--", linewidth=1.2)
    axes[1, 2].set_title("Per-trajectory ADE: noisy vs best SDEdit")
    axes[1, 2].set_xlabel("noisy_input ADE")
    axes[1, 2].set_ylabel("best SDEdit ADE")
    axes[1, 2].grid(True, alpha=0.3)

    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")

    sdedit_supported_vs_noisy = (
        best_row["ADE_mean"] < noisy_row["ADE_mean"]
        and best_row["delta_ADE_mean"] < 0
        and (not np.isnan(best_row["wilcoxon_p_vs_noisy"]) and best_row["wilcoxon_p_vs_noisy"] < 0.05)
    )
    sdedit_supported_vs_kalman = (
        best_row["ADE_mean"] < kalman_row["ADE_mean"]
        and best_row["delta_ADE_vs_kalman_mean"] < 0
        and (not np.isnan(best_row["wilcoxon_p_vs_kalman"]) and best_row["wilcoxon_p_vs_kalman"] < 0.05)
    )

    if sdedit_supported_vs_noisy:
        interpretation = "stable_refinement_signal_vs_noisy"
    elif best_row["ADE_mean"] < noisy_row["ADE_mean"]:
        interpretation = "weak_or_unstable_signal"
    else:
        interpretation = "no_refinement_signal"

    save_json(
        CONCLUSION_PATH,
        {
            "N": N,
            "degradation": DEGRADATION_NAME,
            "sdedit_seeds": SDEDIT_SEEDS,
            "t_start_list": T_LIST,
            "noisy_ADE_mean": noisy_row["ADE_mean"],
            "kalman_ADE_mean": kalman_row["ADE_mean"],
            "best_sdedit_method": best_sdedit_method,
            "best_sdedit_ADE_mean": best_row["ADE_mean"],
            "best_delta_ADE_vs_noisy": best_row["delta_ADE_mean"],
            "best_improved_fraction_vs_noisy": best_row["improved_fraction_ADE"],
            "best_wilcoxon_p_vs_noisy": best_row["wilcoxon_p_vs_noisy"],
            "best_delta_ADE_vs_kalman": best_row["delta_ADE_vs_kalman_mean"],
            "best_improved_fraction_vs_kalman": best_row["improved_fraction_ADE_vs_kalman"],
            "best_wilcoxon_p_vs_kalman": best_row["wilcoxon_p_vs_kalman"],
            "sdedit_supported_vs_noisy": sdedit_supported_vs_noisy,
            "sdedit_supported_vs_kalman": sdedit_supported_vs_kalman,
            "interpretation": interpretation,
        },
    )

    print("=== Step 3f: Gaussian-only Full SDEdit Diagnostic ===")
    print(f"N trajectories: {N}")
    print(f"SDEdit seeds: {SDEDIT_SEEDS}")
    print(f"t_start list: {T_LIST}")
    print("degradation: gaussian_medium")
    print("")
    print("--- gaussian_medium ---")
    print(
        f"{'method':15s}  "
        f"{'ADE_mean':>10s}  "
        f"{'ADE_std':>10s}  "
        f"{'RMSE_mean':>10s}  "
        f"{'smooth_mean':>12s}  "
        f"{'ADE_vs_noisy':>14s}  "
        f"{'delta_ADE':>11s}  "
        f"{'improved_frac':>14s}  "
        f"{'p_noisy':>12s}  "
        f"{'p_kalman':>12s}"
    )
    for method in METHODS:
        row = summary_map[method]
        print(
            f"{method:15s}  "
            f"{row['ADE_mean']:10.4f}  "
            f"{row['ADE_std']:10.4f}  "
            f"{row['RMSE_mean']:10.4f}  "
            f"{row['smooth_mean']:12.4f}  "
            f"{row['ADE_vs_noisy_percent']:13.1f}%  "
            f"{row['delta_ADE_mean']:11.4f}  "
            f"{row['improved_fraction_ADE']:14.3f}  "
            f"{format_pvalue(row['wilcoxon_p_vs_noisy']):>12s}  "
            f"{format_pvalue(row['wilcoxon_p_vs_kalman']):>12s}"
        )
    print("")
    print("=== 诊断结论 ===")
    if sdedit_supported_vs_noisy:
        print("[gaussian_medium] ✅ SDEdit has statistically supported refinement signal vs noisy_input")
        print(f"best method = {best_sdedit_method}")
        print(f"noisy ADE = {noisy_row['ADE_mean']:.4f}")
        print(f"best SDEdit ADE = {best_row['ADE_mean']:.4f}")
        print(f"mean paired delta ADE vs noisy = {best_row['delta_ADE_mean']:.4f}")
        print(f"improved fraction vs noisy = {best_row['improved_fraction_ADE']:.3f}")
        print(f"Wilcoxon p vs noisy = {format_pvalue(best_row['wilcoxon_p_vs_noisy'])}")
        print(f"ADE improvement vs noisy = {best_row['ADE_vs_noisy_percent']:.1f}%")
    else:
        print("[gaussian_medium] ❌ SDEdit does not show stable refinement value vs noisy_input")
        print("Reason: ADE_mean, paired delta, or Wilcoxon p-value does not support improvement.")

    if sdedit_supported_vs_kalman:
        print("[gaussian_medium] ✅ SDEdit also beats kalman_cv with paired statistical support")
        print(f"kalman ADE = {kalman_row['ADE_mean']:.4f}")
        print(f"mean paired delta ADE vs kalman = {best_row['delta_ADE_vs_kalman_mean']:.4f}")
        print(f"improved fraction vs kalman = {best_row['improved_fraction_ADE_vs_kalman']:.3f}")
        print(f"Wilcoxon p vs kalman = {format_pvalue(best_row['wilcoxon_p_vs_kalman'])}")
    else:
        print("[gaussian_medium] ⚠️ SDEdit does not beat kalman_cv with paired statistical support")
        print(f"kalman ADE = {kalman_row['ADE_mean']:.4f}")
        print(f"mean paired delta ADE vs kalman = {best_row['delta_ADE_vs_kalman_mean']:.4f}")
        print(f"improved fraction vs kalman = {best_row['improved_fraction_ADE_vs_kalman']:.3f}")
        print(f"Wilcoxon p vs kalman = {format_pvalue(best_row['wilcoxon_p_vs_kalman'])}")

    if best_row["smooth_mean"] < noisy_row["smooth_mean"] and best_row["ADE_mean"] >= noisy_row["ADE_mean"]:
        print("注意：SDEdit lowers smoothness but does not improve ADE. This indicates smoothing without accurate correction.")
    if best_row["smooth_mean"] < noisy_row["smooth_mean"] and best_row["ADE_mean"] < noisy_row["ADE_mean"]:
        print("注意：SDEdit improves ADE while also lowering smoothness. This supports mild denoising rather than only visual smoothing.")


if __name__ == "__main__":
    main()
