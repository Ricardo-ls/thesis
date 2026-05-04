from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

torch.set_num_threads(max(1, os.cpu_count() or 1))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D
from tools.simulation.degradation_calibrator import GlobalDegrader
from tools.stage3_global.methods.classical import apply_kalman_rts


TEST_PATH = PROJECT_ROOT / "data" / "simulated" / "test_trajs.npy"
VAL_PATH = PROJECT_ROOT / "data" / "simulated" / "val_trajs.npy"
CURVATURE_PATH = PROJECT_ROOT / "data" / "simulated" / "test_curvature.npy"
DEGRADATION_PARAMS_PATH = PROJECT_ROOT / "data" / "simulated" / "degradation_params.json"
ETH_CKPT_PATH = PROJECT_ROOT / "outputs" / "prior" / "train" / "ddpm_eth_ucy_none_h128" / "seed42-100epoch" / "best_model.pt"
FINETUNED_CKPT_PATH = PROJECT_ROOT / "outputs" / "prior" / "train" / "ddpm_finetuned_h128" / "seed42-100epoch" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_sim"
FULL_RESULTS_PATH = OUTPUT_DIR / "full_results.csv"
CURVATURE_RESULTS_PATH = OUTPUT_DIR / "curvature_results.csv"
PER_TRAJ_RESULTS_PATH = OUTPUT_DIR / "per_trajectory_metrics.csv"
CONFIG_PATH = OUTPUT_DIR / "experiment_config.json"

METHODS = ["noisy_input", "kalman_cv", "ddpm_eth_sdedit", "ddpm_finetuned_sdedit"]
DEGRADATIONS = [
    "gaussian_medium",
    "bias_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
    "combined_medium",
]
DEGRADATION_SEED_BASE = {
    "gaussian_medium": 10000,
    "bias_medium": 20000,
    "drift_medium": 30000,
    "jump_medium": 40000,
    "burst_medium": 50000,
    "combined_medium": 60000,
}
DDPM_SEEDS = [42, 1, 2, 43, 44]
VAL_DEGRADATION_SEED_BASE = {name: 70000 + idx for idx, name in enumerate(DEGRADATIONS)}
KALMAN_SOURCE = str(PROJECT_ROOT / "tools" / "stage3_global" / "methods" / "classical.py")

TIMESTEPS = 100
HIDDEN_DIM = 128
DDPM_CHUNK_SIZE = 256


def ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required path is missing: {path}")


def load_inputs():
    for path in [
        TEST_PATH,
        VAL_PATH,
        CURVATURE_PATH,
        DEGRADATION_PARAMS_PATH,
        ETH_CKPT_PATH,
        FINETUNED_CKPT_PATH,
        Path(KALMAN_SOURCE),
    ]:
        ensure_exists(path)

    test_abs = np.load(TEST_PATH, allow_pickle=False).astype(np.float32)[:1000]
    val_abs = np.load(VAL_PATH, allow_pickle=False).astype(np.float32)
    curvature = np.load(CURVATURE_PATH, allow_pickle=False).astype(np.float32)[:1000]

    if tuple(test_abs.shape) != (1000, 20, 2):
        raise RuntimeError(f"Expected sliced test_trajs shape (1000, 20, 2), got {test_abs.shape}")
    if tuple(val_abs.shape) != (2000, 20, 2):
        raise RuntimeError(f"Expected val_trajs shape (2000, 20, 2), got {val_abs.shape}")
    if tuple(curvature.shape) != (1000,):
        raise RuntimeError(f"Expected sliced test_curvature shape (1000,), got {curvature.shape}")

    with DEGRADATION_PARAMS_PATH.open("r", encoding="utf-8") as f:
        degradation_params = json.load(f)
    return test_abs, val_abs, curvature, degradation_params


def abs_to_rel(abs_trajs: np.ndarray) -> np.ndarray:
    return (abs_trajs[:, 1:, :] - abs_trajs[:, :-1, :]).astype(np.float32)


def rel_to_abs(rel_trajs: np.ndarray, degraded_abs: np.ndarray) -> np.ndarray:
    rel_trajs = np.asarray(rel_trajs, dtype=np.float32)
    degraded_abs = np.asarray(degraded_abs, dtype=np.float32)
    out = np.zeros((rel_trajs.shape[0], rel_trajs.shape[1] + 1, 2), dtype=np.float32)
    out[:, 0, :] = degraded_abs[:, 0, :]
    out[:, 1:, :] = degraded_abs[:, 0:1, :] + np.cumsum(rel_trajs, axis=1)
    return out.astype(np.float32)


def extract_method_params(params_json: dict, degradation_name: str):
    entry = params_json[degradation_name]
    if degradation_name.startswith("gaussian"):
        return "gaussian", {"sigma": float(entry["sigma"])}
    if degradation_name.startswith("bias"):
        return "bias", {"sigma": float(entry["sigma"])}
    if degradation_name.startswith("drift"):
        return "drift", {"sigma_step": float(entry["sigma_step"])}
    if degradation_name.startswith("jump"):
        return "jump", {"sigma": float(entry["sigma"])}
    if degradation_name.startswith("burst"):
        return "burst", {"sigma": float(entry["sigma"])}
    if degradation_name.startswith("combined"):
        return "combined", {
            "sigma_g": float(entry["sigma_g"]),
            "sigma_d": float(entry["sigma_d"]),
            "sigma_b": float(entry["sigma_b"]),
        }
    raise ValueError(f"Unsupported degradation entry: {degradation_name}")


def compute_signal_and_tstart(val_abs: np.ndarray, degradation_name: str, params_json: dict):
    degrader = GlobalDegrader()
    deg_type, deg_params = extract_method_params(params_json, degradation_name)
    degraded_val_abs = degrader.apply_batch(
        val_abs,
        deg_type,
        deg_params,
        seed_base=VAL_DEGRADATION_SEED_BASE[degradation_name],
    )
    clean_val_rel = abs_to_rel(val_abs)
    degraded_val_rel = abs_to_rel(degraded_val_abs)

    signal_scale = float(clean_val_rel.std())
    rel_noise_scale = float(np.sqrt(np.mean((degraded_val_rel - clean_val_rel) ** 2)))

    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device="cpu")
    noise_curve = torch.sqrt(1.0 - diffusion.alpha_bars).cpu().numpy()
    ratio = float(rel_noise_scale / max(signal_scale, 1e-8))
    t_start = int(np.argmin(np.abs(noise_curve - ratio)))
    t_start = int(np.clip(t_start, 1, 80))
    return signal_scale, rel_noise_scale, t_start


def load_checkpoint_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model = TemporalDenoiser1D(
        max_timesteps=TIMESTEPS,
        in_channels=2,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def compute_per_traj_metrics(clean_abs: np.ndarray, pred_abs: np.ndarray) -> dict[str, np.ndarray]:
    diff = pred_abs - clean_abs
    point_err = np.linalg.norm(diff, axis=-1)
    ade = point_err.mean(axis=1).astype(np.float32)
    rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1), axis=1)).astype(np.float32)
    smooth = np.linalg.norm(pred_abs[:, 2:, :] - 2.0 * pred_abs[:, 1:-1, :] + pred_abs[:, :-2, :], axis=-1).mean(axis=1)
    return {
        "ADE": ade.astype(np.float32),
        "RMSE": rmse.astype(np.float32),
        "smooth": smooth.astype(np.float32),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Saved: {path}")


@torch.no_grad()
def run_sdedit_batch(
    model: torch.nn.Module,
    degraded_abs: np.ndarray,
    t_start: int,
    device: str,
) -> np.ndarray:
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)
    degraded_rel = abs_to_rel(degraded_abs)
    x_deg_np = np.transpose(degraded_rel, (0, 2, 1)).astype(np.float32)
    n_traj = x_deg_np.shape[0]
    num_seed = len(DDPM_SEEDS)
    abs_out = np.zeros((n_traj, num_seed, 20, 2), dtype=np.float32)
    for seed_idx, seed in enumerate(DDPM_SEEDS):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        for start in range(0, n_traj, DDPM_CHUNK_SIZE):
            end = min(start + DDPM_CHUNK_SIZE, n_traj)
            x_deg = torch.from_numpy(x_deg_np[start:end]).to(device)
            t_vec = torch.full((x_deg.shape[0],), int(t_start), device=device, dtype=torch.long)

            init_noise = torch.randn(x_deg.shape, generator=generator, device=device, dtype=x_deg.dtype)
            x = diffusion.sqrt_alpha_bars[t_vec].view(-1, 1, 1) * x_deg + diffusion.sqrt_one_minus_alpha_bars[t_vec].view(-1, 1, 1) * init_noise

            for t in range(int(t_start), -1, -1):
                t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
                pred_noise = model(x, t_batch)
                alpha_t = diffusion.alphas[t]
                alpha_bar_t = diffusion.alpha_bars[t]
                beta_t = diffusion.betas[t]
                mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)
                if t > 0:
                    step_noise = torch.randn(x_deg.shape, generator=generator, device=device, dtype=x_deg.dtype)
                    x = mean + torch.sqrt(beta_t) * step_noise
                else:
                    x = mean

            rel_chunk = np.transpose(x.cpu().numpy().astype(np.float32), (0, 2, 1))
            abs_out[start:end, seed_idx, :, :] = rel_to_abs(rel_chunk, degraded_abs[start:end])
    return abs_out.astype(np.float32)


def build_curve_groups(curvature: np.ndarray):
    p33 = float(np.percentile(curvature, 33.3333))
    p66 = float(np.percentile(curvature, 66.6667))
    groups = np.empty(curvature.shape[0], dtype=object)
    groups[curvature < p33] = "low_curve"
    groups[(curvature >= p33) & (curvature < p66)] = "mid_curve"
    groups[curvature >= p66] = "high_curve"
    return groups, {"p33": p33, "p66": p66}


def main():
    test_abs, val_abs, curvature, degradation_params = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    curve_groups, curve_thresholds = build_curve_groups(curvature)
    signal_scale = None
    relative_noise_scales = {}
    t_start_values = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eth_model = load_checkpoint_model(ETH_CKPT_PATH, device=device)
    finetuned_model = load_checkpoint_model(FINETUNED_CKPT_PATH, device=device)
    degrader = GlobalDegrader()

    full_rows = []
    curvature_rows = []
    per_traj_rows = []

    for degradation in DEGRADATIONS:
        if signal_scale is None:
            signal_scale, rel_noise_scale, t_start = compute_signal_and_tstart(val_abs, degradation, degradation_params)
        else:
            _, rel_noise_scale, t_start = compute_signal_and_tstart(val_abs, degradation, degradation_params)
        relative_noise_scales[degradation] = float(rel_noise_scale)
        t_start_values[degradation] = int(t_start)

        deg_type, deg_params = extract_method_params(degradation_params, degradation)
        degraded_abs = degrader.apply_batch(
            test_abs,
            deg_type,
            deg_params,
            seed_base=DEGRADATION_SEED_BASE[degradation],
        )

        method_metrics = {}

        noisy_metrics = compute_per_traj_metrics(test_abs, degraded_abs)
        method_metrics["noisy_input"] = noisy_metrics

        kalman_pred = apply_kalman_rts(degraded_abs, process_noise=0.01, obs_noise=0.05, dt=1.0)
        kalman_metrics = compute_per_traj_metrics(test_abs, kalman_pred)
        method_metrics["kalman_cv"] = kalman_metrics
        print(f"[stage] {degradation:16s} kalman done")

        eth_samples = run_sdedit_batch(eth_model, degraded_abs, t_start=t_start, device=device)
        print(f"[stage] {degradation:16s} ddpm_eth done")
        ft_samples = run_sdedit_batch(finetuned_model, degraded_abs, t_start=t_start, device=device)
        print(f"[stage] {degradation:16s} ddpm_finetuned done")

        for method_name, samples in [
            ("ddpm_eth_sdedit", eth_samples),
            ("ddpm_finetuned_sdedit", ft_samples),
        ]:
            per_seed = {key: [] for key in ["ADE", "RMSE", "smooth"]}
            for seed_idx in range(samples.shape[1]):
                metrics = compute_per_traj_metrics(test_abs, samples[:, seed_idx, :, :])
                for key in per_seed:
                    per_seed[key].append(metrics[key])
            averaged = {
                key: np.stack(vals, axis=1).mean(axis=1).astype(np.float32)
                for key, vals in per_seed.items()
            }
            method_metrics[method_name] = averaged

        for method_name in METHODS:
            metrics = method_metrics[method_name]
            n_seed = 1 if method_name in {"noisy_input", "kalman_cv"} else 5

            row = {
                "method": method_name,
                "degradation": degradation,
                "n_traj": int(test_abs.shape[0]),
                "n_seed": int(n_seed),
            }
            for metric_name in ["ADE", "RMSE", "smooth"]:
                stats = summarize(metrics[metric_name])
                for suffix, value in stats.items():
                    row[f"{metric_name}_{suffix}"] = float(value)
            full_rows.append(row)

            for group_name in ["low_curve", "mid_curve", "high_curve"]:
                mask = curve_groups == group_name
                group_row = {
                    "method": method_name,
                    "degradation": degradation,
                    "curve_group": group_name,
                    "n_traj": int(mask.sum()),
                    "n_seed": int(n_seed),
                }
                for metric_name in ["ADE", "RMSE", "smooth"]:
                    stats = summarize(metrics[metric_name][mask])
                    for suffix, value in stats.items():
                        group_row[f"{metric_name}_{suffix}"] = float(value)
                curvature_rows.append(group_row)

            for traj_idx in range(test_abs.shape[0]):
                per_traj_rows.append(
                    {
                        "traj_idx": int(traj_idx),
                        "method": method_name,
                        "degradation": degradation,
                        "curve_group": str(curve_groups[traj_idx]),
                        "ADE": float(metrics["ADE"][traj_idx]),
                        "RMSE": float(metrics["RMSE"][traj_idx]),
                        "smooth": float(metrics["smooth"][traj_idx]),
                        "n_seed": int(n_seed),
                    }
                )

        print(f"[done] {degradation:16s} t_start={t_start:3d} rel_noise_scale={rel_noise_scale:.4f}")

    full_fieldnames = [
        "method", "degradation",
        "ADE_mean", "ADE_std", "ADE_median", "ADE_p25", "ADE_p75",
        "RMSE_mean", "RMSE_std", "RMSE_median", "RMSE_p25", "RMSE_p75",
        "smooth_mean", "smooth_std", "smooth_median", "smooth_p25", "smooth_p75",
        "n_traj", "n_seed",
    ]
    curvature_fieldnames = [
        "method", "degradation", "curve_group",
        "ADE_mean", "ADE_std", "ADE_median", "ADE_p25", "ADE_p75",
        "RMSE_mean", "RMSE_std", "RMSE_median", "RMSE_p25", "RMSE_p75",
        "smooth_mean", "smooth_std", "smooth_median", "smooth_p25", "smooth_p75",
        "n_traj", "n_seed",
    ]
    per_traj_fieldnames = ["traj_idx", "method", "degradation", "curve_group", "ADE", "RMSE", "smooth", "n_seed"]

    save_csv(FULL_RESULTS_PATH, full_rows, full_fieldnames)
    save_csv(CURVATURE_RESULTS_PATH, curvature_rows, curvature_fieldnames)
    save_csv(PER_TRAJ_RESULTS_PATH, per_traj_rows, per_traj_fieldnames)

    config = {
        "test_n": int(test_abs.shape[0]),
        "val_n": int(val_abs.shape[0]),
        "degradation_params": degradation_params,
        "degradation_seed_base": DEGRADATION_SEED_BASE,
        "ddpm_seeds": DDPM_SEEDS,
        "t_start_values": t_start_values,
        "relative_noise_scales": relative_noise_scales,
        "signal_scale": float(signal_scale),
        "curvature_thresholds": curve_thresholds,
        "eth_checkpoint_path": str(ETH_CKPT_PATH),
        "finetuned_checkpoint_path": str(FINETUNED_CKPT_PATH),
        "kalman_source": KALMAN_SOURCE,
    }
    save_json(CONFIG_PATH, config)

    gaussian_preview = {
        row["method"]: row["ADE_mean"]
        for row in full_rows
        if row["degradation"] == "gaussian_medium"
    }
    bias_preview = {
        row["method"]: row["ADE_mean"]
        for row in full_rows
        if row["degradation"] == "bias_medium"
    }
    high_curve_preview = {
        row["method"]: row["ADE_mean"]
        for row in curvature_rows
        if row["degradation"] == "gaussian_medium" and row["curve_group"] == "high_curve"
    }

    print("=== 指令4 验证输出 ===")
    print(f"full_results.csv 行数        : {len(full_rows)}   ← 必须为 24")
    print(f"curvature_results.csv 行数   : {len(curvature_rows)}  ← 必须为 72")
    print(f"per_trajectory_metrics.csv 行数: {len(per_traj_rows)} ← 必须为 120000")
    print()

    print("=== SDEdit t_start values ===")
    for deg, t in t_start_values.items():
        rns = relative_noise_scales[deg]
        print(f"  {deg:20s}: t_start={t:3d}  rel_noise_scale={rns:.4f}")
    print(f"  signal_scale (clean val rel std) = {float(signal_scale):.4f}")
    print()

    print("=== 核心结果预览（gaussian_medium，ADE_mean）===")
    for method in METHODS:
        print(f"  {method:28s}: {float(gaussian_preview[method]):.4f}")
    print()

    print("=== 核心结果预览（bias_medium，ADE_mean）===")
    for method in METHODS:
        print(f"  {method:28s}: {float(bias_preview[method]):.4f}")
    print()

    print("=== 曲率分层预览（gaussian_medium，high_curve，ADE_mean）===")
    for method in ["kalman_cv", "ddpm_eth_sdedit", "ddpm_finetuned_sdedit"]:
        print(f"  {method:28s}: {float(high_curve_preview[method]):.4f}")
    print()

    print("注意：smooth 是输出轨迹自身曲率，不是误差指标，低不代表好。")
    print("=== 等待人工确认 ===")


if __name__ == "__main__":
    main()
