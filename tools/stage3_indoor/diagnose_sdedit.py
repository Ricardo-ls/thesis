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

from tools.stage3_indoor.train_ddpm_prior import DDPMProcess, HIDDEN_DIM, TemporalDenoiser1D, TIMESTEPS

DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "sdedit_diagnostic"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_prior"
CHECKPOINT_PATH = CHECKPOINT_DIR / "val_selected_model.pt"

CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
GAUSSIAN_PATH = DATA_DIR / "degraded_gaussian_medium.npy"
BURST_PATH = DATA_DIR / "degraded_burst_medium.npy"

SUMMARY_PATH = OUTPUT_DIR / "diagnostic_summary.csv"
PER_TRAJ_PATH = OUTPUT_DIR / "diagnostic_per_traj.csv"
FIG_PATH = OUTPUT_DIR / "diagnostic_examples.png"
CONFIG_PATH = OUTPUT_DIR / "diagnostic_config.json"

N_EVAL = 1000
BATCH_SIZE = 128
DEVICE = torch.device("cpu")
SEEDS = [42, 43, 44]
T_START_VALUES = [1, 3, 5, 10, 20]
DEGRADATIONS = ["gaussian_medium", "burst_medium"]
METHODS = [
    "noisy_input",
    "ddpm_sdedit_t1",
    "ddpm_sdedit_t3",
    "ddpm_sdedit_t5",
    "ddpm_sdedit_t10",
    "ddpm_sdedit_t20",
]


def load_required_array(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    array = np.load(path)
    if array.shape != expected_shape:
        raise ValueError(f"Shape mismatch for {path}: expected {expected_shape}, got {array.shape}")
    return array.astype(np.float32)


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


def rel_from_abs(abs_trajs: np.ndarray) -> np.ndarray:
    return (abs_trajs[:, 1:, :] - abs_trajs[:, :-1, :]).astype(np.float32)


def reconstruct_abs_from_rel(degraded_abs: np.ndarray, refined_rel: np.ndarray) -> np.ndarray:
    refined_abs = np.zeros((degraded_abs.shape[0], 20, 2), dtype=np.float32)
    refined_abs[:, 0, :] = degraded_abs[:, 0, :]
    refined_abs[:, 1:, :] = degraded_abs[:, 0:1, :] + np.cumsum(refined_rel, axis=1)
    return refined_abs


def run_sdedit_sampling(
    model: TemporalDenoiser1D,
    diffusion: DDPMProcess,
    degraded_abs: np.ndarray,
    t_start: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    degraded_rel = rel_from_abs(degraded_abs)
    outputs = []

    with torch.no_grad():
        for start in range(0, degraded_rel.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, degraded_rel.shape[0])
            batch_rel = degraded_rel[start:end]
            x0 = torch.from_numpy(batch_rel.transpose(0, 2, 1)).to(device=DEVICE, dtype=torch.float32)

            t_tensor = torch.full((x0.shape[0],), t_start, device=DEVICE, dtype=torch.long)
            eps = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t_tensor, eps)

            for t_idx in reversed(range(t_start + 1)):
                t = torch.full((x_t.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
                eps_pred = model(x_t, t)

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

            refined_rel = x_t.permute(0, 2, 1).cpu().numpy().astype(np.float32)
            outputs.append(reconstruct_abs_from_rel(degraded_abs[start:end], refined_rel))

    return np.concatenate(outputs, axis=0).astype(np.float32)


def compute_per_traj_metrics(pred: np.ndarray, clean: np.ndarray) -> dict[str, np.ndarray]:
    errors = np.linalg.norm(pred - clean, axis=-1)
    ade = errors.mean(axis=1)
    rmse = np.sqrt(np.mean(errors**2, axis=1))

    steps = pred[:, 1:, :] - pred[:, :-1, :]
    step_norm = np.linalg.norm(steps, axis=-1)
    step_mean = step_norm.mean(axis=1)
    step_p95 = np.percentile(step_norm, 95, axis=1)
    step_max = np.max(step_norm, axis=1)

    acc = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
    acc_norm = np.linalg.norm(acc, axis=-1)
    smooth = acc_norm.mean(axis=1)
    acc_mean = acc_norm.mean(axis=1)
    acc_p95 = np.percentile(acc_norm, 95, axis=1)

    out_of_room = (
        (pred[..., 0] < 0.0)
        | (pred[..., 0] > 3.0)
        | (pred[..., 1] < 0.0)
        | (pred[..., 1] > 3.0)
    )
    out_of_room_ratio = out_of_room.mean(axis=1)

    return {
        "ADE": ade.astype(np.float32),
        "RMSE": rmse.astype(np.float32),
        "smooth": smooth.astype(np.float32),
        "step_mean": step_mean.astype(np.float32),
        "step_p95": step_p95.astype(np.float32),
        "step_max": step_max.astype(np.float32),
        "acc_mean": acc_mean.astype(np.float32),
        "acc_p95": acc_p95.astype(np.float32),
        "out_of_room_ratio": out_of_room_ratio.astype(np.float32),
    }


def summarize_metrics(metrics: dict[str, np.ndarray], degradation: str, method: str, t_start: int) -> dict:
    ade = metrics["ADE"]
    return {
        "degradation": degradation,
        "method": method,
        "t_start": int(t_start),
        "ADE_mean": float(np.mean(ade)),
        "ADE_std": float(np.std(ade)),
        "ADE_median": float(np.median(ade)),
        "ADE_p25": float(np.percentile(ade, 25)),
        "ADE_p75": float(np.percentile(ade, 75)),
        "RMSE_mean": float(np.mean(metrics["RMSE"])),
        "smooth_mean": float(np.mean(metrics["smooth"])),
        "step_mean": float(np.mean(metrics["step_mean"])),
        "step_p95_mean": float(np.mean(metrics["step_p95"])),
        "step_max_mean": float(np.mean(metrics["step_max"])),
        "acc_mean": float(np.mean(metrics["acc_mean"])),
        "acc_p95_mean": float(np.mean(metrics["acc_p95"])),
        "out_of_room_ratio_mean": float(np.mean(metrics["out_of_room_ratio"])),
        "n_traj": int(len(ade)),
    }


def plot_examples(
    clean_eval: np.ndarray,
    predictions: dict[tuple[str, str], np.ndarray],
    metrics_map: dict[tuple[str, str], dict[str, np.ndarray]],
) -> None:
    col_methods = [
        "noisy_input",
        "ddpm_sdedit_t1",
        "ddpm_sdedit_t5",
        "ddpm_sdedit_t10",
        "ddpm_sdedit_t20",
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)

    for row, degradation in enumerate(DEGRADATIONS):
        noisy_metrics = metrics_map[(degradation, "noisy_input")]
        noisy_ade = noisy_metrics["ADE"]
        target = float(np.median(noisy_ade))
        traj_index = int(np.argmin(np.abs(noisy_ade - target)))

        for col, method in enumerate(col_methods):
            ax = axes[row, col]
            pred = predictions[(degradation, method)]
            ade = metrics_map[(degradation, method)]["ADE"][traj_index]
            ax.plot(clean_eval[traj_index, :, 0], clean_eval[traj_index, :, 1], color="black", linewidth=1.8)
            ax.plot(pred[traj_index, :, 0], pred[traj_index, :, 1], color="red", linewidth=1.8)
            ax.scatter(clean_eval[traj_index, 0, 0], clean_eval[traj_index, 0, 1], color="black", marker="o", s=26)
            ax.scatter(pred[traj_index, 0, 0], pred[traj_index, 0, 1], color="red", marker="s", s=28)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 3.5)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.35)
            ax.set_title(
                f"{degradation}\n{method}\ntraj={traj_index}  ADE={ade:.4f}",
                fontsize=9,
            )

    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")


def main() -> None:
    clean_all = load_required_array(CLEAN_PATH, (2000, 20, 2))
    clean_eval = clean_all[:N_EVAL]
    if clean_eval.shape != (1000, 20, 2):
        raise ValueError(f"Shape mismatch for clean_eval: got {clean_eval.shape}")

    degraded_gaussian = load_required_array(GAUSSIAN_PATH, (1000, 20, 2))
    degraded_burst = load_required_array(BURST_PATH, (1000, 20, 2))

    if not CHECKPOINT_PATH.is_file():
        print(f"Missing required checkpoint: {CHECKPOINT_PATH}")
        print(f"Files under {CHECKPOINT_DIR}:")
        for item in sorted(CHECKPOINT_DIR.iterdir()):
            print(item.name)
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = TemporalDenoiser1D(hidden_dim=HIDDEN_DIM).to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    diffusion = DDPMProcess(timesteps=TIMESTEPS, device=DEVICE)

    degraded_map = {
        "gaussian_medium": degraded_gaussian,
        "burst_medium": degraded_burst,
    }

    predictions: dict[tuple[str, str], np.ndarray] = {}
    metrics_map: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    summary_rows: list[dict] = []
    per_traj_rows: list[dict] = []

    for degradation in DEGRADATIONS:
        degraded_abs = degraded_map[degradation]

        noisy_pred = degraded_abs.copy()
        noisy_metrics = compute_per_traj_metrics(noisy_pred, clean_eval)
        predictions[(degradation, "noisy_input")] = noisy_pred
        metrics_map[(degradation, "noisy_input")] = noisy_metrics
        summary_rows.append(summarize_metrics(noisy_metrics, degradation, "noisy_input", 0))

        for traj_index in range(N_EVAL):
            per_traj_rows.append(
                {
                    "degradation": degradation,
                    "method": "noisy_input",
                    "t_start": 0,
                    "traj_index": traj_index,
                    "ADE": float(noisy_metrics["ADE"][traj_index]),
                    "RMSE": float(noisy_metrics["RMSE"][traj_index]),
                    "smooth": float(noisy_metrics["smooth"][traj_index]),
                    "step_mean": float(noisy_metrics["step_mean"][traj_index]),
                    "step_p95": float(noisy_metrics["step_p95"][traj_index]),
                    "step_max": float(noisy_metrics["step_max"][traj_index]),
                    "acc_mean": float(noisy_metrics["acc_mean"][traj_index]),
                    "acc_p95": float(noisy_metrics["acc_p95"][traj_index]),
                    "out_of_room_ratio": float(noisy_metrics["out_of_room_ratio"][traj_index]),
                }
            )

        for t_start in T_START_VALUES:
            seed_predictions = []
            for seed in SEEDS:
                refined_abs = run_sdedit_sampling(
                    model=model,
                    diffusion=diffusion,
                    degraded_abs=degraded_abs,
                    t_start=t_start,
                    seed=seed,
                )
                if refined_abs.shape != (1000, 20, 2):
                    raise ValueError(
                        f"Refined output shape mismatch for {degradation}, t={t_start}, seed={seed}: "
                        f"{refined_abs.shape}"
                    )
                seed_predictions.append(refined_abs)

            refined_abs_mean = np.mean(np.stack(seed_predictions, axis=0), axis=0).astype(np.float32)
            method = f"ddpm_sdedit_t{t_start}"
            metrics = compute_per_traj_metrics(refined_abs_mean, clean_eval)

            predictions[(degradation, method)] = refined_abs_mean
            metrics_map[(degradation, method)] = metrics
            summary_rows.append(summarize_metrics(metrics, degradation, method, t_start))

            for traj_index in range(N_EVAL):
                per_traj_rows.append(
                    {
                        "degradation": degradation,
                        "method": method,
                        "t_start": t_start,
                        "traj_index": traj_index,
                        "ADE": float(metrics["ADE"][traj_index]),
                        "RMSE": float(metrics["RMSE"][traj_index]),
                        "smooth": float(metrics["smooth"][traj_index]),
                        "step_mean": float(metrics["step_mean"][traj_index]),
                        "step_p95": float(metrics["step_p95"][traj_index]),
                        "step_max": float(metrics["step_max"][traj_index]),
                        "acc_mean": float(metrics["acc_mean"][traj_index]),
                        "acc_p95": float(metrics["acc_p95"][traj_index]),
                        "out_of_room_ratio": float(metrics["out_of_room_ratio"][traj_index]),
                    }
                )

    summary_fieldnames = [
        "degradation",
        "method",
        "t_start",
        "ADE_mean",
        "ADE_std",
        "ADE_median",
        "ADE_p25",
        "ADE_p75",
        "RMSE_mean",
        "smooth_mean",
        "step_mean",
        "step_p95_mean",
        "step_max_mean",
        "acc_mean",
        "acc_p95_mean",
        "out_of_room_ratio_mean",
        "n_traj",
    ]
    per_traj_fieldnames = [
        "degradation",
        "method",
        "t_start",
        "traj_index",
        "ADE",
        "RMSE",
        "smooth",
        "step_mean",
        "step_p95",
        "step_max",
        "acc_mean",
        "acc_p95",
        "out_of_room_ratio",
    ]

    save_csv(SUMMARY_PATH, summary_rows, summary_fieldnames)
    save_csv(PER_TRAJ_PATH, per_traj_rows, per_traj_fieldnames)
    plot_examples(clean_eval, predictions, metrics_map)
    save_json(
        CONFIG_PATH,
        {
            "n_eval": 1000,
            "degradations": DEGRADATIONS,
            "methods": METHODS,
            "t_start_values": T_START_VALUES,
            "seeds": SEEDS,
            "checkpoint_path": "outputs/stage3_indoor/ddpm_prior/val_selected_model.pt",
            "batch_size": 128,
            "device": "cpu",
            "timesteps": 100,
            "input_representation": "relative_displacement",
            "absolute_reconstruction_start": "degraded_abs[:,0,:]",
            "seed_aggregation": "mean over absolute trajectories",
            "no_endpoint_clamp": True,
            "no_room_clip": True,
            "no_smoothing_postprocess": True,
            "no_oracle_t_start": True,
            "no_seed_best": True,
        },
    )

    print("=== Step 3.5 验证输出 ===")
    print(f"diagnostic_summary.csv 行数: {len(summary_rows)}  ← 必须为 12")
    print(f"diagnostic_per_traj.csv 行数: {len(per_traj_rows)}  ← 必须为 12000")
    print()

    print("=== Gaussian medium: ADE_mean ===")
    for method in METHODS:
        row = next(r for r in summary_rows if r["degradation"] == "gaussian_medium" and r["method"] == method)
        print(
            f"{method:20s} ADE={row['ADE_mean']:.4f}  smooth={row['smooth_mean']:.4f}  "
            f"out_room={row['out_of_room_ratio_mean']:.4f}  step_max={row['step_max_mean']:.4f}"
        )
    print()

    print("=== Burst medium: ADE_mean ===")
    for method in METHODS:
        row = next(r for r in summary_rows if r["degradation"] == "burst_medium" and r["method"] == method)
        print(
            f"{method:20s} ADE={row['ADE_mean']:.4f}  smooth={row['smooth_mean']:.4f}  "
            f"out_room={row['out_of_room_ratio_mean']:.4f}  step_max={row['step_max_mean']:.4f}"
        )
    print()

    print("=== 判读辅助 ===")
    print("若 t1/t3/t5 的 ADE 低于 noisy_input，且 out_room 没有明显升高，则 DDPM 可作为 weak refinement。")
    print("若所有 t_start 的 ADE 都高于 noisy_input，则当前 DDPM/SDEdit 接口不适合继续扩大到 Step 4。")
    print("若 t_start 越大 ADE 越差，则说明强 prior intervention 会破坏轨迹结构。")
    print("=== 等待人工确认 ===")


if __name__ == "__main__":
    main()
