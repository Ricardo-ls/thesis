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

from tools.stage3_indoor.train_ddpm_prior import DDPMProcess, HIDDEN_DIM, TemporalDenoiser1D, TIMESTEPS


CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor" / "seed42" / "best_model.pt"
NORM_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "rel_norm_params.npz"
VAL_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "val_trajs.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor" / "seed42"
SUMMARY_PATH = OUTPUT_DIR / "oor_diagnosis_summary.csv"
FIG_PATH = OUTPUT_DIR / "oor_diagnosis.png"

N = 1000


def unconditional_sample(
    model: TemporalDenoiser1D,
    diffusion: DDPMProcess,
    num_samples: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.randn((num_samples, 2, 19), device=device, dtype=torch.float32)
        for t_idx in reversed(range(TIMESTEPS)):
            t = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
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
    return x_t.permute(0, 2, 1).cpu().numpy().astype(np.float32)


def reconstruct_abs(starts: np.ndarray, rel: np.ndarray) -> np.ndarray:
    abs_trajs = np.zeros((rel.shape[0], 20, 2), dtype=np.float32)
    abs_trajs[:, 0, :] = starts.astype(np.float32)
    abs_trajs[:, 1:, :] = starts[:, None, :] + np.cumsum(rel, axis=1)
    return abs_trajs


def summarize_strategy(name: str, abs_trajs: np.ndarray) -> dict:
    x_vals = abs_trajs[..., 0]
    y_vals = abs_trajs[..., 1]

    left_violation = np.maximum(0.0, -np.min(x_vals, axis=1))
    right_violation = np.maximum(0.0, np.max(x_vals, axis=1) - 3.0)
    bottom_violation = np.maximum(0.0, -np.min(y_vals, axis=1))
    top_violation = np.maximum(0.0, np.max(y_vals, axis=1) - 3.0)
    max_deviation = np.maximum.reduce(
        [left_violation, right_violation, bottom_violation, top_violation]
    ).astype(np.float32)

    oor_mask = max_deviation > 0.0
    endpoint_drift = np.linalg.norm(abs_trajs[:, -1, :] - abs_trajs[:, 0, :], axis=1)

    if np.any(oor_mask):
        oor_mean_dev = float(max_deviation[oor_mask].mean())
    else:
        oor_mean_dev = 0.0

    return {
        "strategy": name,
        "oor_ratio": float(np.mean(oor_mask)),
        "oor_mean_dev": oor_mean_dev,
        "oor_max_dev": float(np.max(max_deviation)),
        "endpoint_drift_mean": float(np.mean(endpoint_drift)),
        "endpoint_drift_p95": float(np.percentile(endpoint_drift, 95)),
        "_oor_mask": oor_mask,
        "_abs_trajs": abs_trajs,
    }


def save_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "strategy",
        "oor_ratio",
        "oor_mean_dev",
        "oor_max_dev",
        "endpoint_drift_mean",
        "endpoint_drift_p95",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"Saved: {path}")


def plot_room_box(ax) -> None:
    ax.plot([0, 3, 3, 0, 0], [0, 0, 3, 3, 0], color="black", linewidth=1.2)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def plot_overlay(ax, title: str, abs_trajs: np.ndarray, oor_mask: np.ndarray) -> None:
    for traj, is_oor in zip(abs_trajs[:100], oor_mask[:100]):
        color = "red" if is_oor else "blue"
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.35, linewidth=1.0)
    plot_room_box(ax)
    ax.set_title(title)


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {CHECKPOINT_PATH}")
    if not NORM_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {NORM_PATH}")
    if not VAL_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {VAL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    norm_params = np.load(NORM_PATH)
    rel_mean = norm_params["rel_mean"].astype(np.float32)
    rel_std = norm_params["rel_std"].astype(np.float32)

    val_trajs = np.load(VAL_PATH).astype(np.float32)
    if val_trajs.shape != (2000, 20, 2):
        raise ValueError(f"Expected val_trajs shape (2000, 20, 2), got {val_trajs.shape}")

    real_abs = val_trajs[:N]
    real_rel = (real_abs[:, 1:, :] - real_abs[:, :-1, :]).astype(np.float32)

    device = torch.device("cpu")
    diffusion = DDPMProcess(timesteps=TIMESTEPS, device=device)
    model = TemporalDenoiser1D(hidden_dim=HIDDEN_DIM).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    gen_rel_norm = unconditional_sample(model, diffusion, N, device)
    if gen_rel_norm.shape != (1000, 19, 2):
        raise ValueError(f"Expected generated rel norm shape (1000, 19, 2), got {gen_rel_norm.shape}")
    gen_rel = (gen_rel_norm * rel_std[None, None, :] + rel_mean[None, None, :]).astype(np.float32)

    rng = np.random.default_rng(42)
    random_starts = rng.uniform(0.3, 2.7, size=(N, 2)).astype(np.float32)
    center_starts = np.full((N, 2), 1.5, dtype=np.float32)
    real_starts = real_abs[:, 0, :].astype(np.float32)

    random_abs = reconstruct_abs(random_starts, gen_rel)
    center_abs = reconstruct_abs(center_starts, gen_rel)
    real_start_abs = reconstruct_abs(real_starts, gen_rel)
    real_disp_random_abs = reconstruct_abs(random_starts, real_rel)

    rows = [
        summarize_strategy("random_start", random_abs),
        summarize_strategy("center_start", center_abs),
        summarize_strategy("real_start", real_start_abs),
        summarize_strategy("real_displacement_random_start", real_disp_random_abs),
    ]

    save_summary_csv(SUMMARY_PATH, rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    plot_overlay(axes[0, 0], "random_start", rows[0]["_abs_trajs"], rows[0]["_oor_mask"])
    plot_overlay(axes[0, 1], "center_start", rows[1]["_abs_trajs"], rows[1]["_oor_mask"])
    plot_overlay(axes[1, 0], "real_start", rows[2]["_abs_trajs"], rows[2]["_oor_mask"])

    scatter_rng = np.random.default_rng(123)
    real_points = real_rel.reshape(-1, 2)
    gen_points = gen_rel.reshape(-1, 2)
    real_pick = scatter_rng.choice(real_points.shape[0], size=2000, replace=False)
    gen_pick = scatter_rng.choice(gen_points.shape[0], size=2000, replace=False)
    axes[1, 1].scatter(real_points[real_pick, 0], real_points[real_pick, 1], s=8, alpha=0.35, color="blue", label="real")
    axes[1, 1].scatter(gen_points[gen_pick, 0], gen_points[gen_pick, 1], s=8, alpha=0.35, color="red", label="generated")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlim(-0.6, 0.6)
    axes[1, 1].set_ylim(-0.6, 0.6)
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend()
    axes[1, 1].set_title("Displacement Direction Scatter")

    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")

    dx_mean = float(gen_rel[..., 0].mean())
    dy_mean = float(gen_rel[..., 1].mean())
    real_dx_mean = float(real_rel[..., 0].mean())
    real_dy_mean = float(real_rel[..., 1].mean())

    random_oor = rows[0]["oor_ratio"]
    center_oor = rows[1]["oor_ratio"]

    print("=== Step 3c 归因诊断 ===")
    print("")
    print("起点策略对比：")
    print(f"{'strategy':30s}  OOR_ratio  mean_dev  max_dev  endpoint_drift_mean  endpoint_drift_p95")
    for row in rows:
        print(
            f"{row['strategy']:30s}  {row['oor_ratio']:.4f}    {row['oor_mean_dev']:.4f}    "
            f"{row['oor_max_dev']:.4f}    {row['endpoint_drift_mean']:.4f}              "
            f"{row['endpoint_drift_p95']:.4f}"
        )

    print("")
    print("方向偏置检查：")
    print(f"DDPM generated  dx_mean={dx_mean:.6f}  dy_mean={dy_mean:.6f}")
    print(f"Real val set    dx_mean={real_dx_mean:.6f}  dy_mean={real_dy_mean:.6f}")

    print("")
    print("=== 诊断结论 ===")
    if center_oor < random_oor * 0.5:
        print("主因：起点采样位置。center_start OOR 显著低于 random_start。")
        print("Prior 的 relative displacement 本身基本合理，")
        print("出界主要因为起点靠近边界时余量不足。")
        print("→ 建议：进入 Step 4 时用退化轨迹的真实起点，不影响 refinement 实验。")
    elif abs(dx_mean) > 0.005 or abs(dy_mean) > 0.005:
        print("主因：方向性漂移。生成的 displacement 有系统偏置。")
        print(f"→ dx_mean={dx_mean:.6f}, dy_mean={dy_mean:.6f}")
        print("→ 建议：检查训练数据是否有方向不平衡，或增加数据增强。")
    else:
        print("主因：累积方差。displacement 无偏但方差累积导致扩散。")
        print("→ 建议：prior 本身可用于 refinement（SDEdit t_start 小时），")
        print("         但 unconditional 完整采样不适合直接生成室内轨迹。")


if __name__ == "__main__":
    main()
