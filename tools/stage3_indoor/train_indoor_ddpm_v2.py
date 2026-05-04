from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import csv
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D


DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
TRAIN_PATH = DATA_DIR / "train_trajs.npy"
VAL_PATH = DATA_DIR / "val_trajs.npy"
NORM_V2_PATH = DATA_DIR / "rel_norm_params_v2.npz"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor_v2" / "seed42"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
BEST_EMA_PATH = OUTPUT_DIR / "best_ema_model.pt"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_model.pt"
FINAL_EMA_PATH = OUTPUT_DIR / "final_ema_model.pt"
LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve.csv"
OUTPUT_NORM_V2_PATH = OUTPUT_DIR / "rel_norm_params_v2.npz"
PRIOR_CHECK_PATH = OUTPUT_DIR / "prior_check_v2.json"
FIG_PATH = OUTPUT_DIR / "sampling_check_v2.png"

CENTER = np.array([1.5, 1.5], dtype=np.float32)
SEED = 42
TIMESTEPS = 100
MAX_EPOCHS = 60
MIN_EPOCHS = 30
EARLY_STOP_PATIENCE = 12
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EMA_DECAY = 0.999
N_SAMPLE = 1000


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_torch(path: Path, payload) -> None:
    torch.save(payload, path)
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


def apply_augmentations(train_trajs: np.ndarray) -> np.ndarray:
    x = train_trajs[..., 0]
    y = train_trajs[..., 1]

    augments = [
        np.stack([x, y], axis=-1),
        np.stack([3.0 - x, y], axis=-1),
        np.stack([x, 3.0 - y], axis=-1),
        np.stack([3.0 - x, 3.0 - y], axis=-1),
        np.stack([1.5 + (y - 1.5), 1.5 - (x - 1.5)], axis=-1),
        np.stack([1.5 - (y - 1.5), 1.5 + (x - 1.5)], axis=-1),
    ]
    return np.concatenate(augments, axis=0).astype(np.float32)


def sample_unconditional(
    model: TemporalDenoiser1D,
    diffusion: DDPMForwardProcess,
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


def turning_angles_from_rel(rel: np.ndarray) -> np.ndarray:
    steps = rel
    norms = np.linalg.norm(steps, axis=-1)
    angles = []
    for i in range(steps.shape[0]):
        for t in range(steps.shape[1] - 1):
            a = steps[i, t]
            b = steps[i, t + 1]
            na = norms[i, t]
            nb = norms[i, t + 1]
            if na < 1e-6 or nb < 1e-6:
                continue
            cos_val = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            angles.append(np.degrees(np.arccos(cos_val)))
    return np.asarray(angles, dtype=np.float32)


def main() -> None:
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {TRAIN_PATH}")
    if not VAL_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {VAL_PATH}")

    train_trajs = np.load(TRAIN_PATH).astype(np.float32)
    val_trajs = np.load(VAL_PATH).astype(np.float32)
    if train_trajs.shape != (10000, 20, 2):
        raise ValueError(f"Expected train_trajs shape (10000, 20, 2), got {train_trajs.shape}")
    if val_trajs.shape != (2000, 20, 2):
        raise ValueError(f"Expected val_trajs shape (2000, 20, 2), got {val_trajs.shape}")

    augmented_trajs = apply_augmentations(train_trajs)
    if augmented_trajs.shape != (60000, 20, 2):
        raise ValueError(f"Expected augmented_trajs shape (60000, 20, 2), got {augmented_trajs.shape}")

    x_min = float(augmented_trajs[..., 0].min())
    x_max = float(augmented_trajs[..., 0].max())
    y_min = float(augmented_trajs[..., 1].min())
    y_max = float(augmented_trajs[..., 1].max())
    if x_min < -1e-4 or x_max > 3.0001 or y_min < -1e-4 or y_max > 3.0001:
        raise RuntimeError(
            f"Augmented trajectory range invalid: x=[{x_min:.6f},{x_max:.6f}], "
            f"y=[{y_min:.6f},{y_max:.6f}]"
        )

    augmented_rel = (augmented_trajs[:, 1:, :] - augmented_trajs[:, :-1, :]).astype(np.float32)
    val_rel = (val_trajs[:, 1:, :] - val_trajs[:, :-1, :]).astype(np.float32)

    rel_mean = augmented_rel.mean(axis=(0, 1)).astype(np.float32)
    rel_std = augmented_rel.std(axis=(0, 1)).astype(np.float32)
    if np.any(rel_std < 1e-8):
        raise RuntimeError(f"rel_std too small: {rel_std}")

    np.savez(NORM_V2_PATH, rel_mean=rel_mean, rel_std=rel_std)
    print(f"Saved: {NORM_V2_PATH}")

    train_rel_norm = ((augmented_rel - rel_mean) / rel_std).astype(np.float32)
    val_rel_norm = ((val_rel - rel_mean) / rel_std).astype(np.float32)

    print("=== Part A: 数据增强与标准化 ===")
    print(f"augmented_trajs shape: {augmented_trajs.shape}")
    print(f"augmented_rel shape:   {augmented_rel.shape}")
    print(f"val_rel shape:         {val_rel.shape}")
    print(f"augmented x range: [{x_min:.4f}, {x_max:.4f}]")
    print(f"augmented y range: [{y_min:.4f}, {y_max:.4f}]")
    print(f"augmented rel_mean: ({rel_mean[0]:.6f}, {rel_mean[1]:.6f})  预期接近 (0, 0)")
    print(f"augmented rel_std:  ({rel_std[0]:.6f}, {rel_std[1]:.6f})")
    print("rel_norm_params_v2 saved to data/stage3_indoor/rel_norm_params_v2.npz")

    set_all_seeds(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_NORM_V2_PATH, rel_mean=rel_mean, rel_std=rel_std)
    print(f"Saved: {OUTPUT_NORM_V2_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = DDPMForwardProcess(timesteps=TIMESTEPS, device=device)

    model = TemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=2, hidden_dim=128).to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_tensor = torch.from_numpy(train_rel_norm.transpose(0, 2, 1))
    val_tensor = torch.from_numpy(val_rel_norm.transpose(0, 2, 1))
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    val_rng = np.random.default_rng(SEED + 1000)
    val_t_all = torch.from_numpy(val_rng.integers(0, TIMESTEPS, size=(val_tensor.shape[0],), dtype=np.int64)).to(
        device=device, dtype=torch.long
    )
    val_noise_all = torch.from_numpy(
        val_rng.normal(0.0, 1.0, size=(val_tensor.shape[0], 2, 19)).astype(np.float32)
    ).to(device=device, dtype=torch.float32)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    final_val_loss = float("nan")
    stopped_epoch = MAX_EPOCHS
    epochs_since_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for (batch_x,) in train_loader:
            x0 = batch_x.to(device=device, dtype=torch.float32)
            t = diffusion.sample_timesteps(batch_size=x0.shape[0])
            x_t, noise = diffusion.q_sample(x0, t)

            optimizer.zero_grad()
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(model_p.data, alpha=1.0 - EMA_DECAY)

            train_total += loss.item() * x0.shape[0]
            train_count += x0.shape[0]

        train_loss = train_total / train_count

        ema_model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for start in range(0, val_tensor.shape[0], BATCH_SIZE):
                end = min(start + BATCH_SIZE, val_tensor.shape[0])
                x0 = val_tensor[start:end].to(device=device, dtype=torch.float32)
                t = val_t_all[start:end]
                noise = val_noise_all[start:end]
                x_t, _ = diffusion.q_sample(x0, t, noise=noise)
                eps_pred = ema_model(x_t, t)
                loss = F.mse_loss(eps_pred, noise)
                val_total += loss.item() * x0.shape[0]
                val_count += x0.shape[0]
        val_loss_ema = val_total / val_count
        final_val_loss = val_loss_ema

        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_loss_ema": f"{val_loss_ema:.8f}",
            }
        )

        improved = val_loss_ema < best_val_loss
        if improved:
            best_val_loss = val_loss_ema
            best_epoch = epoch
            epochs_since_improve = 0
            save_torch(BEST_MODEL_PATH, model.state_dict())
            save_torch(BEST_EMA_PATH, ema_model.state_dict())
        else:
            epochs_since_improve += 1

        print(f"Epoch {epoch:03d}/{MAX_EPOCHS}  train_loss={train_loss:.6f}  val_loss_ema={val_loss_ema:.6f}")

        if epoch >= MIN_EPOCHS and epochs_since_improve >= EARLY_STOP_PATIENCE:
            stopped_epoch = epoch
            break

    save_torch(FINAL_MODEL_PATH, model.state_dict())
    save_torch(FINAL_EMA_PATH, ema_model.state_dict())
    save_csv(LOSS_CURVE_PATH, history, ["epoch", "train_loss", "val_loss_ema"])

    best_ema_model = TemporalDenoiser1D(max_timesteps=TIMESTEPS, in_channels=2, hidden_dim=128).to(device)
    best_ema_model.load_state_dict(torch.load(BEST_EMA_PATH, map_location=device))
    best_ema_model.eval()

    print("=== Part C1: Loss 检查 ===")
    print(f"best epoch:      {best_epoch}")
    print(f"best val_loss:   {best_val_loss:.6f}")
    print(f"final val_loss:  {final_val_loss:.6f}")
    print(f"stopped epoch:   {stopped_epoch}")

    print("=== Part C2: One-step denoise 检查 ===")
    print(f"{'t':>5} {'noisy_mse':>12} {'denoised_mse':>14} {'improve_%':>10}")
    one_step_results = {}
    val_check = torch.from_numpy(val_rel_norm[:200].transpose(0, 2, 1)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for t_value in [3, 5, 10, 20, 50]:
            t = torch.full((val_check.shape[0],), t_value, device=device, dtype=torch.long)
            noise = torch.randn_like(val_check)
            x_t, _ = diffusion.q_sample(val_check, t, noise=noise)
            noise_pred = best_ema_model(x_t, t)
            alpha_bar_t = diffusion.alpha_bars[t_value]
            x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            noisy_mse = float(F.mse_loss(x_t, val_check).item())
            denoised_mse = float(F.mse_loss(x0_hat, val_check).item())
            improve_pct = (noisy_mse - denoised_mse) / noisy_mse * 100.0
            one_step_results[t_value] = improve_pct
            print(f"{t_value:5d} {noisy_mse:12.6f} {denoised_mse:14.6f} {improve_pct:10.2f}")

    gen_rel_norm = sample_unconditional(best_ema_model, diffusion, N_SAMPLE, device)
    gen_rel = (gen_rel_norm * rel_std[None, None, :] + rel_mean[None, None, :]).astype(np.float32)
    real_rel = val_rel[:1000].astype(np.float32)

    center_starts = np.repeat(CENTER[None, :], N_SAMPLE, axis=0).astype(np.float32)
    gen_abs = np.zeros((N_SAMPLE, 20, 2), dtype=np.float32)
    gen_abs[:, 0, :] = center_starts
    gen_abs[:, 1:, :] = center_starts[:, None, :] + np.cumsum(gen_rel, axis=1)

    real_step = np.linalg.norm(real_rel, axis=-1)
    gen_step = np.linalg.norm(gen_rel, axis=-1)
    real_acc = np.linalg.norm(real_rel[:, 1:, :] - real_rel[:, :-1, :], axis=-1)
    gen_acc = np.linalg.norm(gen_rel[:, 1:, :] - gen_rel[:, :-1, :], axis=-1)

    real_step_mean = float(real_step.mean())
    gen_step_mean = float(gen_step.mean())
    real_step_p95 = float(np.percentile(real_step, 95))
    gen_step_p95 = float(np.percentile(gen_step, 95))
    real_step_p99 = float(np.percentile(real_step, 99))
    gen_step_p99 = float(np.percentile(gen_step, 99))
    gen_step_max = float(gen_step.max())
    real_acc_mean = float(real_acc.mean())
    gen_acc_mean = float(gen_acc.mean())
    large_step_ratio = float(np.mean(gen_step > 0.6))
    out_of_room_ratio = float(
        np.mean(
            np.any(
                (gen_abs[..., 0] < 0.0)
                | (gen_abs[..., 0] > 3.0)
                | (gen_abs[..., 1] < 0.0)
                | (gen_abs[..., 1] > 3.0),
                axis=1,
            )
        )
    )

    print("=== Part C3: 采样分布检查 (EMA model) ===")
    print(f"real step mean:  {real_step_mean:.4f}")
    print(f"gen  step mean:  {gen_step_mean:.4f}")
    print(f"real step p95:   {real_step_p95:.4f}")
    print(f"gen  step p95:   {gen_step_p95:.4f}")
    print(f"real step p99:   {real_step_p99:.4f}")
    print(f"gen  step p99:   {gen_step_p99:.4f}")
    print(f"gen  step max:   {gen_step_max:.4f}")
    print(f"real acc mean:   {real_acc_mean:.4f}")
    print(f"gen  acc mean:   {gen_acc_mean:.4f}")
    print(f"large_step_ratio_0p6: {large_step_ratio:.4f}")
    print(f"out_of_room_ratio_center_start: {out_of_room_ratio:.4f}  诊断项，不作为否决指标")

    gen_dx_mean = float(gen_rel[..., 0].mean())
    gen_dy_mean = float(gen_rel[..., 1].mean())
    real_dx_mean = float(real_rel[..., 0].mean())
    real_dy_mean = float(real_rel[..., 1].mean())
    dx_bias_gap = abs(gen_dx_mean - real_dx_mean)
    dy_bias_gap = abs(gen_dy_mean - real_dy_mean)

    print("=== Part C4: 方向偏置检查 ===")
    print(f"DDPM generated  dx_mean={gen_dx_mean:.6f}  dy_mean={gen_dy_mean:.6f}")
    print(f"Real val set    dx_mean={real_dx_mean:.6f}  dy_mean={real_dy_mean:.6f}")
    print(f"bias gap        dx_gap={dx_bias_gap:.6f}  dy_gap={dy_bias_gap:.6f}")

    criterion_1 = gen_step_p95 <= real_step_p95 * 1.5
    criterion_2 = gen_step_p99 <= real_step_p99 * 2.0
    criterion_3 = large_step_ratio < 0.005
    criterion_4 = gen_acc_mean <= real_acc_mean * 1.5
    criterion_5 = dx_bias_gap < 0.01 and dy_bias_gap < 0.01
    criterion_6 = all(one_step_results[t] > 0.0 for t in [5, 10, 20])
    passed = all([criterion_1, criterion_2, criterion_3, criterion_4, criterion_5, criterion_6])

    print("=== Part C5: Prior pass/fail 判定 ===")
    print(f"gen_step_p95 <= real_step_p95 * 1.5: {criterion_1}")
    print(f"gen_step_p99 <= real_step_p99 * 2.0: {criterion_2}")
    print(f"large_step_ratio < 0.005: {criterion_3}")
    print(f"gen_acc_mean <= real_acc_mean * 1.5: {criterion_4}")
    print(f"direction bias gap < 0.01: {criterion_5}")
    print(f"one-step denoise t=5/10/20 all positive: {criterion_6}")
    if passed:
        print("✅ PRIOR V2 PASSED — 可以进入小范围 SDEdit 诊断")
    else:
        print("❌ PRIOR V2 NOT PASSED — 不建议进入 Step 4，需要继续修 prior")

    save_json(
        PRIOR_CHECK_PATH,
        {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "final_val_loss": float(final_val_loss),
            "stopped_epoch": int(stopped_epoch),
            "real_step_mean": real_step_mean,
            "gen_step_mean": gen_step_mean,
            "real_step_p95": real_step_p95,
            "gen_step_p95": gen_step_p95,
            "real_step_p99": real_step_p99,
            "gen_step_p99": gen_step_p99,
            "gen_step_max": gen_step_max,
            "real_acc_mean": real_acc_mean,
            "gen_acc_mean": gen_acc_mean,
            "large_step_ratio_0p6": large_step_ratio,
            "out_of_room_ratio_center_start": out_of_room_ratio,
            "gen_dx_mean": gen_dx_mean,
            "gen_dy_mean": gen_dy_mean,
            "real_dx_mean": real_dx_mean,
            "real_dy_mean": real_dy_mean,
            "dx_bias_gap": dx_bias_gap,
            "dy_bias_gap": dy_bias_gap,
            "pass_criteria": {
                "criterion_1": criterion_1,
                "criterion_2": criterion_2,
                "criterion_3": criterion_3,
                "criterion_4": criterion_4,
                "criterion_5": criterion_5,
                "criterion_6": criterion_6,
            },
            "passed": passed,
        },
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    rng = np.random.default_rng(SEED)
    gen_idx = rng.choice(gen_abs.shape[0], size=100, replace=False)
    real_idx = rng.choice(val_trajs[:1000].shape[0], size=100, replace=False)

    for idx in gen_idx:
        axes[0, 0].plot(gen_abs[idx, :, 0], gen_abs[idx, :, 1], color="red", alpha=0.18, linewidth=1.0)
    axes[0, 0].plot([0, 3, 3, 0, 0], [0, 0, 3, 3, 0], color="black", linewidth=1.2)
    axes[0, 0].set_xlim(-0.5, 3.5)
    axes[0, 0].set_ylim(-0.5, 3.5)
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_title("Generated Trajectories")

    real_abs_subset = val_trajs[:1000]
    for idx in real_idx:
        axes[0, 1].plot(real_abs_subset[idx, :, 0], real_abs_subset[idx, :, 1], color="blue", alpha=0.18, linewidth=1.0)
    axes[0, 1].plot([0, 3, 3, 0, 0], [0, 0, 3, 3, 0], color="black", linewidth=1.2)
    axes[0, 1].set_xlim(-0.5, 3.5)
    axes[0, 1].set_ylim(-0.5, 3.5)
    axes[0, 1].set_aspect("equal", adjustable="box")
    axes[0, 1].set_title("Real Validation Trajectories")

    axes[0, 2].hist(real_step.reshape(-1), bins=50, alpha=0.5, color="blue", label="real")
    axes[0, 2].hist(gen_step.reshape(-1), bins=50, alpha=0.5, color="red", label="generated")
    axes[0, 2].set_title("Step-size Histogram")
    axes[0, 2].legend()

    axes[1, 0].hist(real_acc.reshape(-1), bins=50, alpha=0.5, color="blue", label="real")
    axes[1, 0].hist(gen_acc.reshape(-1), bins=50, alpha=0.5, color="red", label="generated")
    axes[1, 0].set_title("Acceleration Histogram")
    axes[1, 0].legend()

    real_points = real_rel.reshape(-1, 2)
    gen_points = gen_rel.reshape(-1, 2)
    real_pick = rng.choice(real_points.shape[0], size=2000, replace=False)
    gen_pick = rng.choice(gen_points.shape[0], size=2000, replace=False)
    axes[1, 1].scatter(real_points[real_pick, 0], real_points[real_pick, 1], s=8, alpha=0.35, color="blue", label="real")
    axes[1, 1].scatter(gen_points[gen_pick, 0], gen_points[gen_pick, 1], s=8, alpha=0.35, color="red", label="generated")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlim(-0.6, 0.6)
    axes[1, 1].set_ylim(-0.6, 0.6)
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].legend()
    axes[1, 1].set_title("Displacement Scatter")

    epochs = [int(row["epoch"]) for row in history]
    train_losses = [float(row["train_loss"]) for row in history]
    val_losses = [float(row["val_loss_ema"]) for row in history]
    axes[1, 2].plot(epochs, train_losses, color="tab:blue", label="train_loss")
    axes[1, 2].plot(epochs, val_losses, color="tab:orange", label="val_loss_ema")
    axes[1, 2].set_title("Loss Curve")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")


if __name__ == "__main__":
    main()
