from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tools.stage3_indoor.train_ddpm_prior import DDPMProcess, HIDDEN_DIM, TemporalDenoiser1D, TIMESTEPS


DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
TRAIN_PATH = DATA_DIR / "train_trajs.npy"
VAL_PATH = DATA_DIR / "val_trajs.npy"
NORM_PATH = DATA_DIR / "rel_norm_params.npz"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_indoor" / "seed42"
BEST_PATH = OUTPUT_DIR / "best_model.pt"
FINAL_PATH = OUTPUT_DIR / "final_model.pt"
LOSS_PATH = OUTPUT_DIR / "loss_curve.csv"
OUTPUT_NORM_PATH = OUTPUT_DIR / "rel_norm_params.npz"
FIG_PATH = OUTPUT_DIR / "sampling_check.png"

SEED = 42
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
N_GEN = 1000


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


def turning_angles(abs_trajs: np.ndarray) -> np.ndarray:
    steps = abs_trajs[:, 1:, :] - abs_trajs[:, :-1, :]
    norms = np.linalg.norm(steps, axis=-1)
    angles = []
    for i in range(abs_trajs.shape[0]):
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


def main() -> None:
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {TRAIN_PATH}")
    if not VAL_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {VAL_PATH}")

    train = np.load(TRAIN_PATH).astype(np.float32)
    val = np.load(VAL_PATH).astype(np.float32)
    if train.shape != (10000, 20, 2):
        raise ValueError(f"Expected train shape (10000, 20, 2), got {train.shape}")
    if val.shape != (2000, 20, 2):
        raise ValueError(f"Expected val shape (2000, 20, 2), got {val.shape}")

    train_rel = (train[:, 1:, :] - train[:, :-1, :]).astype(np.float32)
    val_rel = (val[:, 1:, :] - val[:, :-1, :]).astype(np.float32)

    rel_mean = train_rel.mean(axis=(0, 1)).astype(np.float32)
    rel_std = train_rel.std(axis=(0, 1)).astype(np.float32)
    rel_std_safe = np.maximum(rel_std, 1e-6).astype(np.float32)

    train_rel_norm = ((train_rel - rel_mean) / rel_std_safe).astype(np.float32)
    val_rel_norm = ((val_rel - rel_mean) / rel_std_safe).astype(np.float32)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(NORM_PATH, rel_mean=rel_mean, rel_std=rel_std, rel_std_safe=rel_std_safe)
    print(f"Saved: {NORM_PATH}")
    np.savez(OUTPUT_NORM_PATH, rel_mean=rel_mean, rel_std=rel_std, rel_std_safe=rel_std_safe)
    print(f"Saved: {OUTPUT_NORM_PATH}")

    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = DDPMProcess(timesteps=TIMESTEPS, device=device)
    model = TemporalDenoiser1D(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    final_val_loss = float("nan")
    final_train_loss = float("nan")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for (batch_x,) in train_loader:
            x0 = batch_x.to(device=device, dtype=torch.float32)
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)

            optimizer.zero_grad()
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)
            loss.backward()
            optimizer.step()

            train_total += loss.item() * x0.shape[0]
            train_count += x0.shape[0]

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                x0 = batch_x.to(device=device, dtype=torch.float32)
                t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=device, dtype=torch.long)
                noise = torch.randn_like(x0)
                x_t = diffusion.q_sample(x0, t, noise)
                eps_pred = model(x_t, t)
                loss = F.mse_loss(eps_pred, noise)
                val_total += loss.item() * x0.shape[0]
                val_count += x0.shape[0]

        train_loss = train_total / train_count
        val_loss = val_total / val_count
        final_train_loss = train_loss
        final_val_loss = val_loss
        history.append({"epoch": epoch, "train_loss": f"{train_loss:.8f}", "val_loss": f"{val_loss:.8f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_torch(BEST_PATH, model.state_dict())

        print(f"Epoch {epoch:03d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    save_torch(FINAL_PATH, model.state_dict())
    save_csv(LOSS_PATH, history, ["epoch", "train_loss", "val_loss"])

    best_model = TemporalDenoiser1D(hidden_dim=HIDDEN_DIM).to(device)
    best_model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    best_model.eval()

    print("=== Part C1: Loss 检查 ===")
    print(f"best epoch:     {best_epoch}")
    print(f"best val_loss:  {best_val_loss:.6f}")
    print(f"final val_loss: {final_val_loss:.6f}")

    print("=== Part C2: One-step denoise 检查 ===")
    print(f"{'t':>5} {'noisy_mse':>12} {'denoised_mse':>14} {'improve_%':>10}")
    val_check = torch.from_numpy(val_rel_norm[:200].transpose(0, 2, 1)).to(device=device, dtype=torch.float32)
    for t_value in [3, 5, 10, 20, 50]:
        t = torch.full((val_check.shape[0],), t_value, device=device, dtype=torch.long)
        noise = torch.randn_like(val_check)
        alpha_bar_t = diffusion.alpha_bars[t_value]
        x_t = diffusion.q_sample(val_check, t, noise)
        with torch.no_grad():
            noise_pred = best_model(x_t, t)
            x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        noisy_error = float(F.mse_loss(x_t, val_check).item())
        denoised_error = float(F.mse_loss(x0_pred, val_check).item())
        improvement = noisy_error - denoised_error
        improve_pct = improvement / noisy_error * 100.0
        print(f"{t_value:5d} {noisy_error:12.6f} {denoised_error:14.6f} {improve_pct:10.2f}")

    gen_rel_norm = unconditional_sample(best_model, diffusion, N_GEN, device)
    gen_rel = (gen_rel_norm * rel_std_safe[None, None, :] + rel_mean[None, None, :]).astype(np.float32)
    rng = np.random.default_rng(42)
    start_xy = rng.uniform(0.3, 2.7, size=(N_GEN, 2)).astype(np.float32)
    gen_abs = np.zeros((N_GEN, 20, 2), dtype=np.float32)
    gen_abs[:, 0, :] = start_xy
    gen_abs[:, 1:, :] = start_xy[:, None, :] + np.cumsum(gen_rel, axis=1)

    real_abs = val
    real_rel = val_rel

    real_step = np.linalg.norm(real_rel, axis=-1)
    gen_step = np.linalg.norm(gen_rel, axis=-1)
    real_acc = np.linalg.norm(real_abs[:, 2:, :] - 2 * real_abs[:, 1:-1, :] + real_abs[:, :-2, :], axis=-1)
    gen_acc = np.linalg.norm(gen_abs[:, 2:, :] - 2 * gen_abs[:, 1:-1, :] + gen_abs[:, :-2, :], axis=-1)

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
    oor_ratio = float(
        np.mean(
            (gen_abs[..., 0] < 0.0)
            | (gen_abs[..., 0] > 3.0)
            | (gen_abs[..., 1] < 0.0)
            | (gen_abs[..., 1] > 3.0)
        )
    )

    print("=== Part C3: 采样分布检查 ===")
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
    print(f"out_of_room_ratio:    {oor_ratio:.4f}")

    cond1 = gen_step_p95 <= real_step_p95 * 1.5
    cond2 = gen_step_p99 <= real_step_p99 * 2.0
    cond3 = large_step_ratio < 0.005
    cond4 = oor_ratio < 0.10
    cond5 = gen_acc_mean <= real_acc_mean * 1.5
    pass_criteria = cond1 and cond2 and cond3 and cond4 and cond5

    print("=== Part C4: Prior pass/fail 判定 ===")
    print(f"gen_step_p95 <= real_step_p95 * 1.5: {cond1}")
    print(f"gen_step_p99 <= real_step_p99 * 2.0: {cond2}")
    print(f"large_step_ratio < 0.005: {cond3}")
    print(f"out_of_room_ratio < 0.10: {cond4}")
    print(f"gen_acc_mean <= real_acc_mean * 1.5: {cond5}")
    if pass_criteria:
        print("✅ PRIOR PASSED — 可以进入 Step 4")
    else:
        print("❌ PRIOR NOT PASSED — 需要进一步调整")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    rng_vis = np.random.default_rng(42)
    gen_idx = rng_vis.choice(gen_abs.shape[0], size=100, replace=False)
    real_idx = rng_vis.choice(real_abs.shape[0], size=100, replace=False)

    for idx in gen_idx:
        axes[0, 0].plot(gen_abs[idx, :, 0], gen_abs[idx, :, 1], color="red", alpha=0.18, linewidth=1.0)
    axes[0, 0].plot([0, 3, 3, 0, 0], [0, 0, 3, 3, 0], color="black", linewidth=1.2)
    axes[0, 0].set_xlim(-0.5, 3.5)
    axes[0, 0].set_ylim(-0.5, 3.5)
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_title("Generated Overlay")

    for idx in real_idx:
        axes[0, 1].plot(real_abs[idx, :, 0], real_abs[idx, :, 1], color="gray", alpha=0.18, linewidth=1.0)
    axes[0, 1].plot([0, 3, 3, 0, 0], [0, 0, 3, 3, 0], color="black", linewidth=1.2)
    axes[0, 1].set_xlim(-0.5, 3.5)
    axes[0, 1].set_ylim(-0.5, 3.5)
    axes[0, 1].set_aspect("equal", adjustable="box")
    axes[0, 1].set_title("Real Overlay")

    axes[0, 2].hist(real_step.reshape(-1), bins=50, alpha=0.5, label="real", color="gray")
    axes[0, 2].hist(gen_step.reshape(-1), bins=50, alpha=0.5, label="generated", color="red")
    axes[0, 2].set_title("Step Size Histogram")
    axes[0, 2].legend()

    real_angles = turning_angles(real_abs)
    gen_angles = turning_angles(gen_abs)
    axes[1, 0].hist(real_angles, bins=50, alpha=0.5, label="real", color="gray")
    axes[1, 0].hist(gen_angles, bins=50, alpha=0.5, label="generated", color="red")
    axes[1, 0].set_title("Turning Angle Histogram")
    axes[1, 0].legend()

    axes[1, 1].hist(real_acc.reshape(-1), bins=50, alpha=0.5, label="real", color="gray")
    axes[1, 1].hist(gen_acc.reshape(-1), bins=50, alpha=0.5, label="generated", color="red")
    axes[1, 1].set_title("Acceleration Histogram")
    axes[1, 1].legend()

    epochs = [int(row["epoch"]) for row in history]
    train_losses = [float(row["train_loss"]) for row in history]
    val_losses = [float(row["val_loss"]) for row in history]
    axes[1, 2].plot(epochs, train_losses, label="train_loss", color="tab:blue")
    axes[1, 2].plot(epochs, val_losses, label="val_loss", color="tab:orange")
    axes[1, 2].set_title("Loss Curve")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIG_PATH}")


if __name__ == "__main__":
    main()
