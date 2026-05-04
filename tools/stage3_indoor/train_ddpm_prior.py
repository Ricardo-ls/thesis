import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_prior"

RANDOM_SEED = 42
VAL_PROTOCOL_SEED = 1042
TIMESTEPS = 100
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
HIDDEN_DIM = 128
TRAIN_SLICE = slice(1000, 1800)
VAL_SLICE = slice(1800, 2000)
N_GEN = 200


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


def save_numpy(path: Path, array: np.ndarray) -> None:
    np.save(path, array.astype(np.float32))
    print(f"Saved: {path}")


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def save_loss_curve(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved: {path}")


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    freq_exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * freq_exponent
    )
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TemporalDenoiser1D(nn.Module):
    def __init__(self, hidden_dim: int = 128, time_embed_dim: int = 128):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(hidden_dim, 2, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = sinusoidal_timestep_embedding(t, self.time_embed_dim)
        t_hidden = self.time_mlp(t_embed).unsqueeze(-1)

        h = F.silu(self.conv1(x_t))
        h = F.silu(self.conv2(h))
        h = h + t_hidden
        h = F.silu(self.conv3(h))
        return self.conv4(h)


class DDPMProcess:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise


def compute_fixed_val_protocol(val_n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(VAL_PROTOCOL_SEED)
    val_t_all = rng.integers(0, TIMESTEPS, size=(val_n,), dtype=np.int64)
    val_noise_all = rng.normal(0.0, 1.0, size=(val_n, 2, 19)).astype(np.float32)
    return val_t_all, val_noise_all


def compute_val_loss(
    model: nn.Module,
    diffusion: DDPMProcess,
    val_x: torch.Tensor,
    val_t_all: torch.Tensor,
    val_noise_all: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, val_x.shape[0], batch_size):
            end = min(start + batch_size, val_x.shape[0])
            x0 = val_x[start:end]
            t = val_t_all[start:end]
            noise = val_noise_all[start:end]
            x_t = diffusion.q_sample(x0, t, noise)
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)
            total_loss += loss.item() * x0.shape[0]
            total_count += x0.shape[0]
    return total_loss / total_count


def summarize_flat(values: np.ndarray, prefix: str) -> dict[str, float]:
    flat = values.reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_p50": float(np.percentile(flat, 50)),
        f"{prefix}_p75": float(np.percentile(flat, 75)),
        f"{prefix}_p90": float(np.percentile(flat, 90)),
        f"{prefix}_p95": float(np.percentile(flat, 95)),
        f"{prefix}_max": float(np.max(flat)),
    }


def plot_generated_examples(generated_abs: np.ndarray, val_abs: np.ndarray, output_path: Path) -> None:
    rng = np.random.default_rng(420)
    indices = rng.choice(generated_abs.shape[0], size=16, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    for ax, idx in zip(axes.flatten(), indices):
        ax.plot(val_abs[idx, :, 0], val_abs[idx, :, 1], color="lightgray", linewidth=1.4, label="val_ref")
        ax.plot(generated_abs[idx, :, 0], generated_abs[idx, :, 1], color="tab:blue", linewidth=1.8, label="generated")
        ax.scatter(generated_abs[idx, 0, 0], generated_abs[idx, 0, 1], color="green", s=25, marker="o")
        ax.scatter(generated_abs[idx, -1, 0], generated_abs[idx, -1, 1], color="red", s=25, marker="s")
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"idx={idx}", fontsize=10)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {DATA_PATH}")

    clean_trajs = np.load(DATA_PATH)
    if clean_trajs.shape != (2000, 20, 2):
        raise ValueError(f"Expected clean_trajs shape (2000, 20, 2), got {clean_trajs.shape}")

    set_all_seeds(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_abs = clean_trajs[TRAIN_SLICE].astype(np.float32)
    val_abs = clean_trajs[VAL_SLICE].astype(np.float32)
    train_rel = (train_abs[:, 1:, :] - train_abs[:, :-1, :]).astype(np.float32)
    val_rel = (val_abs[:, 1:, :] - val_abs[:, :-1, :]).astype(np.float32)

    if train_rel.shape != (800, 19, 2):
        raise ValueError(f"Expected train_rel shape (800, 19, 2), got {train_rel.shape}")
    if val_rel.shape != (200, 19, 2):
        raise ValueError(f"Expected val_rel shape (200, 19, 2), got {val_rel.shape}")

    train_tensor = torch.from_numpy(train_rel.transpose(0, 2, 1))
    val_tensor = torch.from_numpy(val_rel.transpose(0, 2, 1))

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    example_batch_shape = tuple(next(iter(train_loader))[0].shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = DDPMProcess(timesteps=TIMESTEPS, device=device)
    model = TemporalDenoiser1D(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    val_t_np, val_noise_np = compute_fixed_val_protocol(val_n=val_tensor.shape[0])
    val_x = val_tensor.to(device)
    val_t_all = torch.from_numpy(val_t_np).to(device=device, dtype=torch.long)
    val_noise_all = torch.from_numpy(val_noise_np).to(device=device, dtype=torch.float32)

    best_val_loss = float("inf")
    best_epoch = -1
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    history: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for (batch_x,) in train_loader:
            x0 = batch_x.to(device=device, dtype=torch.float32)
            batch_size = x0.shape[0]
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)

            optimizer.zero_grad()
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_size
            total_train_count += batch_size

        train_loss = total_train_loss / total_train_count
        val_loss = compute_val_loss(
            model=model,
            diffusion=diffusion,
            val_x=val_x,
            val_t_all=val_t_all,
            val_noise_all=val_noise_all,
            batch_size=BATCH_SIZE,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_loss": f"{val_loss:.8f}",
            }
        )

        final_train_loss = train_loss
        final_val_loss = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_torch(OUTPUT_DIR / "val_selected_model.pt", model.state_dict())

        print(f"Epoch {epoch:03d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    save_torch(OUTPUT_DIR / "last_model.pt", model.state_dict())
    save_loss_curve(OUTPUT_DIR / "loss_curve.csv", history)

    selected_model = TemporalDenoiser1D(hidden_dim=HIDDEN_DIM).to(device)
    selected_model.load_state_dict(torch.load(OUTPUT_DIR / "val_selected_model.pt", map_location=device))
    selected_model.eval()

    with torch.no_grad():
        x_t = torch.randn((N_GEN, 2, 19), device=device, dtype=torch.float32)
        for t_idx in reversed(range(TIMESTEPS)):
            t = torch.full((N_GEN,), t_idx, device=device, dtype=torch.long)
            eps_pred = selected_model(x_t, t)

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

        generated_rel = x_t.permute(0, 2, 1).cpu().numpy().astype(np.float32)

    starts = val_abs[:N_GEN, 0, :].astype(np.float32)
    generated_abs = np.zeros((N_GEN, 20, 2), dtype=np.float32)
    generated_abs[:, 0, :] = starts
    generated_abs[:, 1:, :] = starts[:, None, :] + np.cumsum(generated_rel, axis=1)

    if not np.isfinite(generated_rel).all():
        raise RuntimeError("generated_rel contains NaN or Inf")
    if not np.isfinite(generated_abs).all():
        raise RuntimeError("generated_abs contains NaN or Inf")

    real_step = np.linalg.norm(val_rel, axis=-1)
    gen_step = np.linalg.norm(generated_rel, axis=-1)
    real_acc = np.linalg.norm(val_rel[:, 1:, :] - val_rel[:, :-1, :], axis=-1)
    gen_acc = np.linalg.norm(generated_rel[:, 1:, :] - generated_rel[:, :-1, :], axis=-1)

    real_step_stats = summarize_flat(real_step, "real_step")
    gen_step_stats = summarize_flat(gen_step, "gen_step")
    real_acc_stats = summarize_flat(real_acc, "real_acc")
    gen_acc_stats = summarize_flat(gen_acc, "gen_acc")

    out_of_room = (
        (generated_abs[..., 0] < 0.0)
        | (generated_abs[..., 0] > 3.0)
        | (generated_abs[..., 1] < 0.0)
        | (generated_abs[..., 1] > 3.0)
    )
    out_of_room_ratio = float(out_of_room.mean())

    real_step_mean = real_step_stats["real_step_mean"]
    gen_step_mean = gen_step_stats["gen_step_mean"]
    if not (0.5 * real_step_mean <= gen_step_mean <= 1.8 * real_step_mean):
        raise RuntimeError(
            f"gen_step_mean check failed: gen={gen_step_mean:.4f}, real={real_step_mean:.4f}"
        )
    if out_of_room_ratio >= 0.30:
        raise RuntimeError(f"out_of_room_ratio check failed: {out_of_room_ratio:.4f}")

    save_numpy(OUTPUT_DIR / "generated_rel.npy", generated_rel)
    save_numpy(OUTPUT_DIR / "generated_abs.npy", generated_abs)

    prior_check = {
        "train_rel_shape": [800, 19, 2],
        "val_rel_shape": [200, 19, 2],
        "model_input_shape": [BATCH_SIZE, 2, 19],
        "selected_epoch": int(best_epoch),
        "selected_val_loss": float(best_val_loss),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        **real_step_stats,
        **gen_step_stats,
        **real_acc_stats,
        **gen_acc_stats,
        "out_of_room_ratio": out_of_room_ratio,
        "n_generated": N_GEN,
        "timesteps": TIMESTEPS,
        "epochs": EPOCHS,
        "random_seed": RANDOM_SEED,
    }
    save_json(OUTPUT_DIR / "prior_check.json", prior_check)
    plot_generated_examples(generated_abs, val_abs[:N_GEN], OUTPUT_DIR / "generated_examples.png")

    print("=== Step 3 验证输出 ===")
    print(f"train_rel shape: {train_rel.shape}")
    print(f"val_rel shape:   {val_rel.shape}")
    print(f"model input shape example: {example_batch_shape}")
    print(f"selected_epoch: {best_epoch}")
    print(f"selected_val_loss: {best_val_loss:.6f}")
    print(f"final_train_loss: {final_train_loss:.6f}")
    print(f"final_val_loss:   {final_val_loss:.6f}")
    print()
    print("=== Prior step-size check ===")
    print(
        f"real step mean/std: {real_step_stats['real_step_mean']:.4f} / "
        f"{real_step_stats['real_step_std']:.4f}"
    )
    print(
        f"gen  step mean/std: {gen_step_stats['gen_step_mean']:.4f} / "
        f"{gen_step_stats['gen_step_std']:.4f}"
    )
    print(
        "real step p50/p75/p90/p95/max: "
        f"{real_step_stats['real_step_p50']:.4f} / {real_step_stats['real_step_p75']:.4f} / "
        f"{real_step_stats['real_step_p90']:.4f} / {real_step_stats['real_step_p95']:.4f} / "
        f"{real_step_stats['real_step_max']:.4f}"
    )
    print(
        "gen  step p50/p75/p90/p95/max: "
        f"{gen_step_stats['gen_step_p50']:.4f} / {gen_step_stats['gen_step_p75']:.4f} / "
        f"{gen_step_stats['gen_step_p90']:.4f} / {gen_step_stats['gen_step_p95']:.4f} / "
        f"{gen_step_stats['gen_step_max']:.4f}"
    )
    print()
    print("=== Prior acceleration check ===")
    print(
        f"real acc mean/std: {real_acc_stats['real_acc_mean']:.4f} / "
        f"{real_acc_stats['real_acc_std']:.4f}"
    )
    print(
        f"gen  acc mean/std: {gen_acc_stats['gen_acc_mean']:.4f} / "
        f"{gen_acc_stats['gen_acc_std']:.4f}"
    )
    print(
        "real acc p50/p75/p90/p95/max: "
        f"{real_acc_stats['real_acc_p50']:.4f} / {real_acc_stats['real_acc_p75']:.4f} / "
        f"{real_acc_stats['real_acc_p90']:.4f} / {real_acc_stats['real_acc_p95']:.4f} / "
        f"{real_acc_stats['real_acc_max']:.4f}"
    )
    print(
        "gen  acc p50/p75/p90/p95/max: "
        f"{gen_acc_stats['gen_acc_p50']:.4f} / {gen_acc_stats['gen_acc_p75']:.4f} / "
        f"{gen_acc_stats['gen_acc_p90']:.4f} / {gen_acc_stats['gen_acc_p95']:.4f} / "
        f"{gen_acc_stats['gen_acc_max']:.4f}"
    )
    print()
    print(f"out_of_room_ratio: {out_of_room_ratio:.4f}")
    print("outputs saved to: outputs/stage3_indoor/ddpm_prior/")
    print("=== 等待人工确认 ===")


if __name__ == "__main__":
    main()
