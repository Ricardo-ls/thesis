from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tools.stage3_indoor.train_ddpm_prior import DDPMProcess, HIDDEN_DIM, TemporalDenoiser1D, TIMESTEPS


DATA_PATH = PROJECT_ROOT / "data" / "stage3_indoor" / "clean_trajs.npy"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_prior" / "val_selected_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage3_indoor" / "ddpm_prior_diagnostics"

OVERFIT_CSV = OUTPUT_DIR / "overfit_loss_curve.csv"
ONE_STEP_CSV = OUTPUT_DIR / "one_step_denoise.csv"
SAMPLING_JSON = OUTPUT_DIR / "sampling_distribution.json"
GEN_ABS_PATH = OUTPUT_DIR / "generated_abs_check.npy"
GEN_REL_PATH = OUTPUT_DIR / "generated_rel_check.npy"
SAMPLING_FIG = OUTPUT_DIR / "sampling_examples.png"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def save_numpy(path: Path, array: np.ndarray) -> None:
    np.save(path, array.astype(np.float32))
    print(f"Saved: {path}")


def rel_to_abs(start_points: np.ndarray, rel: np.ndarray) -> np.ndarray:
    abs_trajs = np.zeros((rel.shape[0], rel.shape[1] + 1, 2), dtype=np.float32)
    abs_trajs[:, 0, :] = start_points.astype(np.float32)
    abs_trajs[:, 1:, :] = start_points[:, None, :] + np.cumsum(rel, axis=1)
    return abs_trajs


def flatten_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    flat = values.reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_p50": float(np.percentile(flat, 50)),
        f"{prefix}_p75": float(np.percentile(flat, 75)),
        f"{prefix}_p90": float(np.percentile(flat, 90)),
        f"{prefix}_p95": float(np.percentile(flat, 95)),
        f"{prefix}_p99": float(np.percentile(flat, 99)),
        f"{prefix}_max": float(np.max(flat)),
    }


def compute_l2_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm(a - b, dim=1).mean().item())


def sample_unconditional(
    model: TemporalDenoiser1D,
    diffusion: DDPMProcess,
    num_samples: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    set_seed(seed)
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


def plot_sampling_examples(real_abs: np.ndarray, gen_abs: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    indices = np.arange(16)
    gen_steps = np.linalg.norm(gen_abs[:, 1:, :] - gen_abs[:, :-1, :], axis=-1)
    out_room_flags = (
        (gen_abs[..., 0] < 0.0)
        | (gen_abs[..., 0] > 3.0)
        | (gen_abs[..., 1] < 0.0)
        | (gen_abs[..., 1] > 3.0)
    ).any(axis=1)

    for ax, idx in zip(axes.flatten(), indices):
        ax.plot(real_abs[idx, :, 0], real_abs[idx, :, 1], color="lightgray", linewidth=1.6)
        ax.plot(gen_abs[idx, :, 0], gen_abs[idx, :, 1], color="red", linewidth=1.8)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"real/gen idx={idx}\nmax_step={gen_steps[idx].max():.3f}  out={bool(out_room_flags[idx])}",
            fontsize=9,
        )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {DATA_PATH}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing required checkpoint: {CHECKPOINT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_all = np.load(DATA_PATH).astype(np.float32)
    clean = clean_all[:1000]
    if clean.shape != (1000, 20, 2):
        raise ValueError(f"Expected clean shape (1000, 20, 2), got {clean.shape}")

    train_abs = clean[:800]
    val_abs = clean[800:1000]
    train_rel = (train_abs[:, 1:, :] - train_abs[:, :-1, :]).astype(np.float32)
    val_rel = (val_abs[:, 1:, :] - val_abs[:, :-1, :]).astype(np.float32)

    if train_rel.shape != (800, 19, 2):
        raise ValueError(f"Expected train_rel shape (800, 19, 2), got {train_rel.shape}")
    if val_rel.shape != (200, 19, 2):
        raise ValueError(f"Expected val_rel shape (200, 19, 2), got {val_rel.shape}")

    train_x = train_rel.transpose(0, 2, 1)
    x_example_shape = (min(128, train_x.shape[0]), 2, 19)

    rng = np.random.default_rng(42)
    check_indices = rng.choice(clean.shape[0], size=10, replace=False)
    sampled_abs = clean[check_indices]
    sampled_rel = sampled_abs[:, 1:, :] - sampled_abs[:, :-1, :]
    reconstructed_abs = rel_to_abs(sampled_abs[:, 0, :], sampled_rel)
    rel_reconstruction_max_error = float(np.max(np.abs(reconstructed_abs - sampled_abs)))
    if rel_reconstruction_max_error >= 1e-5:
        print(f"relative displacement reconstruction failed: {rel_reconstruction_max_error:.8f}")
        raise RuntimeError("relative displacement reconstruction error too large")

    device = torch.device("cpu")
    diffusion = DDPMProcess(timesteps=TIMESTEPS, device=device)

    overfit_rel = (clean[:32, 1:, :] - clean[:32, :-1, :]).astype(np.float32)
    overfit_x = torch.from_numpy(overfit_rel.transpose(0, 2, 1)).to(device=device, dtype=torch.float32)
    set_seed(123)
    overfit_model = TemporalDenoiser1D(hidden_dim=HIDDEN_DIM).to(device)
    overfit_optimizer = torch.optim.Adam(overfit_model.parameters(), lr=1e-4)
    overfit_rows: list[dict] = []
    initial_loss = float("nan")
    final_loss = float("nan")

    for iteration in range(1, 1001):
        t = torch.randint(0, TIMESTEPS, (32,), device=device, dtype=torch.long)
        noise = torch.randn_like(overfit_x)
        x_t = diffusion.q_sample(overfit_x, t, noise)

        overfit_optimizer.zero_grad()
        eps_pred = overfit_model(x_t, t)
        loss = F.mse_loss(eps_pred, noise)
        loss.backward()
        overfit_optimizer.step()

        if iteration % 100 == 0:
            loss_value = float(loss.item())
            overfit_rows.append({"iteration": iteration, "loss": f"{loss_value:.8f}"})
            if len(overfit_rows) == 1:
                initial_loss = loss_value
            final_loss = loss_value

    save_csv(OVERFIT_CSV, overfit_rows, ["iteration", "loss"])
    loss_drop_pct = float((initial_loss - final_loss) / initial_loss * 100.0)

    model = TemporalDenoiser1D(hidden_dim=HIDDEN_DIM).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    val_x = torch.from_numpy(val_rel.transpose(0, 2, 1)).to(device=device, dtype=torch.float32)
    one_step_rows: list[dict] = []
    set_seed(2024)
    with torch.no_grad():
        for t_value in [1, 3, 5, 10, 20, 50]:
            t = torch.full((val_x.shape[0],), t_value, device=device, dtype=torch.long)
            noise = torch.randn_like(val_x)
            alpha_bar_t = diffusion.alpha_bars[t_value]
            x_t = diffusion.q_sample(val_x, t, noise)
            noise_pred = model(x_t, t)
            x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            noisy_error = compute_l2_mean(x_t, val_x)
            denoised_error = compute_l2_mean(x0_hat, val_x)
            improvement = noisy_error - denoised_error
            improvement_pct = improvement / noisy_error * 100.0

            one_step_rows.append(
                {
                    "t": t_value,
                    "noisy_error": f"{noisy_error:.8f}",
                    "denoised_error": f"{denoised_error:.8f}",
                    "improvement": f"{improvement:.8f}",
                    "improvement_pct": f"{improvement_pct:.4f}",
                }
            )

    save_csv(
        ONE_STEP_CSV,
        one_step_rows,
        ["t", "noisy_error", "denoised_error", "improvement", "improvement_pct"],
    )

    gen_rel = sample_unconditional(
        model=model,
        diffusion=diffusion,
        num_samples=512,
        seed=456,
        device=device,
    )
    gen_abs = rel_to_abs(clean[:512, 0, :], gen_rel)
    real_abs = clean[:512]
    real_rel = (real_abs[:, 1:, :] - real_abs[:, :-1, :]).astype(np.float32)

    real_step = np.linalg.norm(real_rel, axis=-1)
    gen_step = np.linalg.norm(gen_rel, axis=-1)
    real_acc = np.linalg.norm(real_rel[:, 1:, :] - real_rel[:, :-1, :], axis=-1)
    gen_acc = np.linalg.norm(gen_rel[:, 1:, :] - gen_rel[:, :-1, :], axis=-1)

    real_step_stats = flatten_stats(real_step, "real_step")
    gen_step_stats = flatten_stats(gen_step, "gen_step")
    real_acc_stats = flatten_stats(real_acc, "real_acc")
    gen_acc_stats = flatten_stats(gen_acc, "gen_acc")

    out_of_room = (
        (gen_abs[..., 0] < 0.0)
        | (gen_abs[..., 0] > 3.0)
        | (gen_abs[..., 1] < 0.0)
        | (gen_abs[..., 1] > 3.0)
    )
    out_of_room_ratio = float(out_of_room.mean())

    large_step_mask = gen_step > 0.6
    large_step_count_0p6 = int(np.sum(large_step_mask))
    large_step_ratio_0p6 = float(np.mean(large_step_mask))

    save_numpy(GEN_ABS_PATH, gen_abs)
    save_numpy(GEN_REL_PATH, gen_rel)
    save_json(
        SAMPLING_JSON,
        {
            **real_step_stats,
            **gen_step_stats,
            **real_acc_stats,
            **gen_acc_stats,
            "out_of_room_ratio": out_of_room_ratio,
            "large_step_count_0p6": large_step_count_0p6,
            "large_step_ratio_0p6": large_step_ratio_0p6,
            "num_samples": 512,
            "sampling_seed": 456,
            "checkpoint_path": "outputs/stage3_indoor/ddpm_prior/val_selected_model.pt",
        },
    )
    plot_sampling_examples(real_abs, gen_abs, SAMPLING_FIG)

    print("=== Step 3.6 DDPM Prior 诊断输出 ===")

    print()
    print("[1] 数据表示检查")
    print(f"clean shape: {clean.shape}")
    print(f"train_rel shape: {train_rel.shape}")
    print(f"val_rel shape:   {val_rel.shape}")
    print(f"model input shape example: {x_example_shape}")
    print(f"rel reconstruction max error: {rel_reconstruction_max_error:.8f}")

    print()
    print("[2] 小样本 overfit 检查")
    print(f"initial overfit loss: {initial_loss:.6f}")
    print(f"final overfit loss:   {final_loss:.6f}")
    print(f"loss drop:            {loss_drop_pct:.2f}%")

    print()
    print("[3] One-step denoise 检查")
    print(f"{'t':>5} {'noisy_error':>14} {'denoised_error':>16} {'improvement':>14} {'improve_%':>12}")
    for row in one_step_rows:
        print(
            f"{int(row['t']):5d} {float(row['noisy_error']):14.6f} {float(row['denoised_error']):16.6f} "
            f"{float(row['improvement']):14.6f} {float(row['improvement_pct']):12.2f}"
        )

    print()
    print("[4] Unconditional sampling 分布检查")
    print(f"real step mean/std: {real_step_stats['real_step_mean']:.4f} / {real_step_stats['real_step_std']:.4f}")
    print(f"gen  step mean/std: {gen_step_stats['gen_step_mean']:.4f} / {gen_step_stats['gen_step_std']:.4f}")
    print(
        "real step p50/p75/p90/p95/p99/max: "
        f"{real_step_stats['real_step_p50']:.4f} / {real_step_stats['real_step_p75']:.4f} / "
        f"{real_step_stats['real_step_p90']:.4f} / {real_step_stats['real_step_p95']:.4f} / "
        f"{real_step_stats['real_step_p99']:.4f} / {real_step_stats['real_step_max']:.4f}"
    )
    print(
        "gen  step p50/p75/p90/p95/p99/max: "
        f"{gen_step_stats['gen_step_p50']:.4f} / {gen_step_stats['gen_step_p75']:.4f} / "
        f"{gen_step_stats['gen_step_p90']:.4f} / {gen_step_stats['gen_step_p95']:.4f} / "
        f"{gen_step_stats['gen_step_p99']:.4f} / {gen_step_stats['gen_step_max']:.4f}"
    )

    print()
    print(f"real acc mean/std: {real_acc_stats['real_acc_mean']:.4f} / {real_acc_stats['real_acc_std']:.4f}")
    print(f"gen  acc mean/std: {gen_acc_stats['gen_acc_mean']:.4f} / {gen_acc_stats['gen_acc_std']:.4f}")
    print(
        "real acc p50/p75/p90/p95/p99/max: "
        f"{real_acc_stats['real_acc_p50']:.4f} / {real_acc_stats['real_acc_p75']:.4f} / "
        f"{real_acc_stats['real_acc_p90']:.4f} / {real_acc_stats['real_acc_p95']:.4f} / "
        f"{real_acc_stats['real_acc_p99']:.4f} / {real_acc_stats['real_acc_max']:.4f}"
    )
    print(
        "gen  acc p50/p75/p90/p95/p99/max: "
        f"{gen_acc_stats['gen_acc_p50']:.4f} / {gen_acc_stats['gen_acc_p75']:.4f} / "
        f"{gen_acc_stats['gen_acc_p90']:.4f} / {gen_acc_stats['gen_acc_p95']:.4f} / "
        f"{gen_acc_stats['gen_acc_p99']:.4f} / {gen_acc_stats['gen_acc_max']:.4f}"
    )

    print()
    print(f"out_of_room_ratio: {out_of_room_ratio:.4f}")
    print(f"large_step_count_0p6: {large_step_count_0p6}")
    print(f"large_step_ratio_0p6: {large_step_ratio_0p6:.6f}")

    print()
    print("输出已保存至 outputs/stage3_indoor/ddpm_prior_diagnostics/")
    print("=== 等待人工确认 ===")


if __name__ == "__main__":
    main()
