from pathlib import Path
import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.temporal_denoiser import TemporalDenoiser1D
from utils.prior.ablation_paths import (
    get_eval_ratios_by_name,
    get_paths_by_name,
    get_train_record_by_name,
    resolve_variant_or_objective,
    to_abs_path,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_ddpm_schedule(timesteps: int, device: str):
    betas = torch.linspace(1e-4, 2e-2, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


@torch.no_grad()
def sample_ddpm(model, timesteps, num_generate, channels, seq_len, device):
    betas, alphas, alpha_bars = make_ddpm_schedule(timesteps, device=device)
    x = torch.randn(num_generate, channels, seq_len, device=device)

    for t in reversed(range(timesteps)):
        t_batch = torch.full((num_generate,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x - coef2 * pred_noise)

        if t > 0:
            x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
        else:
            x = mean

    return x


@torch.no_grad()
def one_step_denoise_check(model, real_rel, timesteps, device, t_vis):
    _, _, alpha_bars = make_ddpm_schedule(timesteps, device=device)

    idx = np.random.randint(0, real_rel.shape[0])
    x0 = torch.from_numpy(real_rel[idx]).float().unsqueeze(0).permute(0, 2, 1).to(device)

    t = torch.tensor([t_vis], device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    pred_noise = model(xt, t)
    x0_pred = (xt - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

    return (
        x0.squeeze(0).permute(1, 0).cpu().numpy(),
        xt.squeeze(0).permute(1, 0).cpu().numpy(),
        x0_pred.squeeze(0).permute(1, 0).cpu().numpy(),
    )


def rel_to_abs(rel):
    rel = np.asarray(rel)
    if rel.ndim != 3:
        raise ValueError(f"rel ndim should be 3, got {rel.shape}")

    if rel.shape[1] == 2 and rel.shape[2] == 19:
        rel = np.transpose(rel, (0, 2, 1))
    elif rel.shape[1] == 19 and rel.shape[2] == 2:
        pass
    else:
        raise ValueError(f"unrecognized rel shape: {rel.shape}")

    zero = np.zeros((rel.shape[0], 1, 2), dtype=rel.dtype)
    return np.concatenate([zero, np.cumsum(rel, axis=1)], axis=1)


def plot_real_vs_generated(real_abs, gen_abs, save_path, num_show=16):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    for i in range(min(num_show, len(real_abs))):
        plt.plot(real_abs[i, :, 0], real_abs[i, :, 1], alpha=0.8)
    plt.title("Real trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    for i in range(min(num_show, len(gen_abs))):
        plt.plot(gen_abs[i, :, 0], gen_abs[i, :, 1], alpha=0.8)
    plt.title("Generated trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_denoise_check(x0, xt, x0_pred, save_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(x0[:, 0], x0[:, 1], marker="o")
    plt.title("Real x0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    plt.subplot(1, 3, 2)
    plt.plot(xt[:, 0], xt[:, 1], marker="o")
    plt.title("Noisy xt")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    plt.subplot(1, 3, 3)
    plt.plot(x0_pred[:, 0], x0_pred[:, 1], marker="o")
    plt.title("Predicted x0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="motion_balanced")
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=19)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--num_generate", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_vis", type=int, default=80)
    args = parser.parse_args()

    set_seed(args.seed)
    resolved_variant = resolve_variant_or_objective(args.variant)
    cfg = get_paths_by_name(args.variant)
    train_record = get_train_record_by_name(args.variant)
    eval_ratios = get_eval_ratios_by_name(args.variant)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = to_abs_path(cfg["rel_path"])
    ckpt_path = to_abs_path(cfg["ckpt_path"])
    out_dir = to_abs_path(cfg["sample_dir"]) / f"reverse_sampling_check_{args.num_generate}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"input_name       = {args.variant}")
    print(f"resolved_variant = {resolved_variant}")
    print(f"device           = {device}")
    print(f"data_path        = {data_path}")
    print(f"ckpt_path        = {ckpt_path}")
    print(f"output_dir       = {out_dir}")
    print(f"timesteps        = {args.timesteps}")
    print(f"hidden_dim       = {args.hidden_dim}")
    print(f"seq_len          = {args.seq_len}")
    print(f"channels         = {args.channels}")
    print(f"num_generate     = {args.num_generate}")
    print(f"train_record     = {train_record}")
    print(f"eval_ratios      = {eval_ratios}")
    print("=" * 60)

    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    real_rel = np.load(data_path).astype(np.float32)
    print(f"real_rel shape = {real_rel.shape}")

    model = TemporalDenoiser1D(
        max_timesteps=args.timesteps,
        in_channels=args.channels,
        hidden_dim=args.hidden_dim,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing keys    = {missing}")
    print(f"unexpected keys = {unexpected}")
    model.eval()

    generated_rel = sample_ddpm(
        model=model,
        timesteps=args.timesteps,
        num_generate=args.num_generate,
        channels=args.channels,
        seq_len=args.seq_len,
        device=device,
    )
    generated_rel_np = generated_rel.cpu().numpy().astype(np.float32)
    np.save(out_dir / "generated_rel_samples.npy", generated_rel_np)

    generated_abs = rel_to_abs(generated_rel_np)
    real_abs = rel_to_abs(real_rel)
    np.save(out_dir / "generated_abs_samples.npy", generated_abs.astype(np.float32))

    plot_real_vs_generated(real_abs, generated_abs, out_dir / "real_vs_generated.png")

    x0, xt, x0_pred = one_step_denoise_check(model, real_rel, args.timesteps, device, args.t_vis)
    plot_denoise_check(x0, xt, x0_pred, out_dir / "denoise_check.png")

    print("saved:")
    print(f"  - {out_dir / 'generated_rel_samples.npy'}")
    print(f"  - {out_dir / 'generated_abs_samples.npy'}")
    print(f"  - {out_dir / 'real_vs_generated.png'}")
    print(f"  - {out_dir / 'denoise_check.png'}")


if __name__ == "__main__":
    main()
