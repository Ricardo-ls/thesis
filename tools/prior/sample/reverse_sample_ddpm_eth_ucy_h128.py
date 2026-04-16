from pathlib import Path
import argparse
import json
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
    get_paths_by_name,
    resolve_variant_or_objective,
    to_abs_path,
)
from utils.prior.run_metadata import resolve_current_run_metadata


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> str:
    device_name = device_name.lower()
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda device, but CUDA is not available.")
        return "cuda"
    if device_name == "cpu":
        return "cpu"
    raise ValueError("Unsupported device. Use auto, cpu, or cuda.")


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
def one_step_denoise_check(model, real_rel, timesteps, device, idx, t_vis):
    _, _, alpha_bars = make_ddpm_schedule(timesteps, device=device)

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


def deterministic_indices(size: int, num_show: int, seed: int):
    if size <= 0:
        return []
    num_show = min(num_show, size)
    rng = np.random.default_rng(seed)
    return rng.choice(size, size=num_show, replace=False).tolist()


def resolve_endpoint_quantile_index(real_rel: np.ndarray, quantile: float):
    if real_rel.ndim != 3:
        raise ValueError(f"real_rel ndim should be 3, got {real_rel.shape}")
    endpoint_vec = real_rel.sum(axis=1)
    endpoint_disp = np.linalg.norm(endpoint_vec, axis=1)
    if endpoint_disp.size == 0:
        raise ValueError("Cannot resolve denoise index from an empty dataset.")
    q = float(np.clip(quantile, 0.0, 1.0))
    sorted_idx = np.argsort(endpoint_disp, kind="mergesort")
    pos = int(round(q * (len(sorted_idx) - 1)))
    pos = max(0, min(pos, len(sorted_idx) - 1))
    resolved = int(sorted_idx[pos])
    return resolved, endpoint_disp.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="motion_balanced")
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--vis_seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=19)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--num_generate", type=int, default=512)
    parser.add_argument("--num_show", type=int, default=16)
    parser.add_argument("--denoise_selection", type=str, default="endpoint_quantile")
    parser.add_argument("--denoise_quantile", type=float, default=0.5)
    parser.add_argument("--reference_tag", type=str, default="reference_seed42")
    parser.add_argument("--save_manifest", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--t_vis", type=int, default=80)
    args = parser.parse_args()

    if args.sample_seed is not None:
        set_seed(args.sample_seed)
    resolved_variant = resolve_variant_or_objective(args.variant)
    cfg = get_paths_by_name(args.variant)
    current_run = resolve_current_run_metadata(
        variant=resolved_variant,
        train_seed=args.train_seed,
        train_epochs=args.train_epochs,
        train_root=PROJECT_ROOT / "outputs" / "prior" / "train",
    )
    run_tag = current_run["run_tag"]

    device = resolve_device(args.device)
    data_path = to_abs_path(cfg["rel_path"])
    ckpt_path = Path(current_run["ckpt_path"])
    out_dir = PROJECT_ROOT / "outputs" / "prior" / "sample" / cfg["sample_tag"] / run_tag / args.reference_tag
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
    print(f"train_seed       = {args.train_seed}")
    print(f"train_epochs     = {args.train_epochs}")
    print(f"run_tag          = {run_tag}")
    print(f"seq_len          = {args.seq_len}")
    print(f"channels         = {args.channels}")
    print(f"num_generate     = {args.num_generate}")
    print(f"num_show         = {args.num_show}")
    print(f"sample_seed      = {args.sample_seed}")
    print(f"vis_seed         = {args.vis_seed}")
    print(f"denoise_selection= {args.denoise_selection}")
    print(f"denoise_quantile = {args.denoise_quantile}")
    print(f"reference_tag    = {args.reference_tag}")
    print(f"current_run_best_epoch     = {current_run['current_run_best_epoch']}")
    print(f"current_run_best_val_loss  = {current_run['current_run_best_val_loss']}")
    print("=" * 60)

    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    real_rel = np.load(data_path).astype(np.float32)
    print(f"real_rel shape = {real_rel.shape}")
    denoise_index_resolved = None
    denoise_endpoint_displacements = None
    if args.denoise_selection == "endpoint_quantile":
        denoise_index_resolved, denoise_endpoint_displacements = resolve_endpoint_quantile_index(
            real_rel, args.denoise_quantile
        )
        print(f"resolved denoise index = {denoise_index_resolved}")
    else:
        raise ValueError(
            f"Unsupported denoise_selection: {args.denoise_selection}. "
            "Supported: endpoint_quantile"
        )

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

    real_plot_indices = deterministic_indices(len(real_abs), args.num_show, args.vis_seed)
    generated_plot_indices = deterministic_indices(len(generated_abs), args.num_show, args.vis_seed)
    plot_real_vs_generated(
        real_abs[real_plot_indices],
        generated_abs[generated_plot_indices],
        out_dir / "real_vs_generated.png",
        num_show=args.num_show,
    )

    x0, xt, x0_pred = one_step_denoise_check(
        model,
        real_rel,
        args.timesteps,
        device,
        denoise_index_resolved if denoise_index_resolved is not None else 0,
        args.t_vis,
    )
    plot_denoise_check(x0, xt, x0_pred, out_dir / "denoise_check.png")

    manifest = {
        "input_name": args.variant,
        "resolved_variant": resolved_variant,
        "variant": resolved_variant,
        "data_path": str(data_path),
        "rel_path": str(data_path),
        "ckpt_path": str(ckpt_path),
        "output_dir": str(out_dir),
        "sample_dir": str(out_dir),
        "device": device,
        "sample_seed": args.sample_seed,
        "vis_seed": args.vis_seed,
        "num_generate": args.num_generate,
        "num_show": args.num_show,
        "denoise_selection": args.denoise_selection,
        "denoise_quantile": args.denoise_quantile,
        "denoise_index_resolved": denoise_index_resolved,
        "timesteps": args.timesteps,
        "hidden_dim": args.hidden_dim,
        "train_seed": args.train_seed,
        "train_epochs": args.train_epochs,
        "run_tag": run_tag,
        "current_run_best_epoch": current_run["current_run_best_epoch"],
        "current_run_best_val_loss": current_run["current_run_best_val_loss"],
        "current_run_run_note_path": current_run["run_note_path"],
        "current_run_loss_history_path": current_run["loss_history_path"],
        "real_plot_indices": real_plot_indices,
        "generated_plot_indices": generated_plot_indices,
        "generated_rel_path": str(out_dir / "generated_rel_samples.npy"),
        "generated_abs_path": str(out_dir / "generated_abs_samples.npy"),
        "reference_tag": args.reference_tag,
    }

    if args.save_manifest:
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("saved:")
    print(f"  - {out_dir / 'generated_rel_samples.npy'}")
    print(f"  - {out_dir / 'generated_abs_samples.npy'}")
    print(f"  - {out_dir / 'real_vs_generated.png'}")
    print(f"  - {out_dir / 'denoise_check.png'}")
    if args.save_manifest:
        print(f"  - {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
