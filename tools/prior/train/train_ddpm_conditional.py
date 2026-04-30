from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from utils.prior.ablation_paths import get_eth_ucy_variant_paths
from datasets.traj_dataset import TrajectoryDataset
from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import ConditionalTemporalDenoiser1D


SPAN_START = 8
SPAN_END = 12


def build_run_dir(train_tag: str, random_seed: int, epochs: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "prior"
        / "train"
        / train_tag
        / f"seed{random_seed}-{epochs}epoch"
    )


def save_loss_curve(history, output_dir: Path):
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    start_idx = 9 if len(epochs) >= 10 else 0
    plot_epochs = epochs[start_idx:]
    plot_train = train_losses[start_idx:]
    plot_val = val_losses[start_idx:]

    best_idx = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
    best_epoch = epochs[best_idx]
    best_val = val_losses[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(plot_epochs, plot_train, color="#1f4e79", linewidth=2.0, label="train_loss")
    plt.plot(plot_epochs, plot_val, color="#ff8c00", linewidth=2.0, label="val_loss")
    if best_epoch >= plot_epochs[0]:
        plt.scatter(
            [best_epoch],
            [best_val],
            color="#1f4e79",
            s=60,
            label=f"best val @ epoch {best_epoch}",
            zorder=3,
        )
    plt.title("Conditional DDPM Training / Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve_epoch10plus.png", dpi=200)
    plt.savefig(output_dir / "loss_curve_epoch10plus.svg")
    plt.close()


def save_run_note(output_dir: Path, variant: str, args, best_epoch: int, best_val_loss: float):
    note_path = output_dir / f"RUN_NOTE_{variant}_ep{args.epochs}_seed{args.random_seed}.md"
    note_lines = [
        f"# conditional {variant} variant - {args.epochs}-epoch training snapshot",
        "",
        "## Purpose",
        "",
        f"This file records the {args.epochs}-epoch conditional DDPM training snapshot for the `{variant}` variant.",
        "",
        "## Training configuration",
        "",
        f"- variant: `{variant}`",
        f"- epochs: `{args.epochs}`",
        f"- batch_size: `{args.batch_size}`",
        f"- timesteps: `{args.timesteps}`",
        f"- hidden_dim: `{args.hidden_dim}`",
        f"- random_seed: `{args.random_seed}`",
        f"- train_ratio: `{args.train_ratio}`",
        f"- lr: `{args.lr}`",
        f"- span_start: `{SPAN_START}`",
        f"- span_end_exclusive: `{SPAN_END}`",
        "",
        "## Final result",
        "",
        f"- best_epoch: `{best_epoch}`",
        f"- best_val_loss: `{best_val_loss:.6f}`",
        "- best_model: `best_model.pt`",
        "- last_model: `last_model.pt`",
        "- loss_history: `loss_history.csv`",
        "- loss_curve_png: `loss_curve_epoch10plus.png`",
        "- loss_curve_svg: `loss_curve_epoch10plus.svg`",
        "",
    ]
    note_path.write_text("\n".join(note_lines), encoding="utf-8")


def make_conditional_inputs(x0: torch.Tensor):
    batch_size, _, seq_len = x0.shape
    obs_mask = torch.ones(batch_size, 1, seq_len, device=x0.device, dtype=x0.dtype)
    obs_mask[:, :, SPAN_START:SPAN_END] = 0.0
    masked_obs = x0.clone()
    masked_obs[:, :, SPAN_START:SPAN_END] = 0.0
    return obs_mask, masked_obs


def build_masked_forward_sample(diffusion: DDPMForwardProcess, x0: torch.Tensor, t: torch.Tensor):
    batch_size = x0.shape[0]
    noise = torch.zeros_like(x0)
    noise[:, :, SPAN_START:SPAN_END] = torch.randn(
        batch_size,
        x0.shape[1],
        SPAN_END - SPAN_START,
        device=x0.device,
        dtype=x0.dtype,
    )

    xt = x0.clone()
    xt_missing, _ = diffusion.q_sample(
        x0[:, :, SPAN_START:SPAN_END],
        t,
        noise[:, :, SPAN_START:SPAN_END],
    )
    xt[:, :, SPAN_START:SPAN_END] = xt_missing
    return xt, noise


def run_one_epoch(model, diffusion, loader, optimizer, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x0 = batch.permute(0, 2, 1).to(device)
        t = diffusion.sample_timesteps(batch_size=x0.shape[0])
        obs_mask, masked_obs = make_conditional_inputs(x0)
        xt, noise = build_masked_forward_sample(diffusion, x0, t)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred_noise = model(xt, t, obs_mask, masked_obs)
            loss = F.mse_loss(
                pred_noise[:, :, SPAN_START:SPAN_END],
                noise[:, :, SPAN_START:SPAN_END],
            )

            if train:
                loss.backward()
                optimizer.step()

        batch_size = x0.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="none", choices=["none", "q10", "q20", "q30"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    paths = get_eth_ucy_variant_paths(args.variant)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rel_data_path = PROJECT_ROOT / paths["rel_path"]
    train_tag = f"ddpm_eth_ucy_conditional_{args.variant}_h128"
    output_dir = build_run_dir(train_tag, args.random_seed, args.epochs)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"variant      = {args.variant}")
    print(f"device       = {device}")
    print(f"rel_path     = {rel_data_path}")
    print(f"output_dir   = {output_dir}")
    print(f"batch_size   = {args.batch_size}")
    print(f"epochs       = {args.epochs}")
    print(f"lr           = {args.lr}")
    print(f"timesteps    = {args.timesteps}")
    print(f"hidden_dim   = {args.hidden_dim}")
    print(f"train_ratio  = {args.train_ratio}")
    print(f"random_seed  = {args.random_seed}")
    print(f"span_start   = {SPAN_START}")
    print(f"span_end     = {SPAN_END}")

    dataset = TrajectoryDataset(str(rel_data_path))
    total_len = len(dataset)
    train_len = int(total_len * args.train_ratio)
    val_len = total_len - train_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.random_seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f"total_len    = {total_len}")
    print(f"train_len    = {train_len}")
    print(f"val_len      = {val_len}")

    diffusion = DDPMForwardProcess(timesteps=args.timesteps, device=device)
    model = ConditionalTemporalDenoiser1D(
        max_timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(model, diffusion, train_loader, optimizer, device, train=True)
        val_loss = run_one_epoch(model, diffusion, val_loader, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss = {train_loss:.6f} | val_loss = {val_loss:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "variant": args.variant,
                    "train_tag": train_tag,
                    "random_seed": args.random_seed,
                    "span_start": SPAN_START,
                    "span_end_exclusive": SPAN_END,
                    "conditioning": "obs_mask + masked_obs",
                },
                output_dir / "best_model.pt",
            )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "variant": args.variant,
            "train_tag": train_tag,
            "random_seed": args.random_seed,
            "span_start": SPAN_START,
            "span_end_exclusive": SPAN_END,
            "conditioning": "obs_mask + masked_obs",
        },
        output_dir / "last_model.pt",
    )

    with (output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    save_loss_curve(history, output_dir)
    save_run_note(output_dir, args.variant, args, best_epoch, best_val_loss)

    print("=" * 60)
    print("Conditional DDPM training finished")
    print(f"best_epoch    = {best_epoch}")
    print(f"best_val_loss = {best_val_loss:.6f}")
    print(f"best_model    = {output_dir / 'best_model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
