from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from utils.prior.ablation_paths import get_eth_ucy_variant_paths
from datasets.traj_dataset import TrajectoryDataset
from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D


def run_one_epoch(model, diffusion, loader, optimizer, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        # batch: [B, 19, 2] -> [B, 2, 19]
        x0 = batch.permute(0, 2, 1).to(device)

        t = diffusion.sample_timesteps(batch_size=x0.shape[0])   # [B]
        xt, noise = diffusion.q_sample(x0, t)                    # [B, 2, 19]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = x0.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default="q20",
        choices=["none", "q10", "q20", "q30"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    paths = get_eth_ucy_variant_paths(args.variant)

    # ===== 基础配置 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    abs_data_path = PROJECT_ROOT / paths["abs_path"]
    rel_data_path = PROJECT_ROOT / paths["rel_path"]

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    timesteps = args.timesteps
    hidden_dim = args.hidden_dim
    train_ratio = args.train_ratio
    random_seed = args.random_seed

    output_dir = PROJECT_ROOT / "outputs" / "prior" / "train" / paths["train_tag"] / f"seed{random_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"variant      = {args.variant}")
    print(f"device       = {device}")
    print(f"abs_path     = {abs_data_path}")
    print(f"rel_path     = {rel_data_path}")
    print(f"output_dir   = {output_dir}")
    print(f"batch_size   = {batch_size}")
    print(f"epochs       = {epochs}")
    print(f"lr           = {lr}")
    print(f"timesteps    = {timesteps}")
    print(f"hidden_dim   = {hidden_dim}")
    print(f"train_ratio  = {train_ratio}")
    print(f"random_seed  = {random_seed}")

    # ===== 数据集 =====
    dataset = TrajectoryDataset(str(rel_data_path))
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(random_seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"total_len    = {total_len}")
    print(f"train_len    = {train_len}")
    print(f"val_len      = {val_len}")

    # ===== 模型与扩散过程 =====
    diffusion = DDPMForwardProcess(timesteps=timesteps, device=device)
    model = TemporalDenoiser1D(
        max_timesteps=timesteps,
        in_channels=2,
        hidden_dim=hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ===== 训练记录 =====
    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(
            model=model,
            diffusion=diffusion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss = run_one_epoch(
            model=model,
            diffusion=diffusion,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
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
                    "best_val_loss": best_val_loss,
                    "config": {
                        "variant": args.variant,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr": lr,
                        "timesteps": timesteps,
                        "hidden_dim": hidden_dim,
                        "train_ratio": train_ratio,
                        "random_seed": random_seed,
                        "abs_data_path": str(abs_data_path),
                        "rel_data_path": str(rel_data_path),
                    },
                },
                output_dir / "best_model.pt"
            )

    # 保存最终模型
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "config": {
                "variant": args.variant,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "timesteps": timesteps,
                "hidden_dim": hidden_dim,
                "train_ratio": train_ratio,
                "random_seed": random_seed,
                "abs_data_path": str(abs_data_path),
                "rel_data_path": str(rel_data_path),
            },
        },
        output_dir / "last_model.pt"
    )

    # 保存 loss 历史
    csv_path = output_dir / "loss_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    print("=" * 60)
    print("训练完成")
    print(f"variant       = {args.variant}")
    print(f"best_val_loss = {best_val_loss:.6f}")
    print(f"best_epoch    = {best_epoch}")
    print(f"best model    = {output_dir / 'best_model.pt'}")
    print(f"last model    = {output_dir / 'last_model.pt'}")
    print(f"loss history  = {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
