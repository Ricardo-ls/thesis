#你的训练输入是：
#x0: 干净 relative displacement 轨迹，shape [B, 2, 19]
#训练时你会：
#随机采样时间步 t
#采样高斯噪声 noise
#构造 xt
#用模型预测 pred_noise
#用 MSE(pred_noise, noise) 训练
#这就是最小 DDPM 的标准噪声预测训练。
#所以这一步训练的本质是：
#让模型学会“给定带噪轨迹 xt 和时间步 t，恢复噪声”




from pathlib import Path
import sys
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
            pred_noise = model(xt, t)                            # [B, 2, 19]
            loss = F.mse_loss(pred_noise, noise)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = x0.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def main():
    # ===== 基础配置 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = PROJECT_ROOT / "datasets" / "processed" / "data_eth_20_rel_q20.npy"

    batch_size = 32
    epochs = 50
    lr = 1e-3
    timesteps = 100
    hidden_dim = 64
    train_ratio = 0.8
    random_seed = 42

    output_dir = PROJECT_ROOT / "outputs" / "prior" / "train" / "ddpm_minimal_q20"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device      = {device}")
    print(f"data_path    = {data_path}")
    print(f"output_dir   = {output_dir}")
    print(f"batch_size   = {batch_size}")
    print(f"epochs       = {epochs}")
    print(f"lr           = {lr}")
    print(f"timesteps    = {timesteps}")
    print(f"hidden_dim   = {hidden_dim}")

    # ===== 数据集 =====
    dataset = TrajectoryDataset(str(data_path))
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

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr": lr,
                        "timesteps": timesteps,
                        "hidden_dim": hidden_dim,
                        "train_ratio": train_ratio,
                        "data_path": str(data_path),
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
    print(f"best_val_loss = {best_val_loss:.6f}")
    print(f"best model    = {output_dir / 'best_model.pt'}")
    print(f"last model    = {output_dir / 'last_model.pt'}")
    print(f"loss history  = {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()