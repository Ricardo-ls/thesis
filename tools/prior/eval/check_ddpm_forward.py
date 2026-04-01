#验证整条链的 shape 和 loss 都是通的。

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.traj_dataset import TrajectoryDataset
from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    data_path = PROJECT_ROOT / "datasets" / "processed" / "data_eth_20_rel_q20.npy"
    dataset = TrajectoryDataset(str(data_path))

    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    batch = next(iter(train_loader))         # [B, 19, 2]
    batch = batch.permute(0, 2, 1).to(device)  # [B, 2, 19]

    print(f"x0 shape = {batch.shape}")

    diffusion = DDPMForwardProcess(timesteps=100, device=device)
    model = TemporalDenoiser1D(max_timesteps=100, in_channels=2, hidden_dim=64).to(device)

    t = diffusion.sample_timesteps(batch_size=batch.shape[0])   # [B]
    xt, noise = diffusion.q_sample(batch, t)                    # [B, 2, 19], [B, 2, 19]

    print(f"t shape     = {t.shape}")
    print(f"xt shape    = {xt.shape}")
    print(f"noise shape = {noise.shape}")

    pred_noise = model(xt, t)                                   # [B, 2, 19]
    print(f"pred shape  = {pred_noise.shape}")

    loss = torch.mean((pred_noise - noise) ** 2)
    print(f"loss = {loss.item():.6f}")


if __name__ == "__main__":
    main()