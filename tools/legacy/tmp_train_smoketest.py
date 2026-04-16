from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.traj_dataset import TrajectoryDataset
from diffusion.ddpm_utils import DDPMForwardProcess
from models.temporal_denoiser import TemporalDenoiser1D

project_root = Path.cwd()
data_path = project_root / "datasets" / "processed" / "data_eth_20_rel_q20.npy"

print("data_path =", data_path)
print("exists    =", data_path.exists())

dataset = TrajectoryDataset(str(data_path))
loader = DataLoader(dataset, batch_size=8, shuffle=False)

batch = next(iter(loader))          # [B, 19, 2]
x0 = batch.permute(0, 2, 1)         # [B, 2, 19]

device = "cuda" if torch.cuda.is_available() else "cpu"
x0 = x0.to(device)

diffusion = DDPMForwardProcess(timesteps=100, device=device)
model = TemporalDenoiser1D(
    max_timesteps=100,
    in_channels=2,
    hidden_dim=64
).to(device)

t = diffusion.sample_timesteps(batch_size=x0.shape[0])
xt, noise = diffusion.q_sample(x0, t)
pred_noise = model(xt, t)
loss = F.mse_loss(pred_noise, noise)

print("batch shape      =", batch.shape)
print("x0 shape         =", x0.shape)
print("t shape          =", t.shape)
print("xt shape         =", xt.shape)
print("noise shape      =", noise.shape)
print("pred_noise shape =", pred_noise.shape)
print("loss             =", float(loss.item())) 