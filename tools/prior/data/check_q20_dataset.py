import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.traj_dataset import TrajectoryDataset


def main():
    dataset = TrajectoryDataset("datasets/processed/data_eth_20_rel_q20.npy")

    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    print(f"total_len = {total_len}")
    print(f"train_len = {train_len}")
    print(f"val_len   = {val_len}")

    batch = next(iter(train_loader))
    print(f"batch shape before permute = {batch.shape}")   # [B, 19, 2]

    batch = batch.permute(0, 2, 1)
    print(f"batch shape after permute  = {batch.shape}")   # [B, 2, 19]


if __name__ == "__main__":
    main()