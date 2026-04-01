import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, npy_path: str):
        self.data = np.load(npy_path).astype(np.float32)   # shape: (N, 19, 2)

        if self.data.ndim != 3 or self.data.shape[-1] != 2:
            raise ValueError(f"数据 shape 应为 (N, T, 2)，当前为 {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])   # shape: (19, 2)
        return x