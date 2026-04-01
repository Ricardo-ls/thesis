#这是最小 1D 时序去噪器。
#设计思想很简单：
#输入：xt，shape [B, 2, 19]
#额外输入：t
#用一个很小的时间步 embedding
#用几层 1D Conv 提取时序局部模式
#输出预测噪声，shape 仍然是 [B, 2, 19]

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, max_timesteps: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_timesteps, emb_dim)

    def forward(self, t: torch.Tensor):
        return self.embedding(t)   # [B, emb_dim]


class TemporalDenoiser1D(nn.Module):
    def __init__(self, max_timesteps: int = 100, in_channels: int = 2, hidden_dim: int = 64):
        super().__init__()

        self.time_emb = TimeEmbedding(max_timesteps=max_timesteps, emb_dim=hidden_dim)

        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)

        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.output_proj = nn.Conv1d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: [B, 2, 19]
        t: [B]
        """
        h = self.input_proj(x)   # [B, hidden_dim, 19]

        # 时间步嵌入加到特征图上
        t_emb = self.time_emb(t)                     # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(-1)                 # [B, hidden_dim, 1]
        h = h + t_emb

        h = self.block1(h) + h
        h = self.block2(h) + h

        out = self.output_proj(h)   # [B, 2, 19]
        return out