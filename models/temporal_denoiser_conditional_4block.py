import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, max_timesteps: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_timesteps, emb_dim)

    def forward(self, t: torch.Tensor):
        return self.embedding(t)


def _make_residual_block(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class TemporalDenoiserConditional4Block(nn.Module):
    def __init__(self, max_timesteps: int = 100, in_channels: int = 4, hidden_dim: int = 128):
        super().__init__()

        self.time_emb = TimeEmbedding(max_timesteps=max_timesteps, emb_dim=hidden_dim)
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)

        self.block1 = _make_residual_block(hidden_dim)
        self.block2 = _make_residual_block(hidden_dim)
        self.block3 = _make_residual_block(hidden_dim)
        self.block4 = _make_residual_block(hidden_dim)

        self.output_proj = nn.Conv1d(hidden_dim, 2, kernel_size=3, padding=1)

    def forward(self, x_t_residual: torch.Tensor, x_cond: torch.Tensor, t: torch.Tensor):
        model_input = torch.cat([x_t_residual, x_cond], dim=1)

        h = self.input_proj(model_input)

        t_emb = self.time_emb(t)
        t_emb = t_emb.unsqueeze(-1)
        h = h + t_emb

        h = self.block1(h) + h
        h = self.block2(h) + h
        h = self.block3(h) + h
        h = self.block4(h) + h

        out = self.output_proj(h)
        return out


ConditionalTemporalDenoiser1D4Block = TemporalDenoiserConditional4Block
