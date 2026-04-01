#建立 beta schedule
#计算 alphas
#计算 alpha_bars
#实现 q_sample

from pathlib import Path
import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPMForwardProcess:
    def __init__(self, timesteps: int = 100, device: str = "cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = linear_beta_schedule(timesteps).to(device)                # [T]
        self.alphas = 1.0 - self.betas                                         # [T]
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)                    # [T]

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)                     # [T]
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)     # [T]

    def sample_timesteps(self, batch_size: int):
        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=self.device
        )  # [B]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        x0: [B, C, L]
        t:  [B]
        noise: [B, C, L]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1)                  # [B,1,1]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise