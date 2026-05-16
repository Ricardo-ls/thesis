from __future__ import annotations

import importlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    half_dim = dim // 2
    scale = math.log(10000.0) / max(half_dim - 1, 1)
    exponents = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale
    )
    args = timesteps.float().unsqueeze(1) * exponents.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _coerce_timesteps(t: torch.Tensor | int | float, batch_size: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    else:
        t = t.to(device=device)

    if t.ndim == 0:
        t = t.view(1).repeat(batch_size)
    else:
        t = t.reshape(-1)
        if t.shape[0] == 1 and batch_size != 1:
            t = t.repeat(batch_size)

    if t.shape[0] != batch_size:
        raise ValueError(f"Expected {batch_size} timesteps, got {tuple(t.shape)}")
    return t.long()


def _unwrap_module_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Unsupported S4 module output type: {type(output)!r}")


def _import_official_s4layer() -> type[nn.Module] | None:
    candidates = [
        ("src.imputers.S4Model", "S4Layer"),
        ("src.imputers.SSSDS4Imputer", "S4Layer"),
        ("SSSD.src.imputers.S4Model", "S4Layer"),
        ("SSSD.src.imputers.SSSDS4Imputer", "S4Layer"),
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            s4_cls = getattr(module, class_name, None)
            if s4_cls is not None:
                return s4_cls
        except Exception:
            continue
    return None


def _build_official_s4_instance(
    s4_cls: type[nn.Module],
    hidden_dim: int,
    state_dim: int,
    dropout: float,
) -> nn.Module | None:
    constructor_candidates = [
        {
            "d_model": hidden_dim,
            "d_state": state_dim,
            "dropout": dropout,
            "transposed": True,
        },
        {
            "features": hidden_dim,
            "lmax": None,
            "N": state_dim,
            "dropout": dropout,
        },
        {
            "d_model": hidden_dim,
            "d_state": state_dim,
            "dropout": dropout,
        },
        {
            "features": hidden_dim,
            "N": state_dim,
            "dropout": dropout,
        },
        {
            "dim": hidden_dim,
            "state_dim": state_dim,
            "dropout": dropout,
        },
    ]
    for kwargs in constructor_candidates:
        try:
            return s4_cls(**kwargs)
        except Exception:
            continue
    try:
        return s4_cls(hidden_dim)
    except Exception:
        return None


class OfficialSSSDS4LayerAdapter(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int, dropout: float):
        super().__init__()
        s4_cls = _import_official_s4layer()
        if s4_cls is None:
            raise ImportError("Official SSSD S4Layer was not found.")

        module = _build_official_s4_instance(
            s4_cls=s4_cls,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            dropout=dropout,
        )
        if module is None:
            raise RuntimeError("Unable to instantiate official SSSD S4Layer with known signatures.")

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = _unwrap_module_output(self.module(x))
            if out.shape == x.shape:
                return out
        except Exception:
            pass

        x_bt = x.transpose(1, 2)
        out = _unwrap_module_output(self.module(x_bt))
        if out.shape == x_bt.shape:
            return out.transpose(1, 2)
        raise RuntimeError(
            "Official SSSD S4Layer returned an unexpected shape. "
            f"Input shape was {tuple(x.shape)}, output shape was {tuple(out.shape)}."
        )


class SSSDCompatibleTemporalBlock(nn.Module):
    _warning_printed = False

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        if not SSSDCompatibleTemporalBlock._warning_printed:
            print("WARNING: Using SSSD-compatible fallback temporal block, not official S4Layer.")
            SSSDCompatibleTemporalBlock._warning_printed = True

        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.output_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.depthwise(x)
        gate, value = self.pointwise(h).chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(value)
        h = self.output_proj(h)
        h = self.dropout(h)
        return residual + h


class SSSDResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        try:
            self.temporal_module = OfficialSSSDS4LayerAdapter(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                dropout=dropout,
            )
            self.uses_official_s4 = True
            self.temporal_impl_name = "official_s4layer"
        except Exception:
            self.temporal_module = SSSDCompatibleTemporalBlock(hidden_dim=hidden_dim, dropout=dropout)
            self.uses_official_s4 = False
            self.temporal_impl_name = "fallback_sssd_compatible_block"

        self.output_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, scale_shift: torch.Tensor) -> torch.Tensor:
        residual = x
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm(x)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(h)
        h = self.temporal_module(h)
        h = self.output_proj(h)
        h = self.dropout(h)
        return residual + h


class SSSDDenoiser(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        cond_channels: int = 2,
        hidden_dim: int = 64,
        out_channels: int = 2,
        n_layers: int = 4,
        timestep_dim: int = 128,
        dropout: float = 0.0,
        state_dim: int = 64,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.timestep_dim = timestep_dim
        self.state_dim = state_dim

        self.input_proj = nn.Conv1d(in_channels + cond_channels, hidden_dim, kernel_size=1)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(timestep_dim, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim * 2),
        )
        self.blocks = nn.ModuleList(
            [
                SSSDResidualBlock(
                    hidden_dim=hidden_dim,
                    state_dim=state_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

        self.uses_official_s4 = all(block.uses_official_s4 for block in self.blocks)
        self.temporal_impl_name = (
            "official_s4layer" if self.uses_official_s4 else "fallback_sssd_compatible_block"
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | int | float,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected x to have {self.in_channels} channels, got {x.shape[1]}")

        if cond is None:
            cond = torch.zeros_like(x)
        if cond.shape != x.shape:
            raise ValueError(f"Expected cond shape {tuple(x.shape)}, got {tuple(cond.shape)}")
        if cond.shape[1] != self.cond_channels:
            raise ValueError(f"Expected cond to have {self.cond_channels} channels, got {cond.shape[1]}")

        t_tensor = _coerce_timesteps(t=t, batch_size=x.shape[0], device=x.device)
        t_emb = sinusoidal_timestep_embedding(t_tensor, self.timestep_dim)
        scale_shift = self.timestep_mlp(t_emb)

        h_in = torch.cat([x, cond], dim=1)
        h = self.input_proj(h_in)
        for block in self.blocks:
            h = block(h, scale_shift)
        return self.output_proj(h)
