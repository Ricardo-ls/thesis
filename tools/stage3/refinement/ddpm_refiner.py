from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.temporal_denoiser import TemporalDenoiser1D
from diffusion.ddpm_utils import DDPMForwardProcess
from utils.prior.ablation_paths import get_recommended_prior_paths, to_abs_path
from utils.prior.run_metadata import resolve_current_run_metadata


@dataclass
class DDPMPriorInterfaceConfig:
    objective: str = "optimization_best"
    train_seed: int = 42
    train_epochs: int = 100
    timesteps: int = 100
    hidden_dim: int = 128
    projection_timestep: int = 20
    blend_weight: float = 0.35
    device: str = "cpu"
    batch_size: int = 1024


def resolve_device(device_name: str = "auto"):
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda device, but CUDA is not available.")
        return "cuda"
    if device_name == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device: {device_name}")


def abs_to_rel(abs_traj: np.ndarray):
    if abs_traj.ndim != 3 or abs_traj.shape[-1] != 2:
        raise ValueError(f"Expected absolute trajectories [N, T, 2], got {abs_traj.shape}")
    return np.diff(abs_traj, axis=1).astype(np.float32)


def rel_to_abs(rel_traj: np.ndarray, start_points: np.ndarray):
    if rel_traj.ndim != 3 or rel_traj.shape[-1] != 2:
        raise ValueError(f"Expected relative trajectories [N, T-1, 2], got {rel_traj.shape}")
    if start_points.ndim != 2 or start_points.shape[-1] != 2:
        raise ValueError(f"Expected start points [N, 2], got {start_points.shape}")
    cumsum = np.cumsum(rel_traj, axis=1)
    return np.concatenate([start_points[:, None, :], start_points[:, None, :] + cumsum], axis=1).astype(
        np.float32
    )


@lru_cache(maxsize=4)
def _load_prior_checkpoint_cached(
    objective: str,
    train_seed: int,
    train_epochs: int,
    timesteps: int,
    hidden_dim: int,
    device: str,
):
    config = DDPMPriorInterfaceConfig(
        objective=objective,
        train_seed=train_seed,
        train_epochs=train_epochs,
        timesteps=timesteps,
        hidden_dim=hidden_dim,
        device=device,
    )
    prior_paths = get_recommended_prior_paths(config.objective)
    ckpt_meta = resolve_current_run_metadata(
        variant=prior_paths["variant"],
        train_seed=config.train_seed,
        train_epochs=config.train_epochs,
        train_root=PROJECT_ROOT / "outputs" / "prior" / "train",
    )
    ckpt_path = Path(ckpt_meta["ckpt_path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Prior checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=config.device)
    model = TemporalDenoiser1D(
        max_timesteps=config.timesteps,
        in_channels=2,
        hidden_dim=config.hidden_dim,
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, prior_paths, ckpt_meta


def load_prior_checkpoint(config: DDPMPriorInterfaceConfig):
    return _load_prior_checkpoint_cached(
        config.objective,
        config.train_seed,
        config.train_epochs,
        config.timesteps,
        config.hidden_dim,
        config.device,
    )


@torch.no_grad()
def ddpm_prior_interface_v0(
    coarse_abs: np.ndarray,
    config: DDPMPriorInterfaceConfig | None = None,
):
    """First-version learned-prior refiner.

    This is intentionally a minimal projection-style interface rather than a
    full conditional diffusion posterior. It treats the coarse relative
    displacement sequence as a noisy trajectory sample at a fixed diffusion
    timestep, performs one x0 reconstruction with the Stage 2 prior, and then
    blends that prior projection back with the coarse input.
    """

    config = config or DDPMPriorInterfaceConfig()
    config.device = resolve_device(config.device)

    coarse_abs = np.asarray(coarse_abs, dtype=np.float32)
    rel = abs_to_rel(coarse_abs)
    start_points = coarse_abs[:, 0, :]

    model, checkpoint, prior_paths, ckpt_meta = load_prior_checkpoint(config)
    diffusion = DDPMForwardProcess(timesteps=config.timesteps, device=config.device)

    x_t_np = np.transpose(rel, (0, 2, 1)).astype(np.float32)
    num_samples = x_t_np.shape[0]
    t_scalar = int(np.clip(config.projection_timestep, 0, config.timesteps - 1))
    blend = float(np.clip(config.blend_weight, 0.0, 1.0))
    refined_batches = []
    for start in range(0, num_samples, config.batch_size):
        end = min(start + config.batch_size, num_samples)
        x_t = torch.from_numpy(x_t_np[start:end]).to(config.device)
        t = torch.full((x_t.shape[0],), t_scalar, device=config.device, dtype=torch.long)
        pred_noise = model(x_t, t)
        alpha_bar_t = diffusion.alpha_bars[t].view(-1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        x_refined = (1.0 - blend) * x_t + blend * x0_pred
        refined_batches.append(x_refined.cpu().numpy())
    rel_refined = np.transpose(np.concatenate(refined_batches, axis=0), (0, 2, 1)).astype(np.float32)
    refined_abs = rel_to_abs(rel_refined, start_points)

    metadata = {
        "refiner_name": "ddpm_prior_interface_v0",
        "refiner_type": "projection_style_prior_denoise",
        "stage2_objective": config.objective,
        "stage2_variant": prior_paths["variant"],
        "stage2_train_seed": config.train_seed,
        "stage2_train_epochs": config.train_epochs,
        "stage2_checkpoint_path": str(ckpt_meta["ckpt_path"]),
        "stage2_best_epoch": ckpt_meta["current_run_best_epoch"],
        "stage2_best_val_loss": ckpt_meta["current_run_best_val_loss"],
        "projection_timestep": t_scalar,
        "blend_weight": blend,
        "device": config.device,
        "limitation": (
            "This v0 interface is not a full conditional diffusion posterior. "
            "It performs a one-shot prior projection in relative displacement space."
        ),
    }
    return refined_abs, metadata


def ddpm_prior_masked_replace_v1(
    coarse_abs: np.ndarray,
    obs_mask: np.ndarray,
    ddpm_candidate: np.ndarray | None = None,
    base_metadata: dict | None = None,
    config: DDPMPriorInterfaceConfig | None = None,
):
    """Prior-based candidate generation with masked-only replacement.

    Observed points are preserved from the coarse trajectory. Only the missing
    segment is replaced by the DDPM prior candidate.
    """

    coarse_abs = np.asarray(coarse_abs, dtype=np.float32)
    obs_mask = np.asarray(obs_mask, dtype=np.uint8)
    if coarse_abs.ndim != 3 or coarse_abs.shape[-1] != 2:
        raise ValueError(f"Expected coarse_abs [N, T, 2], got {coarse_abs.shape}")
    if obs_mask.shape != coarse_abs.shape[:2]:
        raise ValueError(f"obs_mask shape {obs_mask.shape} does not match coarse_abs shape {coarse_abs.shape[:2]}")

    if ddpm_candidate is None or base_metadata is None:
        ddpm_candidate, base_meta = ddpm_prior_interface_v0(coarse_abs, config=config)
    else:
        ddpm_candidate = np.asarray(ddpm_candidate, dtype=np.float32)
        base_meta = dict(base_metadata)
    if ddpm_candidate.shape != coarse_abs.shape:
        raise ValueError(
            f"ddpm_candidate shape {ddpm_candidate.shape} does not match coarse_abs shape {coarse_abs.shape}"
        )
    refined = coarse_abs.copy()
    observed = obs_mask == 1
    missing = obs_mask == 0
    refined[missing] = ddpm_candidate[missing]
    refined[observed] = coarse_abs[observed]

    metadata = dict(base_meta)
    metadata.update(
        {
            "refiner_name": "ddpm_prior_masked_replace_v1",
            "refiner_type": "projection_style_prior_masked_replace",
            "base_candidate_refiner": "ddpm_prior_interface_v0",
            "replace_rule": "refined[mask == 1] = coarse[mask == 1]; refined[mask == 0] = ddpm_candidate[mask == 0]",
        }
    )
    return refined.astype(np.float32), metadata
