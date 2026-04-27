from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from tools.stage3.refinement.ddpm_refiner import ddpm_prior_interface_v0

REFINER_NAMES = [
    "identity_refiner",
    "light_savgol_refiner",
    "ddpm_prior_interface_v0",
]

REFINER_LABELS = {
    "identity_refiner": "Identity",
    "light_savgol_refiner": "Light SG",
    "ddpm_prior_interface_v0": "DDPM prior v0",
}


def validate_traj_batch(traj: np.ndarray):
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"Expected trajectory batch [N, T, 2], got {traj.shape}")
    if not np.all(np.isfinite(traj)):
        raise ValueError("Refinement input must be finite over the full trajectory.")
    return np.asarray(traj, dtype=np.float32)


def identity_refiner(traj: np.ndarray):
    return validate_traj_batch(traj).copy()


def light_savgol_refiner(traj: np.ndarray, window_length: int = 5, polyorder: int = 2):
    traj = validate_traj_batch(traj)
    num_samples, _, num_dims = traj.shape
    refined = np.zeros_like(traj, dtype=np.float32)
    for i in range(num_samples):
        for dim in range(num_dims):
            refined[i, :, dim] = savgol_filter(
                traj[i, :, dim],
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            ).astype(np.float32)
    return refined


def run_refiner(refiner_name: str, traj: np.ndarray):
    if refiner_name == "identity_refiner":
        return identity_refiner(traj), {"refiner_name": "identity_refiner"}
    if refiner_name == "light_savgol_refiner":
        return light_savgol_refiner(traj), {
            "refiner_name": "light_savgol_refiner",
            "window_length": 5,
            "polyorder": 2,
        }
    if refiner_name == "ddpm_prior_interface_v0":
        return ddpm_prior_interface_v0(traj)
    raise ValueError(f"Unsupported refiner: {refiner_name}")
