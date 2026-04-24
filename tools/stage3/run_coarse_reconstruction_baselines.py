from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.signal import savgol_filter

from tools.stage3.baselines.run_kalman import kalman_reconstruct, validate_sample as validate_kalman
from tools.stage3.baselines.run_linear_interp import interpolate_sample
from tools.stage3.baselines.run_savgol import validate_and_interp
from utils.stage3.controlled_benchmark import (
    DEFAULT_DRIFT_AMP,
    DEFAULT_NOISE_STD,
    DEFAULT_SEED,
    DEFAULT_SPAN_MODE,
    DEFAULT_SPAN_RATIO,
    DEGRADATION_NAMES,
    METHODS,
    degraded_path,
    ensure_controlled_dirs,
    experiment_tag,
    reconstruction_path,
)


def obs_mask_from_degraded(degraded: np.ndarray):
    if degraded.ndim != 3 or degraded.shape[-1] != 2:
        raise ValueError(f"Expected degraded trajectories with shape [N, T, 2], got {degraded.shape}")
    observed = np.all(np.isfinite(degraded), axis=-1).astype(np.uint8)
    return observed


def run_linear(degraded: np.ndarray, obs_mask: np.ndarray):
    traj_hat = np.zeros_like(degraded, dtype=np.float32)
    for i in range(degraded.shape[0]):
        traj_hat[i] = interpolate_sample(degraded[i], obs_mask[i], index=i)
    return traj_hat


def run_savgol(degraded: np.ndarray, obs_mask: np.ndarray, window_length: int = 5, polyorder: int = 2):
    traj_hat = np.zeros_like(degraded, dtype=np.float32)
    for i in range(degraded.shape[0]):
        filled = validate_and_interp(degraded[i], obs_mask[i], index=i)
        for dim in range(filled.shape[1]):
            traj_hat[i, :, dim] = savgol_filter(
                filled[:, dim],
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            ).astype(np.float32)
    return traj_hat


def run_kalman(degraded: np.ndarray, obs_mask: np.ndarray, dt: float = 1.0, process_var: float = 1e-3, measure_var: float = 1e-2):
    traj_hat = np.zeros_like(degraded, dtype=np.float32)
    for i in range(degraded.shape[0]):
        validate_kalman(obs_mask[i], index=i)
        traj_hat[i] = kalman_reconstruct(
            traj_obs=degraded[i],
            mask=obs_mask[i],
            dt=dt,
            process_var=process_var,
            measure_var=measure_var,
        )
    return traj_hat


def main():
    parser = argparse.ArgumentParser(
        description="Run coarse reconstruction baselines for controlled Stage 3 degradations."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--span_ratio", type=float, default=DEFAULT_SPAN_RATIO)
    parser.add_argument("--span_mode", type=str, default=DEFAULT_SPAN_MODE, choices=["fixed", "random"])
    parser.add_argument("--noise_std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--drift_amp", type=float, default=DEFAULT_DRIFT_AMP)
    args = parser.parse_args()

    ensure_controlled_dirs()
    tag = experiment_tag(args.span_ratio, args.span_mode, args.seed)

    print("=" * 60)
    print("Running controlled coarse reconstruction baselines")
    print(f"experiment_tag = {tag}")

    for degradation_name in DEGRADATION_NAMES:
        input_path = degraded_path(degradation_name, tag)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Degraded input not found for {degradation_name}: {input_path}. "
                "Run build_controlled_degradation first."
            )
        degraded = np.load(input_path, allow_pickle=False).astype(np.float32)
        obs_mask = obs_mask_from_degraded(degraded)

        np.save(reconstruction_path(degradation_name, "input_degraded", tag), degraded)

        linear = run_linear(degraded, obs_mask)
        np.save(reconstruction_path(degradation_name, "linear_interp", tag), linear)

        sg = run_savgol(degraded, obs_mask, window_length=5, polyorder=2)
        np.save(reconstruction_path(degradation_name, "savgol_w5_p2", tag), sg)

        kalman = run_kalman(
            degraded,
            obs_mask,
            dt=1.0,
            process_var=1e-3,
            measure_var=1e-2,
        )
        np.save(reconstruction_path(degradation_name, "kalman_cv_dt1.0_q1e-3_r1e-2", tag), kalman)

        print(f"[done] {degradation_name}")
        print(f"       input    -> {reconstruction_path(degradation_name, 'input_degraded', tag)}")
        for method in METHODS:
            print(f"       {method:34s}-> {reconstruction_path(degradation_name, method, tag)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
