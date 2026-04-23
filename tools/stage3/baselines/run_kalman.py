from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import load_npz, save_npz
from utils.stage3.paths import (
    DEFAULT_EXPERIMENT_ID,
    KALMAN_METHOD_TAG,
    baseline_results_path,
    ensure_stage3_dirs,
    missing_span_path,
)


def build_kalman_mats(dt: float, process_var: float, measure_var: float):
    f = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    q = process_var * np.array(
        [
            [dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0, 0.0],
            [0.0, dt ** 4 / 4.0, 0.0, dt ** 3 / 2.0],
            [dt ** 3 / 2.0, 0.0, dt ** 2, 0.0],
            [0.0, dt ** 3 / 2.0, 0.0, dt ** 2],
        ],
        dtype=np.float64,
    )
    r = measure_var * np.eye(2, dtype=np.float64)
    return f, h, q, r


def validate_sample(mask: np.ndarray, index: int):
    observed_idx = np.flatnonzero(mask)
    if observed_idx.size < 2:
        raise ValueError(f"Sample {index} has fewer than 2 observed points.")
    if observed_idx[0] != 0 or observed_idx[-1] != len(mask) - 1:
        raise ValueError(
            f"Sample {index} is missing a boundary observation. "
            "This minimal Kalman baseline assumes the sequence starts and ends with observations."
        )
    missing_idx = np.flatnonzero(mask == 0)
    if missing_idx.size > 0:
        expected = np.arange(missing_idx[0], missing_idx[-1] + 1)
        if not np.array_equal(missing_idx, expected):
            raise ValueError(f"Sample {index} does not contain a single contiguous missing span.")


def init_state(traj_obs: np.ndarray, mask: np.ndarray, dt: float):
    observed_idx = np.flatnonzero(mask)
    pos0 = traj_obs[observed_idx[0]]
    pos1 = traj_obs[observed_idx[1]]
    step_gap = max(int(observed_idx[1] - observed_idx[0]), 1)
    vel0 = (pos1 - pos0) / (step_gap * dt)

    state = np.array([pos0[0], pos0[1], vel0[0], vel0[1]], dtype=np.float64)
    cov = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)
    return state, cov


def kalman_reconstruct(traj_obs: np.ndarray, mask: np.ndarray, dt: float, process_var: float, measure_var: float):
    f, h, q, r = build_kalman_mats(dt=dt, process_var=process_var, measure_var=measure_var)
    state, cov = init_state(traj_obs=traj_obs, mask=mask, dt=dt)
    traj_hat = np.zeros_like(traj_obs, dtype=np.float32)
    identity = np.eye(4, dtype=np.float64)

    for t in range(traj_obs.shape[0]):
        if t > 0:
            state = f @ state
            cov = f @ cov @ f.T + q

        if mask[t]:
            z = traj_obs[t].astype(np.float64)
            innovation = z - h @ state
            s = h @ cov @ h.T + r
            k = cov @ h.T @ np.linalg.inv(s)
            state = state + k @ innovation
            cov = (identity - k @ h) @ cov

        traj_hat[t] = state[:2].astype(np.float32)

    return traj_hat


def main():
    parser = argparse.ArgumentParser(
        description="Run the minimal Stage 3 constant-velocity Kalman baseline."
    )
    parser.add_argument("--experiment_id", type=str, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--process_var", type=float, default=1e-3)
    parser.add_argument("--measure_var", type=float, default=1e-2)
    args = parser.parse_args()

    ensure_stage3_dirs()
    input_path = Path(args.input_path) if args.input_path else missing_span_path(args.experiment_id)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else baseline_results_path(args.experiment_id, KALMAN_METHOD_TAG)
    )

    if args.dt <= 0:
        raise ValueError("dt must be positive.")
    if args.process_var <= 0 or args.measure_var <= 0:
        raise ValueError("process_var and measure_var must be positive.")

    data = load_npz(input_path)
    if "traj_obs" not in data or "obs_mask" not in data:
        raise KeyError(f"'traj_obs' and 'obs_mask' are required in {input_path}")

    traj_obs = np.asarray(data["traj_obs"], dtype=np.float32)
    obs_mask = np.asarray(data["obs_mask"], dtype=np.uint8)
    if traj_obs.ndim != 3 or traj_obs.shape[-1] != 2:
        raise ValueError(f"Expected traj_obs shape [N, T, 2], got {traj_obs.shape}")
    if obs_mask.shape != traj_obs.shape[:2]:
        raise ValueError(
            f"obs_mask shape {obs_mask.shape} does not match traj_obs shape {traj_obs.shape[:2]}"
        )

    traj_hat = np.zeros_like(traj_obs, dtype=np.float32)
    for i in range(traj_obs.shape[0]):
        validate_sample(obs_mask[i], index=i)
        traj_hat[i] = kalman_reconstruct(
            traj_obs=traj_obs[i],
            mask=obs_mask[i],
            dt=args.dt,
            process_var=args.process_var,
            measure_var=args.measure_var,
        )

    save_npz(output_path, traj_hat=traj_hat)

    print("=" * 60)
    print("Kalman baseline finished")
    print(f"experiment_id  = {args.experiment_id}")
    print(f"method_tag     = {KALMAN_METHOD_TAG}")
    print(f"input_path     = {input_path}")
    print(f"output_path    = {output_path}")
    print(f"dt             = {args.dt}")
    print(f"process_var    = {args.process_var}")
    print(f"measure_var    = {args.measure_var}")
    print(f"traj_hat       = {traj_hat.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
