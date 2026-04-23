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
    LINEAR_METHOD_TAG,
    baseline_results_path,
    ensure_stage3_dirs,
    missing_span_path,
)


def validate_sample(mask: np.ndarray, index: int):
    observed_idx = np.flatnonzero(mask)
    if observed_idx.size < 2:
        raise ValueError(f"Sample {index} has fewer than 2 observed points.")
    if observed_idx[0] != 0 or observed_idx[-1] != len(mask) - 1:
        raise ValueError(
            f"Sample {index} is missing a boundary observation. "
            "Linear interpolation requires one observed point before and after the gap."
        )

    missing_idx = np.flatnonzero(mask == 0)
    if missing_idx.size == 0:
        return

    expected = np.arange(missing_idx[0], missing_idx[-1] + 1)
    if not np.array_equal(missing_idx, expected):
        raise ValueError(f"Sample {index} does not contain a single contiguous missing span.")


def interpolate_sample(traj_obs: np.ndarray, mask: np.ndarray, index: int):
    validate_sample(mask, index=index)
    t_idx = np.arange(traj_obs.shape[0], dtype=np.float32)
    observed_idx = np.flatnonzero(mask)
    traj_hat = traj_obs.copy()

    for dim in range(traj_obs.shape[1]):
        observed_values = traj_obs[observed_idx, dim]
        traj_hat[:, dim] = np.interp(t_idx, observed_idx.astype(np.float32), observed_values)

    return traj_hat


def main():
    parser = argparse.ArgumentParser(
        description="Run the Stage 3 linear interpolation baseline."
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
    args = parser.parse_args()

    ensure_stage3_dirs()
    input_path = Path(args.input_path) if args.input_path else missing_span_path(args.experiment_id)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else baseline_results_path(args.experiment_id, LINEAR_METHOD_TAG)
    )

    data = load_npz(input_path)
    required = ["traj_obs", "obs_mask"]
    for key in required:
        if key not in data:
            raise KeyError(f"'{key}' not found in {input_path}")

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
        traj_hat[i] = interpolate_sample(traj_obs[i], obs_mask[i], index=i)

    save_npz(output_path, traj_hat=traj_hat)

    print("=" * 60)
    print("Linear interpolation baseline finished")
    print(f"experiment_id = {args.experiment_id}")
    print(f"method_tag    = {LINEAR_METHOD_TAG}")
    print(f"input_path    = {input_path}")
    print(f"output_path   = {output_path}")
    print(f"traj_hat      = {traj_hat.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
