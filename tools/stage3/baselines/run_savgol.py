from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import load_npz, save_npz
from utils.stage3.paths import BASELINE_OUT_DIR, DATA_OUT_DIR, ensure_stage3_dirs


def validate_and_interp(traj_obs: np.ndarray, mask: np.ndarray, index: int):
    observed_idx = np.flatnonzero(mask)
    if observed_idx.size < 2:
        raise ValueError(f"Sample {index} has fewer than 2 observed points.")
    if observed_idx[0] != 0 or observed_idx[-1] != len(mask) - 1:
        raise ValueError(
            f"Sample {index} is missing a boundary observation. "
            "Savitzky-Golay baseline requires one observed point before and after the gap."
        )

    missing_idx = np.flatnonzero(mask == 0)
    if missing_idx.size > 0:
        expected = np.arange(missing_idx[0], missing_idx[-1] + 1)
        if not np.array_equal(missing_idx, expected):
            raise ValueError(f"Sample {index} does not contain a single contiguous missing span.")

    t_idx = np.arange(traj_obs.shape[0], dtype=np.float32)
    traj_hat = traj_obs.copy()
    for dim in range(traj_obs.shape[1]):
        values = traj_obs[observed_idx, dim]
        traj_hat[:, dim] = np.interp(t_idx, observed_idx.astype(np.float32), values)
    return traj_hat


def main():
    parser = argparse.ArgumentParser(
        description="Run the Stage 3 Savitzky-Golay baseline."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(DATA_OUT_DIR / "missing_span_windows.npz"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(BASELINE_OUT_DIR / "savgol_results.npz"),
    )
    parser.add_argument("--window_length", type=int, default=5)
    parser.add_argument("--polyorder", type=int, default=2)
    args = parser.parse_args()

    ensure_stage3_dirs()

    if args.window_length <= 0 or args.window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer.")
    if args.polyorder < 0:
        raise ValueError("polyorder must be non-negative.")
    if args.polyorder >= args.window_length:
        raise ValueError("polyorder must be smaller than window_length.")

    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise SystemExit(
            "scipy is required for run_savgol.py. Install scipy to use this baseline."
        ) from exc

    data = load_npz(args.input_path)
    if "traj_obs" not in data or "obs_mask" not in data:
        raise KeyError(f"'traj_obs' and 'obs_mask' are required in {args.input_path}")

    traj_obs = np.asarray(data["traj_obs"], dtype=np.float32)
    obs_mask = np.asarray(data["obs_mask"], dtype=np.uint8)
    if traj_obs.ndim != 3 or traj_obs.shape[-1] != 2:
        raise ValueError(f"Expected traj_obs shape [N, T, 2], got {traj_obs.shape}")
    if obs_mask.shape != traj_obs.shape[:2]:
        raise ValueError(
            f"obs_mask shape {obs_mask.shape} does not match traj_obs shape {traj_obs.shape[:2]}"
        )
    if args.window_length > traj_obs.shape[1]:
        raise ValueError(
            f"window_length {args.window_length} cannot exceed sequence length {traj_obs.shape[1]}"
        )

    traj_hat = np.zeros_like(traj_obs, dtype=np.float32)
    for i in range(traj_obs.shape[0]):
        filled = validate_and_interp(traj_obs[i], obs_mask[i], index=i)
        for dim in range(filled.shape[1]):
            traj_hat[i, :, dim] = savgol_filter(
                filled[:, dim],
                window_length=args.window_length,
                polyorder=args.polyorder,
                mode="interp",
            ).astype(np.float32)

    output_path = Path(args.output_path)
    save_npz(output_path, traj_hat=traj_hat)

    print("=" * 60)
    print("Savitzky-Golay baseline finished")
    print(f"input_path      = {args.input_path}")
    print(f"output_path     = {output_path}")
    print(f"window_length   = {args.window_length}")
    print(f"polyorder       = {args.polyorder}")
    print(f"traj_hat        = {traj_hat.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
