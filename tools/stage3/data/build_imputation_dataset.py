from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import save_npz
from utils.stage3.paths import CLEAN_ABS_INPUT_PATH, DATA_OUT_DIR, ensure_stage3_dirs


EXPECTED_SEQ_LEN = 20


def load_absolute_windows(input_path: Path):
    if input_path.suffix.lower() != ".npy":
        raise ValueError(
            f"Stage 3 Phase 1 expects a .npy absolute-window file, got {input_path.suffix}"
        )
    return np.load(input_path, allow_pickle=False)


def validate_window_array(array: np.ndarray):
    if array.ndim != 3:
        raise ValueError(
            f"Expected clean absolute trajectory windows with shape [N, T, 2], got {array.shape}"
        )
    if array.shape[-1] != 2:
        raise ValueError(
            f"Expected 2D absolute coordinates in the last dimension, got shape {array.shape}"
        )
    if array.shape[1] != EXPECTED_SEQ_LEN:
        raise ValueError(
            f"Expected sequence length T={EXPECTED_SEQ_LEN} for data_eth_ucy_20.npy, got T={array.shape[1]}"
        )
    return array.astype(np.float32, copy=False)


def compute_rel(traj_abs: np.ndarray):
    return np.diff(traj_abs, axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build the minimal Stage 3 clean dataset from pre-windowed absolute trajectories."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(CLEAN_ABS_INPUT_PATH),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DATA_OUT_DIR / "clean_windows.npz"),
    )
    args = parser.parse_args()

    ensure_stage3_dirs()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    raw = load_absolute_windows(input_path)
    traj_abs = validate_window_array(raw)
    traj_rel = compute_rel(traj_abs)
    save_npz(output_path, traj_abs=traj_abs, traj_rel=traj_rel)

    print("=" * 60)
    print("Stage 3 clean dataset built")
    print(f"input_path      = {input_path}")
    print(f"output_path     = {output_path}")
    print(f"traj_abs shape  = {traj_abs.shape}")
    print(f"traj_rel shape  = {traj_rel.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
