from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import save_npz
from utils.stage3.io import save_json
from utils.stage3.paths import (
    CLEAN_ABS_INPUT_PATH,
    CLEAN_ROOM3_META_PATH,
    CLEAN_ROOM3_PATH,
    ensure_stage3_dirs,
)


EXPECTED_SEQ_LEN = 20
SOURCE_X_MIN = -7.69
SOURCE_X_MAX = 15.613144
SOURCE_Y_MIN = -1.81
SOURCE_Y_MAX = 13.89191
TARGET_ROOM_WIDTH = 3.0
TARGET_ROOM_HEIGHT = 3.0
NORMALIZATION_MODE = "separate"


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


def normalize_to_room3(traj_abs: np.ndarray):
    if SOURCE_X_MAX <= SOURCE_X_MIN or SOURCE_Y_MAX <= SOURCE_Y_MIN:
        raise ValueError("Invalid source bounds for room3 normalization.")

    normalized = np.empty_like(traj_abs, dtype=np.float32)
    normalized[:, :, 0] = (
        (traj_abs[:, :, 0] - SOURCE_X_MIN)
        / (SOURCE_X_MAX - SOURCE_X_MIN)
        * TARGET_ROOM_WIDTH
    )
    normalized[:, :, 1] = (
        (traj_abs[:, :, 1] - SOURCE_Y_MIN)
        / (SOURCE_Y_MAX - SOURCE_Y_MIN)
        * TARGET_ROOM_HEIGHT
    )
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Build the Stage 3 Phase 1 canonical room3 clean dataset."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(CLEAN_ABS_INPUT_PATH),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(CLEAN_ROOM3_PATH),
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default=str(CLEAN_ROOM3_META_PATH),
    )
    args = parser.parse_args()

    ensure_stage3_dirs()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    raw = load_absolute_windows(input_path)
    source_traj_abs = validate_window_array(raw)
    traj_abs = normalize_to_room3(source_traj_abs)
    traj_rel = compute_rel(traj_abs)
    save_npz(output_path, traj_abs=traj_abs, traj_rel=traj_rel)

    metadata = {
        "source_input_path": str(input_path),
        "source_x_min": SOURCE_X_MIN,
        "source_x_max": SOURCE_X_MAX,
        "source_y_min": SOURCE_Y_MIN,
        "source_y_max": SOURCE_Y_MAX,
        "target_room_width": TARGET_ROOM_WIDTH,
        "target_room_height": TARGET_ROOM_HEIGHT,
        "normalization_mode": NORMALIZATION_MODE,
        "num_samples": int(traj_abs.shape[0]),
        "sequence_length": int(traj_abs.shape[1]),
    }
    meta_path = Path(args.meta_path)
    save_json(meta_path, metadata)

    print("=" * 60)
    print("Stage 3 canonical room3 clean dataset built")
    print(f"input_path         = {input_path}")
    print(f"output_path        = {output_path}")
    print(f"meta_path          = {meta_path}")
    print(f"normalization_mode = {NORMALIZATION_MODE}")
    print(f"room3_range        = x[0, 3], y[0, 3]")
    print(f"traj_abs shape     = {traj_abs.shape}")
    print(f"traj_rel shape     = {traj_rel.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
