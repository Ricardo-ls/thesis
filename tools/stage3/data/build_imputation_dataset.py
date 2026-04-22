from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import save_json, save_npz
from utils.stage3.paths import DATA_OUT_DIR, ensure_stage3_dirs


def load_input_array(input_path: Path, input_key: str):
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        return np.load(input_path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(input_path, allow_pickle=False) as data:
            if input_key not in data.files:
                raise KeyError(
                    f"Input key '{input_key}' not found in {input_path}. "
                    f"Available keys: {list(data.files)}"
                )
            return data[input_key]
    raise ValueError(f"Unsupported input format: {input_path.suffix}. Use .npy or .npz.")


def validate_window_array(array: np.ndarray, window_size: int):
    if array.ndim != 3:
        raise ValueError(
            f"Mode 'windows' expects an array with shape [N, T, 2], got {array.shape}"
        )
    if array.shape[-1] != 2:
        raise ValueError(
            f"Trajectory coordinate dimension must be 2, got shape {array.shape}"
        )
    if array.shape[1] != window_size:
        raise ValueError(
            f"window_size mismatch: expected T={window_size}, got T={array.shape[1]}"
        )
    return array.astype(np.float32, copy=False)


def cut_long_trajectory(array: np.ndarray, window_size: int, stride: int):
    if array.ndim != 2 or array.shape[-1] != 2:
        raise ValueError(
            f"Mode 'long' expects one trajectory with shape [L, 2], got {array.shape}"
        )
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if window_size <= 1:
        raise ValueError(f"window_size must be at least 2, got {window_size}")
    length = array.shape[0]
    if length < window_size:
        raise ValueError(
            f"Long trajectory length {length} is smaller than window_size {window_size}"
        )

    windows = []
    start_indices = []
    for start in range(0, length - window_size + 1, stride):
        end = start + window_size
        windows.append(array[start:end])
        start_indices.append(start)

    if not windows:
        raise RuntimeError("No windows were generated. Check window_size and stride.")

    return np.stack(windows, axis=0).astype(np.float32), np.asarray(start_indices, dtype=np.int64)


def compute_rel(traj_abs: np.ndarray):
    return np.diff(traj_abs, axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build the minimal Stage 3 clean trajectory window dataset."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--input_key", type=str, default="traj_abs")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--mode", type=str, required=True, choices=["windows", "long"])
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

    raw = load_input_array(input_path, args.input_key)

    if args.mode == "windows":
        traj_abs = validate_window_array(raw, window_size=args.window_size)
        sample_ids = np.arange(traj_abs.shape[0], dtype=np.int64)
        meta = {
            "mode": "windows",
            "input_path": str(input_path),
            "input_key": args.input_key,
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "num_samples": int(traj_abs.shape[0]),
            "sequence_length": int(traj_abs.shape[1]),
        }
    else:
        traj_abs, start_indices = cut_long_trajectory(
            np.asarray(raw, dtype=np.float32),
            window_size=args.window_size,
            stride=args.stride,
        )
        sample_ids = start_indices
        meta = {
            "mode": "long",
            "input_path": str(input_path),
            "input_key": args.input_key,
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "num_samples": int(traj_abs.shape[0]),
            "sequence_length": int(traj_abs.shape[1]),
            "window_start_indices": start_indices.tolist(),
        }

    traj_rel = compute_rel(traj_abs)
    save_npz(
        output_path,
        traj_abs=traj_abs,
        traj_rel=traj_rel,
        sample_ids=sample_ids,
    )

    meta_path = output_path.with_name(f"{output_path.stem}_meta.json")
    save_json(meta_path, meta)

    print("=" * 60)
    print("Stage 3 clean dataset built")
    print(f"input_path      = {input_path}")
    print(f"mode            = {args.mode}")
    print(f"output_path     = {output_path}")
    print(f"meta_path       = {meta_path}")
    print(f"traj_abs shape  = {traj_abs.shape}")
    print(f"traj_rel shape  = {traj_rel.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
