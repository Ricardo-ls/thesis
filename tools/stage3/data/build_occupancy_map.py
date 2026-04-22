from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import save_npz
from utils.stage3.paths import DATA_OUT_DIR, ensure_stage3_dirs


def load_occupancy_array(input_path: Path):
    suffix = input_path.suffix.lower()
    if suffix == ".npy":
        occupancy = np.load(input_path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(input_path, allow_pickle=False) as data:
            if "occupancy" in data.files:
                occupancy = data["occupancy"]
            elif len(data.files) == 1:
                occupancy = data[data.files[0]]
            else:
                raise KeyError(
                    f"Could not resolve occupancy array in {input_path}. "
                    f"Available keys: {list(data.files)}"
                )
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}. Use .npy or .npz.")

    if occupancy.ndim != 2:
        raise ValueError(f"Occupancy map must be 2D, got shape {occupancy.shape}")
    return (occupancy > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Build the minimal Stage 3 occupancy map / free-space mask."
    )
    parser.add_argument("--mode", type=str, required=True, choices=["load", "empty_room"])
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=10.0)
    parser.add_argument("--y_min", type=float, default=0.0)
    parser.add_argument("--y_max", type=float, default=10.0)
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DATA_OUT_DIR / "occupancy_map.npz"),
    )
    args = parser.parse_args()

    ensure_stage3_dirs()

    if args.x_max <= args.x_min or args.y_max <= args.y_min:
        raise ValueError("Invalid map bounds: require x_max > x_min and y_max > y_min.")

    if args.mode == "load":
        if args.input_path is None:
            raise ValueError("--input_path is required when mode=load")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input occupancy file does not exist: {input_path}")
        occupancy = load_occupancy_array(input_path)
    else:
        if args.height <= 0 or args.width <= 0:
            raise ValueError("height and width must be positive in empty_room mode.")
        occupancy = np.zeros((args.height, args.width), dtype=np.uint8)

    output_path = Path(args.output_path)
    save_npz(
        output_path,
        occupancy=occupancy,
        x_min=np.asarray(args.x_min, dtype=np.float32),
        x_max=np.asarray(args.x_max, dtype=np.float32),
        y_min=np.asarray(args.y_min, dtype=np.float32),
        y_max=np.asarray(args.y_max, dtype=np.float32),
    )

    print("=" * 60)
    print("Stage 3 occupancy map built")
    print(f"mode         = {args.mode}")
    print(f"output_path  = {output_path}")
    print(f"occupancy    = {occupancy.shape}")
    print(f"x_range      = [{args.x_min}, {args.x_max}]")
    print(f"y_range      = [{args.y_min}, {args.y_max}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
