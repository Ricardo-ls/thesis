from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import save_npz
from utils.stage3.paths import OCCUPANCY_ROOM3_EMPTY_PATH, ensure_stage3_dirs

def main():
    parser = argparse.ArgumentParser(
        description="Build the Stage 3 Phase 1 canonical room3 empty-room occupancy map."
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=3.0)
    parser.add_argument("--y_min", type=float, default=0.0)
    parser.add_argument("--y_max", type=float, default=3.0)
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(OCCUPANCY_ROOM3_EMPTY_PATH),
    )
    args = parser.parse_args()

    ensure_stage3_dirs()

    if args.x_max <= args.x_min or args.y_max <= args.y_min:
        raise ValueError("Invalid map bounds: require x_max > x_min and y_max > y_min.")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive.")
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
    print(f"output_path  = {output_path}")
    print(f"occupancy    = {occupancy.shape}")
    print(f"x_range      = [{args.x_min}, {args.x_max}]")
    print(f"y_range      = [{args.y_min}, {args.y_max}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
