from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import load_npz, save_json
from utils.stage3.paths import DATA_OUT_DIR, EVAL_OUT_DIR, ensure_stage3_dirs


def infer_method_name(pred_path: Path):
    stem = pred_path.stem
    if stem.endswith("_results"):
        return stem[: -len("_results")]
    return stem


def load_map(map_path: Path):
    data = load_npz(map_path)
    required = ["occupancy", "x_min", "x_max", "y_min", "y_max"]
    for key in required:
        if key not in data:
            raise KeyError(f"'{key}' not found in {map_path}")
    occupancy = np.asarray(data["occupancy"], dtype=np.uint8)
    if occupancy.ndim != 2:
        raise ValueError(f"occupancy must be 2D, got {occupancy.shape}")
    return {
        "occupancy": occupancy,
        "x_min": float(np.asarray(data["x_min"]).item()),
        "x_max": float(np.asarray(data["x_max"]).item()),
        "y_min": float(np.asarray(data["y_min"]).item()),
        "y_max": float(np.asarray(data["y_max"]).item()),
    }


def point_to_grid(point: np.ndarray, map_meta: dict):
    x, y = float(point[0]), float(point[1])
    x_min = map_meta["x_min"]
    x_max = map_meta["x_max"]
    y_min = map_meta["y_min"]
    y_max = map_meta["y_max"]
    occupancy = map_meta["occupancy"]
    height, width = occupancy.shape

    if x < x_min or x > x_max or y < y_min or y > y_max:
        return False, None, None

    if x_max == x_min or y_max == y_min:
        raise ValueError("Invalid map bounds with zero range.")

    x_ratio = (x - x_min) / (x_max - x_min)
    y_ratio = (y - y_min) / (y_max - y_min)

    ix = min(width - 1, int(np.floor(x_ratio * width)))
    iy = min(height - 1, int(np.floor(y_ratio * height)))
    return True, iy, ix


def is_off_map(point: np.ndarray, map_meta: dict):
    valid, iy, ix = point_to_grid(point, map_meta)
    if not valid:
        return True
    return bool(map_meta["occupancy"][iy, ix] == 1)


def segment_crosses_wall(p0: np.ndarray, p1: np.ndarray, map_meta: dict):
    occupancy = map_meta["occupancy"]
    height, width = occupancy.shape
    cell_w = (map_meta["x_max"] - map_meta["x_min"]) / max(width, 1)
    cell_h = (map_meta["y_max"] - map_meta["y_min"]) / max(height, 1)
    spatial_scale = max(cell_w, cell_h, 1e-8)
    segment_len = float(np.linalg.norm(p1 - p0))
    num_samples = max(2, int(np.ceil(segment_len / spatial_scale)) + 1)

    for alpha in np.linspace(0.0, 1.0, num_samples):
        point = (1.0 - alpha) * p0 + alpha * p1
        valid, iy, ix = point_to_grid(point, map_meta)
        if valid and occupancy[iy, ix] == 1:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate minimal Stage 3 geometry feasibility metrics."
    )
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--pred_key", type=str, default="traj_hat")
    parser.add_argument(
        "--map_path",
        type=str,
        default=str(DATA_OUT_DIR / "occupancy_map.npz"),
    )
    parser.add_argument("--method_name", type=str, default=None)
    args = parser.parse_args()

    ensure_stage3_dirs()

    pred_data = load_npz(args.pred_path)
    if args.pred_key not in pred_data:
        raise KeyError(f"'{args.pred_key}' not found in {args.pred_path}")

    traj_hat = np.asarray(pred_data[args.pred_key], dtype=np.float32)
    if traj_hat.ndim != 3 or traj_hat.shape[-1] != 2:
        raise ValueError(f"Expected traj_hat shape [N, T, 2], got {traj_hat.shape}")

    map_meta = load_map(Path(args.map_path))

    off_map_count = 0
    wall_crossing_count = 0
    total_points = traj_hat.shape[0] * traj_hat.shape[1]
    total_segments = traj_hat.shape[0] * max(traj_hat.shape[1] - 1, 0)

    for traj in traj_hat:
        for point in traj:
            if is_off_map(point, map_meta):
                off_map_count += 1
        for t in range(traj.shape[0] - 1):
            if segment_crosses_wall(traj[t], traj[t + 1], map_meta):
                wall_crossing_count += 1

    metrics = {
        "method_name": args.method_name or infer_method_name(Path(args.pred_path)),
        "num_samples": int(traj_hat.shape[0]),
        "sequence_length": int(traj_hat.shape[1]),
        "off_map_ratio": float(off_map_count / total_points) if total_points > 0 else 0.0,
        "wall_crossing_count": int(wall_crossing_count),
        "total_points": int(total_points),
        "total_segments": int(total_segments),
    }

    output_path = EVAL_OUT_DIR / f"{metrics['method_name']}_geometry_metrics.json"
    save_json(output_path, metrics)

    print("=" * 60)
    print("Geometry metrics")
    print(f"method_name           = {metrics['method_name']}")
    print(f"pred_path             = {args.pred_path}")
    print(f"map_path              = {args.map_path}")
    print(f"off_map_ratio         = {metrics['off_map_ratio']:.6f}")
    print(f"wall_crossing_count   = {metrics['wall_crossing_count']}")
    print(f"output_path           = {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
