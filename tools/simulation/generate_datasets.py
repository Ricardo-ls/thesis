from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from indoor_trajectory_generator import IndoorTrajectoryGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "simulated"

T = 20
FPS = 3.0


def save_array(path: Path, array: np.ndarray):
    np.save(path, array)
    print(f"Saved: {path}")


def save_json(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Saved: {path}")


def assert_shape(name: str, array: np.ndarray, expected_shape: tuple[int, ...]):
    if tuple(array.shape) != tuple(expected_shape):
        raise RuntimeError(f"{name} shape mismatch: expected {expected_shape}, got {array.shape}")


def build_split(map_specs: list[tuple[str, int, int]]):
    traj_parts = []
    curvature_parts = []
    total_failed_slots = 0
    total_candidates = 0
    total_accepted = 0
    total_crossing_count = 0

    for map_type, seed, n_traj in map_specs:
        generator = IndoorTrajectoryGenerator(map_type=map_type, seed=seed)
        trajs, curvatures = generator.generate(n_traj=n_traj, T=T, fps=FPS)
        crossing_count = int(sum(generator.has_wall_crossing(traj) for traj in trajs))
        total_crossing_count += crossing_count
        traj_parts.append(trajs.astype(np.float32))
        curvature_parts.append(curvatures.astype(np.float32))
        total_failed_slots += int(generator.failed_slot_count)
        total_candidates += int(generator.total_candidate_count)
        total_accepted += int(generator.accepted_count)

    trajs_all = np.concatenate(traj_parts, axis=0).astype(np.float32)
    curvatures_all = np.concatenate(curvature_parts, axis=0).astype(np.float32)

    stats = {
        "failed_slot_count": total_failed_slots,
        "total_candidate_count": total_candidates,
        "accepted_count": total_accepted,
        "wall_crossing_count": total_crossing_count,
    }
    return trajs_all, curvatures_all, stats


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_specs = [
        ("open_room", 1000, 10000),
        ("obstacle", 1001, 10000),
    ]
    val_specs = [
        ("open_room", 2000, 1000),
        ("obstacle", 2001, 1000),
    ]
    test_specs = [
        ("L_shape", 3000, 2500),
        ("two_room", 3001, 2500),
    ]

    train, train_curv, train_stats = build_split(train_specs)
    val, val_curv, val_stats = build_split(val_specs)
    test, test_curv, test_stats = build_split(test_specs)

    assert_shape("train_trajs.npy", train, (20000, T, 2))
    assert_shape("val_trajs.npy", val, (2000, T, 2))
    assert_shape("test_trajs.npy", test, (5000, T, 2))
    assert_shape("test_curvature.npy", test_curv, (5000,))

    all_trajs = np.concatenate([train, val, test], axis=0).astype(np.float32)
    all_steps = np.linalg.norm(np.diff(all_trajs, axis=1), axis=-1)
    mean_step = float(all_steps.mean())
    std_step = float(all_steps.std())
    mean_curv = float(test_curv.mean())
    p25 = float(np.percentile(test_curv, 25))
    p50 = float(np.percentile(test_curv, 50))
    p75 = float(np.percentile(test_curv, 75))

    total_requested = 20000 + 2000 + 5000
    total_failed_slots = train_stats["failed_slot_count"] + val_stats["failed_slot_count"] + test_stats["failed_slot_count"]
    total_wall_crossing_count = train_stats["wall_crossing_count"] + val_stats["wall_crossing_count"] + test_stats["wall_crossing_count"]
    wall_rate = float(total_wall_crossing_count / max(1, all_trajs.shape[0]))
    fail_rate = float(total_failed_slots / total_requested)

    if wall_rate != 0.0:
        raise RuntimeError(f"wall_crossing_rate must be 0.0, got {wall_rate}")

    save_array(DATA_DIR / "train_trajs.npy", train.astype(np.float32))
    save_array(DATA_DIR / "val_trajs.npy", val.astype(np.float32))
    save_array(DATA_DIR / "test_trajs.npy", test.astype(np.float32))
    save_array(DATA_DIR / "test_curvature.npy", test_curv.astype(np.float32))

    metadata = {
        "train_n": 20000,
        "val_n": 2000,
        "test_n": 5000,
        "T": T,
        "fps": FPS,
        "train_maps": ["open_room", "obstacle"],
        "val_maps": ["open_room", "obstacle"],
        "test_maps": ["L_shape", "two_room"],
        "mean_step_size": mean_step,
        "std_step_size": std_step,
        "mean_curvature": mean_curv,
        "curvature_p25": p25,
        "curvature_p50": p50,
        "curvature_p75": p75,
        "wall_crossing_rate": wall_rate,
        "generation_failure_rate": fail_rate,
    }
    save_json(DATA_DIR / "metadata.json", metadata)

    print("=== 指令1 验证输出 ===")
    print(f"train shape : {train.shape}")
    print(f"val shape   : {val.shape}")
    print(f"test shape  : {test.shape}")
    print(f"mean_step_size      : {mean_step:.4f} m")
    print(f"std_step_size       : {std_step:.4f} m")
    print(f"mean_curvature      : {mean_curv:.4f}")
    print(f"curvature_p25/p50/p75: {p25:.4f} / {p50:.4f} / {p75:.4f}")
    print(f"wall_crossing_rate  : {wall_rate:.6f}  ← 必须为 0.0")
    print(f"failure_rate        : {fail_rate:.4f}")
    print("=== 等待人工确认 ===")


if __name__ == "__main__":
    main()
