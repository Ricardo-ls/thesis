from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from tools.stage3_indoor.generate_indoor_trajs import (
    BEHAVIORS,
    CLIP_MAX,
    CLIP_MIN,
    FPS,
    add_micro_jitter,
    generate_boundary_walk,
    generate_goal_directed,
    generate_multi_goal,
    generate_pacing,
    generate_stationary,
)


OUTPUT_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
LARGE_PATH = OUTPUT_DIR / "clean_trajs_large.npy"
TRAIN_PATH = OUTPUT_DIR / "train_trajs.npy"
VAL_PATH = OUTPUT_DIR / "val_trajs.npy"

N_TOTAL = 12000
TRAIN_N = 10000
VAL_N = 2000
SEED_START = 1000
T = 20


def generate_one_large(idx: int) -> np.ndarray:
    rng = np.random.default_rng(SEED_START + idx)
    behavior_names = [name for name, _ in BEHAVIORS]
    behavior_probs = [weight for _, weight in BEHAVIORS]
    behavior = str(rng.choice(behavior_names, p=behavior_probs))

    if behavior == "goal_directed":
        traj = generate_goal_directed(rng)
    elif behavior == "multi_goal":
        traj = generate_multi_goal(rng)
    elif behavior == "pacing":
        traj = generate_pacing(rng)
    elif behavior == "stationary":
        traj = generate_stationary(rng)
    elif behavior == "boundary_walk":
        traj = generate_boundary_walk(rng)
    else:
        raise ValueError(f"Unsupported behavior: {behavior}")

    traj = add_micro_jitter(traj, rng)
    traj = np.clip(traj, CLIP_MIN, CLIP_MAX).astype(np.float32)
    return traj


def save_array(path: Path, array: np.ndarray) -> None:
    np.save(path, array.astype(np.float32))
    print(f"Saved: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    large = np.zeros((N_TOTAL, T, 2), dtype=np.float32)
    for idx in range(N_TOTAL):
        large[idx] = generate_one_large(idx)

    train = large[:TRAIN_N].copy()
    val = large[TRAIN_N:TRAIN_N + VAL_N].copy()

    if large.shape != (12000, 20, 2):
        raise ValueError(f"Expected large shape (12000, 20, 2), got {large.shape}")
    if train.shape != (10000, 20, 2):
        raise ValueError(f"Expected train shape (10000, 20, 2), got {train.shape}")
    if val.shape != (2000, 20, 2):
        raise ValueError(f"Expected val shape (2000, 20, 2), got {val.shape}")

    save_array(LARGE_PATH, large)
    save_array(TRAIN_PATH, train)
    save_array(VAL_PATH, val)

    train_steps = np.linalg.norm(np.diff(train, axis=1), axis=2)
    train_mean_step = float(train_steps.mean())

    print("=== Part A 验证 ===")
    print(f"clean_trajs_large: {large.shape}")
    print(f"train_trajs:       {train.shape}")
    print(f"val_trajs:         {val.shape}")
    print(f"train x range: [{train[...,0].min():.4f}, {train[...,0].max():.4f}]")
    print(f"train y range: [{train[...,1].min():.4f}, {train[...,1].max():.4f}]")
    print(f"train mean step: {train_mean_step:.4f}")
    print("Part A outputs saved.")


if __name__ == "__main__":
    main()
