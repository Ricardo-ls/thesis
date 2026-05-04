from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_indoor_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
TRAJ_PATH = OUTPUT_DIR / "clean_trajs.npy"
META_PATH = OUTPUT_DIR / "trajs_metadata.json"
GRID_PATH = OUTPUT_DIR / "visualization_grid.png"
DIST_PATH = OUTPUT_DIR / "distribution_check.png"

ROOM_MIN = 0.0
ROOM_MAX = 3.0
SAFE_MIN = 0.2
SAFE_MAX = 2.8
CLIP_MIN = 0.05
CLIP_MAX = 2.95
REFLECT_MARGIN = 0.15
T = 20
FPS = 3.0
N_TRAJ = 2000
GLOBAL_SEED = 42

BEHAVIORS = [
    ("goal_directed", 0.40),
    ("multi_goal", 0.25),
    ("pacing", 0.15),
    ("stationary", 0.10),
    ("boundary_walk", 0.10),
]


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


def reflect_step(pos: np.ndarray, step: np.ndarray) -> np.ndarray:
    next_pos = pos + step
    out = step.copy()
    if next_pos[0] < REFLECT_MARGIN or next_pos[0] > (ROOM_MAX - REFLECT_MARGIN):
        out[0] *= -1.0
    if next_pos[1] < REFLECT_MARGIN or next_pos[1] > (ROOM_MAX - REFLECT_MARGIN):
        out[1] *= -1.0
    return out


def sample_point(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(SAFE_MIN, SAFE_MAX, size=(2,)).astype(np.float32)


def fill_with_last(points: list[np.ndarray]) -> np.ndarray:
    if len(points) >= T:
        return np.asarray(points[:T], dtype=np.float32)
    last = points[-1].copy()
    while len(points) < T:
        points.append(last.copy())
    return np.asarray(points, dtype=np.float32)


def generate_goal_directed(rng: np.random.Generator) -> np.ndarray:
    start = sample_point(rng)
    end = sample_point(rng)
    base_speed = float(rng.uniform(0.5, 1.0))
    direction = end - start
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / dist

    pos = start.copy()
    points = [pos.copy()]
    for _ in range(T - 1):
        speed = base_speed * float(rng.uniform(0.9, 1.1))
        step = direction * np.float32(speed / FPS)
        step = reflect_step(pos, step)
        pos = pos + step
        points.append(pos.copy())
    return np.asarray(points, dtype=np.float32)


def segment_walk(rng: np.random.Generator, start: np.ndarray, end: np.ndarray, speed: float, max_frames: int) -> list[np.ndarray]:
    pos = start.copy()
    points = [pos.copy()]
    direction = end - start
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return points
    direction = direction / dist
    for _ in range(max_frames - 1):
        remain = end - pos
        if float(np.linalg.norm(remain)) <= (speed / FPS):
            pos = end.copy()
            points.append(pos.copy())
            break
        step = direction * np.float32(speed / FPS)
        step = reflect_step(pos, step)
        pos = pos + step
        points.append(pos.copy())
    return points


def generate_multi_goal(rng: np.random.Generator) -> np.ndarray:
    start = sample_point(rng)
    mid = sample_point(rng)
    end = sample_point(rng)
    points = [start.copy()]
    targets = [mid, end]
    for target_idx, target in enumerate(targets):
        speed = float(rng.uniform(0.5, 1.0))
        segment = segment_walk(rng, points[-1], target, speed=speed, max_frames=max(3, T - len(points) + 1))
        points.extend(segment[1:])
        if len(points) >= T:
            break
        if target_idx == 0:
            for _ in range(int(rng.integers(1, 3))):
                points.append(points[-1].copy())
                if len(points) >= T:
                    break
    return fill_with_last(points)


def generate_pacing(rng: np.random.Generator) -> np.ndarray:
    center = sample_point(rng)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(rng.uniform(0.25, 0.75))
    offset = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    p0 = np.clip(center - offset, SAFE_MIN, SAFE_MAX)
    p1 = np.clip(center + offset, SAFE_MIN, SAFE_MAX)
    speed = float(rng.uniform(0.5, 1.0))
    current_target = p1.copy()
    pos = p0.copy()
    points = [pos.copy()]
    stop_left = 0
    for _ in range(T - 1):
        if stop_left > 0:
            points.append(pos.copy())
            stop_left -= 1
            continue
        remain = current_target - pos
        dist = float(np.linalg.norm(remain))
        if dist <= (speed / FPS):
            pos = current_target.copy()
            points.append(pos.copy())
            stop_left = int(rng.integers(1, 3))
            current_target = p0.copy() if np.allclose(current_target, p1) else p1.copy()
            continue
        direction = remain / max(dist, 1e-8)
        step = direction * np.float32(speed / FPS)
        step = reflect_step(pos, step)
        pos = pos + step
        points.append(pos.copy())
    return np.asarray(points, dtype=np.float32)


def generate_stationary(rng: np.random.Generator) -> np.ndarray:
    center = sample_point(rng)
    pos = center.copy()
    points = [pos.copy()]
    for _ in range(T - 1):
        step = rng.normal(0.0, 0.03, size=(2,)).astype(np.float32)
        candidate = pos + step
        if float(np.linalg.norm(candidate - center)) > 0.3:
            candidate = center + 0.5 * (candidate - center)
        candidate = np.clip(candidate, REFLECT_MARGIN, ROOM_MAX - REFLECT_MARGIN)
        pos = candidate.astype(np.float32)
        points.append(pos.copy())
    return np.asarray(points, dtype=np.float32)


def generate_boundary_walk(rng: np.random.Generator) -> np.ndarray:
    wall = str(rng.choice(["left", "right", "bottom", "top"]))
    offset = float(rng.uniform(0.3, 0.5))
    speed = float(rng.uniform(0.4, 0.8))
    step_mag = speed / FPS
    if wall == "left":
        pos = np.array([offset, rng.uniform(SAFE_MIN, SAFE_MAX)], dtype=np.float32)
        direction = np.array([0.0, rng.choice([-1.0, 1.0])], dtype=np.float32)
        parallel_axis = 1
    elif wall == "right":
        pos = np.array([ROOM_MAX - offset, rng.uniform(SAFE_MIN, SAFE_MAX)], dtype=np.float32)
        direction = np.array([0.0, rng.choice([-1.0, 1.0])], dtype=np.float32)
        parallel_axis = 1
    elif wall == "bottom":
        pos = np.array([rng.uniform(SAFE_MIN, SAFE_MAX), offset], dtype=np.float32)
        direction = np.array([rng.choice([-1.0, 1.0]), 0.0], dtype=np.float32)
        parallel_axis = 0
    else:
        pos = np.array([rng.uniform(SAFE_MIN, SAFE_MAX), ROOM_MAX - offset], dtype=np.float32)
        direction = np.array([rng.choice([-1.0, 1.0]), 0.0], dtype=np.float32)
        parallel_axis = 0

    points = [pos.copy()]
    for _ in range(T - 1):
        step = direction * np.float32(step_mag)
        candidate = pos + step
        if candidate[parallel_axis] < REFLECT_MARGIN or candidate[parallel_axis] > (ROOM_MAX - REFLECT_MARGIN):
            if wall in {"left", "right"}:
                direction = np.array([rng.choice([-1.0, 1.0]), 0.0], dtype=np.float32)
            else:
                direction = np.array([0.0, rng.choice([-1.0, 1.0])], dtype=np.float32)
            step = direction * np.float32(step_mag)
            candidate = pos + step
        candidate = np.clip(candidate, REFLECT_MARGIN, ROOM_MAX - REFLECT_MARGIN)
        pos = candidate.astype(np.float32)
        points.append(pos.copy())
    return np.asarray(points, dtype=np.float32)


def add_micro_jitter(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jitter = rng.normal(0.0, 0.005, size=traj.shape).astype(np.float32)
    return (traj + jitter).astype(np.float32)


def generate_one(idx: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(GLOBAL_SEED + idx)
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
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    mean_speed = float(steps.mean() * FPS)
    meta = {"idx": int(idx), "behavior": behavior, "mean_speed": mean_speed}
    return traj, meta


def plot_room(ax):
    ax.plot([0, 3, 3, 0, 0], [0, 0, 3, 3, 0], color="black", linewidth=1.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(alpha=0.2, linewidth=0.5)


def save_visualization_grid(trajs: np.ndarray, metadata: list[dict]):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    behavior_to_indices = {}
    for meta in metadata:
        behavior_to_indices.setdefault(meta["behavior"], []).append(meta["idx"])
    rng = np.random.default_rng(123)
    draw_plan = [("goal_directed", 4), ("multi_goal", 3), ("pacing", 3), ("stationary", 3), ("boundary_walk", 3)]
    selected = []
    for behavior, n_draw in draw_plan:
        picks = rng.choice(behavior_to_indices[behavior], size=n_draw, replace=False)
        for idx in picks:
            selected.append((behavior, int(idx)))

    for ax, (behavior, idx) in zip(axes.flat, selected):
        plot_room(ax)
        traj = trajs[idx]
        ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", linewidth=1.8)
        ax.scatter(traj[0, 0], traj[0, 1], color="green", s=40)
        ax.scatter(traj[-1, 0], traj[-1, 1], color="red", s=40, marker="s")
        ax.set_title(behavior, fontsize=10)

    fig.tight_layout()
    fig.savefig(GRID_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved: {GRID_PATH}")


def save_distribution_check(trajs: np.ndarray):
    steps = np.linalg.norm(np.diff(trajs, axis=1), axis=2).reshape(-1)
    diffs = np.diff(trajs, axis=1)
    angles = np.degrees(np.arctan2(diffs[..., 1], diffs[..., 0]))
    angle_diff = ((np.diff(angles, axis=1) + 180.0) % 360.0) - 180.0
    starts = trajs[:, 0, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(steps, bins=40, color="#4c78a8", alpha=0.85)
    axes[0].axvline(float(steps.mean()), color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Step Size Histogram")
    axes[0].set_xlabel("m / frame")

    axes[1].hist(angle_diff.reshape(-1), bins=40, color="#f58518", alpha=0.85)
    axes[1].set_title("Direction Delta Histogram")
    axes[1].set_xlabel("degrees")

    axes[2].scatter(starts[:, 0], starts[:, 1], s=10, alpha=0.5, color="#54a24b")
    axes[2].set_title("Start Position Scatter")
    axes[2].set_xlim(0, 3)
    axes[2].set_ylim(0, 3)
    axes[2].set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(DIST_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved: {DIST_PATH}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trajs = np.zeros((N_TRAJ, T, 2), dtype=np.float32)
    metadata: list[dict] = []
    for idx in range(N_TRAJ):
        traj, meta = generate_one(idx)
        trajs[idx] = traj
        metadata.append(meta)

    np.save(TRAJ_PATH, trajs)
    print(f"Saved: {TRAJ_PATH}")
    save_json(META_PATH, metadata)
    save_visualization_grid(trajs, metadata)
    save_distribution_check(trajs)

    steps = np.linalg.norm(np.diff(trajs, axis=1), axis=2)
    mean_step = float(steps.mean())
    std_step = float(steps.std())
    mean_speed = float(mean_step * FPS)

    counts = {name: 0 for name, _ in BEHAVIORS}
    for meta in metadata:
        counts[meta["behavior"]] += 1

    print("=== Step 1 验证输出 ===")
    print(f"shape: {trajs.shape}")
    print(f"x range: [{trajs[...,0].min():.4f}, {trajs[...,0].max():.4f}]   预期 ~[0.05, 2.95]")
    print(f"y range: [{trajs[...,1].min():.4f}, {trajs[...,1].max():.4f}]   预期 ~[0.05, 2.95]")
    print(f"mean step (m/frame): {mean_step:.4f}                            预期 0.10~0.30")
    print(f"std  step (m/frame): {std_step:.4f}                             预期 < 0.10")
    print(f"mean speed (m/s):    {mean_speed:.4f}                           预期 0.3~1.0")
    print("behavior distribution:")
    print(f"  goal_directed : {counts['goal_directed']} 条")
    print(f"  multi_goal    : {counts['multi_goal']} 条")
    print(f"  pacing        : {counts['pacing']} 条")
    print(f"  stationary    : {counts['stationary']} 条")
    print(f"  boundary_walk : {counts['boundary_walk']} 条")
    print("输出已保存:")
    print("  data/stage3_indoor/clean_trajs.npy")
    print("  data/stage3_indoor/trajs_metadata.json")
    print("  data/stage3_indoor/visualization_grid.png")
    print("  data/stage3_indoor/distribution_check.png")


if __name__ == "__main__":
    main()
