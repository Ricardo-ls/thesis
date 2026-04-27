from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WallDoorGeometryProfile:
    profile_name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    wall_x: float
    door_y_min: float
    door_y_max: float


def canonical_room3_wall_door_v1() -> WallDoorGeometryProfile:
    return WallDoorGeometryProfile(
        profile_name="canonical_room3_wall_door_v1",
        x_min=0.0,
        x_max=3.0,
        y_min=0.0,
        y_max=3.0,
        wall_x=1.5,
        door_y_min=1.2,
        door_y_max=1.8,
    )


def is_boundary_violation(point: np.ndarray, profile: WallDoorGeometryProfile) -> bool:
    x = float(point[0])
    y = float(point[1])
    return x < profile.x_min or x > profile.x_max or y < profile.y_min or y > profile.y_max


def segment_crossing_y_at_wall(
    p0: np.ndarray,
    p1: np.ndarray,
    profile: WallDoorGeometryProfile,
) -> float | None:
    x0 = float(p0[0])
    x1 = float(p1[0])
    dx0 = x0 - profile.wall_x
    dx1 = x1 - profile.wall_x

    # Count only true side-to-side crossings, not endpoint touches or collinear runs.
    if dx0 == 0.0 or dx1 == 0.0 or dx0 * dx1 >= 0.0:
        return None

    t = (profile.wall_x - x0) / (x1 - x0)
    if t < 0.0 or t > 1.0:
        return None
    return float((1.0 - t) * p0[1] + t * p1[1])


def is_door_crossing(crossing_y: float, profile: WallDoorGeometryProfile) -> bool:
    return profile.door_y_min <= crossing_y <= profile.door_y_max


def compute_wall_door_metrics(
    traj: np.ndarray,
    profile: WallDoorGeometryProfile,
) -> dict[str, float | int]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"Expected trajectory batch [N, T, 2], got {traj.shape}")

    total_points = int(traj.shape[0] * traj.shape[1])
    total_segments = int(traj.shape[0] * max(traj.shape[1] - 1, 0))
    x = traj[:, :, 0]
    y = traj[:, :, 1]
    boundary_mask = (
        (x < profile.x_min)
        | (x > profile.x_max)
        | (y < profile.y_min)
        | (y > profile.y_max)
    )
    boundary_violation_count = int(boundary_mask.sum())

    if traj.shape[1] >= 2:
        x0 = x[:, :-1]
        y0 = y[:, :-1]
        x1 = x[:, 1:]
        y1 = y[:, 1:]

        dx0 = x0 - profile.wall_x
        dx1 = x1 - profile.wall_x
        crossing_mask = (dx0 != 0.0) & (dx1 != 0.0) & ((dx0 * dx1) < 0.0)

        denom = x1 - x0
        t = np.zeros_like(x0, dtype=np.float32)
        np.divide(profile.wall_x - x0, denom, out=t, where=denom != 0.0)
        crossing_y = (1.0 - t) * y0 + t * y1
        door_mask = crossing_mask & (crossing_y >= profile.door_y_min) & (crossing_y <= profile.door_y_max)

        endpoint_off_mask = boundary_mask[:, :-1] | boundary_mask[:, 1:]
        infeasible_transition_mask = endpoint_off_mask | (crossing_mask & ~door_mask)

        internal_wall_crossing_count = int(crossing_mask.sum())
        door_valid_crossing_count = int(door_mask.sum())
        infeasible_transition_count = int(infeasible_transition_mask.sum())
    else:
        internal_wall_crossing_count = 0
        door_valid_crossing_count = 0
        infeasible_transition_count = 0

    off_map_ratio = float(boundary_violation_count / total_points) if total_points > 0 else 0.0
    return {
        "off_map_ratio": off_map_ratio,
        "boundary_violation_count": int(boundary_violation_count),
        "internal_wall_crossing_count": int(internal_wall_crossing_count),
        "door_valid_crossing_count": int(door_valid_crossing_count),
        "infeasible_transition_count": int(infeasible_transition_count),
        "total_points": total_points,
        "total_segments": total_segments,
    }


def compute_wall_door_window_diagnostics(
    traj: np.ndarray,
    profile: WallDoorGeometryProfile,
    obs_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"Expected trajectory batch [N, T, 2], got {traj.shape}")

    num_windows = traj.shape[0]
    seq_len = traj.shape[1]
    total_points = int(num_windows * seq_len)
    total_segments = int(num_windows * max(seq_len - 1, 0))

    x = traj[:, :, 0]
    y = traj[:, :, 1]
    boundary_mask = (
        (x < profile.x_min)
        | (x > profile.x_max)
        | (y < profile.y_min)
        | (y > profile.y_max)
    )

    if seq_len >= 2:
        x0 = x[:, :-1]
        y0 = y[:, :-1]
        x1 = x[:, 1:]
        y1 = y[:, 1:]

        dx0 = x0 - profile.wall_x
        dx1 = x1 - profile.wall_x
        crossing_mask = (dx0 != 0.0) & (dx1 != 0.0) & ((dx0 * dx1) < 0.0)

        denom = x1 - x0
        t = np.zeros_like(x0, dtype=np.float32)
        np.divide(profile.wall_x - x0, denom, out=t, where=denom != 0.0)
        crossing_y = (1.0 - t) * y0 + t * y1
        door_mask = crossing_mask & (crossing_y >= profile.door_y_min) & (crossing_y <= profile.door_y_max)

        endpoint_off_mask = boundary_mask[:, :-1] | boundary_mask[:, 1:]
        infeasible_transition_mask = endpoint_off_mask | (crossing_mask & ~door_mask)
    else:
        crossing_mask = np.zeros((num_windows, 0), dtype=bool)
        door_mask = np.zeros((num_windows, 0), dtype=bool)
        infeasible_transition_mask = np.zeros((num_windows, 0), dtype=bool)

    clean_off_map_ratio = boundary_mask.mean(axis=1).astype(np.float32)
    boundary_violation_count_by_window = boundary_mask.sum(axis=1).astype(np.int64)
    internal_wall_crossing_count_by_window = crossing_mask.sum(axis=1).astype(np.int64)
    door_valid_crossing_count_by_window = door_mask.sum(axis=1).astype(np.int64)
    infeasible_transition_count_by_window = infeasible_transition_mask.sum(axis=1).astype(np.int64)
    window_has_violation = infeasible_transition_count_by_window > 0

    result: dict[str, np.ndarray | float | int] = {
        "clean_off_map_ratio_by_window": clean_off_map_ratio,
        "boundary_violation_count_by_window": boundary_violation_count_by_window,
        "internal_wall_crossing_count_by_window": internal_wall_crossing_count_by_window,
        "door_valid_crossing_count_by_window": door_valid_crossing_count_by_window,
        "infeasible_transition_count_by_window": infeasible_transition_count_by_window,
        "window_has_violation": window_has_violation,
        "infeasible_transition_mask": infeasible_transition_mask,
        "total_points": total_points,
        "total_segments": total_segments,
    }

    if obs_mask is not None:
        obs_mask = np.asarray(obs_mask, dtype=np.uint8)
        if obs_mask.shape != traj.shape[:2]:
            raise ValueError(f"obs_mask shape {obs_mask.shape} does not match trajectory shape {traj.shape[:2]}")
        if seq_len >= 2:
            masked_segment_mask = (obs_mask[:, :-1] == 0) | (obs_mask[:, 1:] == 0)
        else:
            masked_segment_mask = np.zeros((num_windows, 0), dtype=bool)
        result["masked_segment_mask"] = masked_segment_mask
        result["masked_segment_count"] = int(masked_segment_mask.sum())
        result["masked_infeasible_transition_count"] = int((infeasible_transition_mask & masked_segment_mask).sum())
    else:
        result["masked_segment_mask"] = None
        result["masked_segment_count"] = 0
        result["masked_infeasible_transition_count"] = 0

    return result
