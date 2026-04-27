from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryProfile:
    profile_name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    constraint_type: str
    wall_x: float | None = None
    door_y_min: float | None = None
    door_y_max: float | None = None
    obstacle_x_min: float | None = None
    obstacle_x_max: float | None = None
    obstacle_y_min: float | None = None
    obstacle_y_max: float | None = None


def wall_door_v1_profile() -> GeometryProfile:
    return GeometryProfile(
        profile_name="wall_door_v1",
        x_min=0.0,
        x_max=3.0,
        y_min=0.0,
        y_max=3.0,
        constraint_type="internal_wall_with_door",
        wall_x=1.5,
        door_y_min=1.2,
        door_y_max=1.8,
    )


def obstacle_v1_profile() -> GeometryProfile:
    return GeometryProfile(
        profile_name="obstacle_v1",
        x_min=0.0,
        x_max=3.0,
        y_min=0.0,
        y_max=3.0,
        constraint_type="central_rectangular_obstacle",
        obstacle_x_min=1.2,
        obstacle_x_max=1.8,
        obstacle_y_min=1.2,
        obstacle_y_max=1.8,
    )


def two_room_v1_profile() -> GeometryProfile:
    return GeometryProfile(
        profile_name="two_room_v1",
        x_min=0.0,
        x_max=3.0,
        y_min=0.0,
        y_max=3.0,
        constraint_type="internal_wall_with_narrow_door",
        wall_x=1.5,
        door_y_min=1.35,
        door_y_max=1.65,
    )


def all_geometry_profiles() -> list[GeometryProfile]:
    return [wall_door_v1_profile(), obstacle_v1_profile(), two_room_v1_profile()]


def _boundary_mask(traj: np.ndarray, profile: GeometryProfile) -> np.ndarray:
    x = traj[:, :, 0]
    y = traj[:, :, 1]
    return (x < profile.x_min) | (x > profile.x_max) | (y < profile.y_min) | (y > profile.y_max)


def _wall_crossing_masks(traj: np.ndarray, profile: GeometryProfile) -> tuple[np.ndarray, np.ndarray]:
    num_windows, seq_len, _ = traj.shape
    if seq_len < 2 or profile.wall_x is None or profile.door_y_min is None or profile.door_y_max is None:
        empty = np.zeros((num_windows, max(seq_len - 1, 0)), dtype=bool)
        return empty, empty

    x0 = traj[:, :-1, 0]
    y0 = traj[:, :-1, 1]
    x1 = traj[:, 1:, 0]
    y1 = traj[:, 1:, 1]

    dx0 = x0 - profile.wall_x
    dx1 = x1 - profile.wall_x
    crossing_mask = (dx0 != 0.0) & (dx1 != 0.0) & ((dx0 * dx1) < 0.0)

    denom = x1 - x0
    t = np.zeros_like(x0, dtype=np.float32)
    np.divide(profile.wall_x - x0, denom, out=t, where=denom != 0.0)
    crossing_y = (1.0 - t) * y0 + t * y1
    door_mask = crossing_mask & (crossing_y >= profile.door_y_min) & (crossing_y <= profile.door_y_max)
    return crossing_mask, door_mask


def _point_in_obstacle_mask(traj: np.ndarray, profile: GeometryProfile) -> np.ndarray:
    if (
        profile.obstacle_x_min is None
        or profile.obstacle_x_max is None
        or profile.obstacle_y_min is None
        or profile.obstacle_y_max is None
    ):
        return np.zeros(traj.shape[:2], dtype=bool)
    x = traj[:, :, 0]
    y = traj[:, :, 1]
    return (
        (x >= profile.obstacle_x_min)
        & (x <= profile.obstacle_x_max)
        & (y >= profile.obstacle_y_min)
        & (y <= profile.obstacle_y_max)
    )


def _obstacle_segment_crossing_mask(traj: np.ndarray, profile: GeometryProfile) -> np.ndarray:
    num_windows, seq_len, _ = traj.shape
    if seq_len < 2:
        return np.zeros((num_windows, 0), dtype=bool)
    if (
        profile.obstacle_x_min is None
        or profile.obstacle_x_max is None
        or profile.obstacle_y_min is None
        or profile.obstacle_y_max is None
    ):
        return np.zeros((num_windows, seq_len - 1), dtype=bool)

    x0 = traj[:, :-1, 0]
    y0 = traj[:, :-1, 1]
    x1 = traj[:, 1:, 0]
    y1 = traj[:, 1:, 1]

    inside0 = (
        (x0 >= profile.obstacle_x_min)
        & (x0 <= profile.obstacle_x_max)
        & (y0 >= profile.obstacle_y_min)
        & (y0 <= profile.obstacle_y_max)
    )
    inside1 = (
        (x1 >= profile.obstacle_x_min)
        & (x1 <= profile.obstacle_x_max)
        & (y1 >= profile.obstacle_y_min)
        & (y1 <= profile.obstacle_y_max)
    )

    dx = x1 - x0
    dy = y1 - y0
    u1 = np.zeros_like(dx, dtype=np.float32)
    u2 = np.ones_like(dx, dtype=np.float32)
    valid = np.ones_like(dx, dtype=bool)

    for p, q in (
        (-dx, x0 - profile.obstacle_x_min),
        (dx, profile.obstacle_x_max - x0),
        (-dy, y0 - profile.obstacle_y_min),
        (dy, profile.obstacle_y_max - y0),
    ):
        zero_mask = p == 0.0
        valid &= ~(zero_mask & (q < 0.0))

        nz_mask = ~zero_mask
        t = np.zeros_like(dx, dtype=np.float32)
        np.divide(q, p, out=t, where=nz_mask)

        neg_mask = nz_mask & (p < 0.0)
        pos_mask = nz_mask & (p > 0.0)

        u1 = np.where(neg_mask, np.maximum(u1, t), u1)
        valid &= ~(neg_mask & (t > u2))

        u2 = np.where(pos_mask, np.minimum(u2, t), u2)
        valid &= ~(pos_mask & (t < u1))

    return inside0 | inside1 | (valid & (u1 <= u2))


def compute_geometry_window_diagnostics(
    traj: np.ndarray,
    profile: GeometryProfile,
    obs_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"Expected trajectory batch [N, T, 2], got {traj.shape}")

    num_windows = traj.shape[0]
    seq_len = traj.shape[1]
    total_points = int(num_windows * seq_len)
    total_segments = int(num_windows * max(seq_len - 1, 0))

    boundary_mask = _boundary_mask(traj, profile)
    endpoint_off_mask = (boundary_mask[:, :-1] | boundary_mask[:, 1:]) if seq_len >= 2 else np.zeros((num_windows, 0), dtype=bool)

    result: dict[str, np.ndarray | float | int] = {
        "clean_off_map_ratio_by_window": boundary_mask.mean(axis=1).astype(np.float32),
        "boundary_violation_count_by_window": boundary_mask.sum(axis=1).astype(np.int64),
        "total_points": total_points,
        "total_segments": total_segments,
    }

    if profile.constraint_type in {"internal_wall_with_door", "internal_wall_with_narrow_door"}:
        crossing_mask, door_mask = _wall_crossing_masks(traj, profile)
        infeasible_transition_mask = endpoint_off_mask | (crossing_mask & ~door_mask)
        result.update(
            {
                "internal_wall_crossing_count_by_window": crossing_mask.sum(axis=1).astype(np.int64),
                "door_valid_crossing_count_by_window": door_mask.sum(axis=1).astype(np.int64),
                "infeasible_transition_count_by_window": infeasible_transition_mask.sum(axis=1).astype(np.int64),
                "window_has_violation": (infeasible_transition_mask.sum(axis=1) > 0),
                "infeasible_transition_mask": infeasible_transition_mask,
            }
        )
    elif profile.constraint_type == "central_rectangular_obstacle":
        obstacle_point_mask = _point_in_obstacle_mask(traj, profile)
        obstacle_segment_mask = _obstacle_segment_crossing_mask(traj, profile)
        infeasible_transition_mask = endpoint_off_mask | obstacle_segment_mask
        result.update(
            {
                "obstacle_point_violation_count_by_window": obstacle_point_mask.sum(axis=1).astype(np.int64),
                "obstacle_segment_crossing_count_by_window": obstacle_segment_mask.sum(axis=1).astype(np.int64),
                "infeasible_transition_count_by_window": infeasible_transition_mask.sum(axis=1).astype(np.int64),
                "window_has_violation": (infeasible_transition_mask.sum(axis=1) > 0),
                "infeasible_transition_mask": infeasible_transition_mask,
            }
        )
    else:
        raise ValueError(f"Unsupported geometry profile: {profile.constraint_type}")

    if obs_mask is not None:
        obs_mask = np.asarray(obs_mask, dtype=np.uint8)
        if obs_mask.shape != traj.shape[:2]:
            raise ValueError(f"obs_mask shape {obs_mask.shape} does not match trajectory shape {traj.shape[:2]}")
        masked_segment_mask = (obs_mask[:, :-1] == 0) | (obs_mask[:, 1:] == 0) if seq_len >= 2 else np.zeros((num_windows, 0), dtype=bool)
        result["masked_segment_mask"] = masked_segment_mask
        result["masked_segment_count"] = int(masked_segment_mask.sum())
        result["masked_infeasible_transition_count"] = int((result["infeasible_transition_mask"] & masked_segment_mask).sum())
    else:
        result["masked_segment_mask"] = None
        result["masked_segment_count"] = 0
        result["masked_infeasible_transition_count"] = 0

    return result
