from __future__ import annotations

import numpy as np

from tools.stage3.geometry_extension.geometry_profiles import (
    GeometryProfile as WallDoorGeometryProfile,
    compute_geometry_window_diagnostics,
    wall_door_v1_profile,
)


def canonical_room3_wall_door_v1() -> WallDoorGeometryProfile:
    return wall_door_v1_profile()


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
    dx0 = x0 - float(profile.wall_x)
    dx1 = x1 - float(profile.wall_x)
    if dx0 == 0.0 or dx1 == 0.0 or dx0 * dx1 >= 0.0:
        return None
    t = (float(profile.wall_x) - x0) / (x1 - x0)
    if t < 0.0 or t > 1.0:
        return None
    return float((1.0 - t) * p0[1] + t * p1[1])


def is_door_crossing(crossing_y: float, profile: WallDoorGeometryProfile) -> bool:
    return float(profile.door_y_min) <= crossing_y <= float(profile.door_y_max)


def compute_wall_door_metrics(
    traj: np.ndarray,
    profile: WallDoorGeometryProfile,
) -> dict[str, float | int]:
    diagnostics = compute_geometry_window_diagnostics(traj, profile)
    boundary_violation_count = int(np.asarray(diagnostics["boundary_violation_count_by_window"]).sum())
    internal_wall_crossing_count = int(np.asarray(diagnostics["internal_wall_crossing_count_by_window"]).sum())
    door_valid_crossing_count = int(np.asarray(diagnostics["door_valid_crossing_count_by_window"]).sum())
    infeasible_transition_count = int(np.asarray(diagnostics["infeasible_transition_count_by_window"]).sum())
    total_points = int(diagnostics["total_points"])
    return {
        "off_map_ratio": float(boundary_violation_count / total_points) if total_points > 0 else 0.0,
        "boundary_violation_count": boundary_violation_count,
        "internal_wall_crossing_count": internal_wall_crossing_count,
        "door_valid_crossing_count": door_valid_crossing_count,
        "infeasible_transition_count": infeasible_transition_count,
        "total_points": total_points,
        "total_segments": int(diagnostics["total_segments"]),
    }


def compute_wall_door_window_diagnostics(
    traj: np.ndarray,
    profile: WallDoorGeometryProfile,
    obs_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    return compute_geometry_window_diagnostics(traj, profile, obs_mask=obs_mask)
