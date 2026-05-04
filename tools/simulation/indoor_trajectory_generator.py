from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Rect:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, point: np.ndarray, eps: float = 1e-9) -> bool:
        x, y = float(point[0]), float(point[1])
        return (self.x_min - eps) <= x <= (self.x_max + eps) and (self.y_min - eps) <= y <= (self.y_max + eps)

    def center(self) -> np.ndarray:
        return np.array([(self.x_min + self.x_max) * 0.5, (self.y_min + self.y_max) * 0.5], dtype=np.float32)


class IndoorTrajectoryGenerator:
    MAP_DEFS = {
        "open_room": {
            "feasible_rects": [Rect(0.0, 3.0, 0.0, 3.0)],
            "blocked_rects": [],
            "bbox": Rect(0.0, 3.0, 0.0, 3.0),
        },
        "obstacle": {
            "feasible_rects": [Rect(0.0, 3.0, 0.0, 3.0)],
            "blocked_rects": [Rect(1.25, 1.75, 1.25, 1.75)],
            "bbox": Rect(0.0, 3.0, 0.0, 3.0),
        },
        "L_shape": {
            "feasible_rects": [Rect(0.0, 3.0, 0.0, 2.0), Rect(0.0, 1.5, 0.0, 3.0)],
            "blocked_rects": [],
            "bbox": Rect(0.0, 3.0, 0.0, 3.0),
        },
        "two_room": {
            "feasible_rects": [Rect(0.0, 2.5, 0.0, 3.0), Rect(0.5, 3.0, 1.0, 2.0), Rect(1.5, 3.0, 0.0, 3.0)],
            "blocked_rects": [],
            "bbox": Rect(0.0, 3.0, 0.0, 3.0),
        },
    }

    def __init__(self, map_type, seed):
        if map_type not in self.MAP_DEFS:
            raise ValueError(f"Unsupported map_type={map_type}. Supported: {sorted(self.MAP_DEFS.keys())}")
        self.map_type = str(map_type)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        cfg = self.MAP_DEFS[self.map_type]
        self.feasible_rects = list(cfg["feasible_rects"])
        self.blocked_rects = list(cfg["blocked_rects"])
        self.bbox = cfg["bbox"]
        self.boundary_segments = self._build_boundary_segments()
        self.failed_slot_count = 0
        self.total_candidate_count = 0
        self.accepted_count = 0
        self.last_wall_crossing_rate = 0.0
        self.last_generation_failure_rate = 0.0

    def generate(self, n_traj, T=20, fps=3.0):
        n_traj = int(n_traj)
        T = int(T)
        fps = float(fps)
        if n_traj <= 0:
            raise ValueError("n_traj must be positive.")
        if T < 2:
            raise ValueError("T must be at least 2.")
        if fps <= 0:
            raise ValueError("fps must be positive.")

        trajs = np.zeros((n_traj, T, 2), dtype=np.float32)
        curvatures = np.zeros((n_traj,), dtype=np.float32)

        for slot_idx in range(n_traj):
            consecutive_failures = 0
            while True:
                self.total_candidate_count += 1
                candidate = self._build_candidate_trajectory(T=T, fps=fps)
                if candidate is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 20:
                        self.failed_slot_count += 1
                        consecutive_failures = 0
                    continue

                if not self.validate_trajectory(candidate):
                    consecutive_failures += 1
                    if consecutive_failures >= 20:
                        self.failed_slot_count += 1
                        consecutive_failures = 0
                    continue

                trajs[slot_idx] = candidate
                curvatures[slot_idx] = self.compute_curvature(candidate)
                self.accepted_count += 1
                break

        crossing_mask = np.array([self.has_wall_crossing(traj) for traj in trajs], dtype=np.float32)
        self.last_wall_crossing_rate = float(crossing_mask.mean()) if crossing_mask.size > 0 else 0.0
        self.last_generation_failure_rate = float(self.failed_slot_count / max(1, n_traj))
        return trajs.astype(np.float32), curvatures.astype(np.float32)

    def compute_curvature(self, traj: np.ndarray) -> np.float32:
        traj = np.asarray(traj, dtype=np.float32)
        diff1 = np.diff(traj, axis=0)
        diff2 = np.diff(diff1, axis=0)
        if diff2.shape[0] == 0:
            return np.float32(0.0)
        return np.float32(np.linalg.norm(diff2, axis=1).mean())

    def validate_trajectory(self, traj: np.ndarray) -> bool:
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(f"Expected trajectory shape [T, 2], got {traj.shape}")
        for idx in range(traj.shape[0] - 1):
            if not self.segment_is_feasible(traj[idx], traj[idx + 1]):
                return False
        return True

    def has_wall_crossing(self, traj: np.ndarray) -> bool:
        return not self.validate_trajectory(traj)

    def _build_candidate_trajectory(self, T: int, fps: float) -> np.ndarray | None:
        current = self._sample_point_with_margin(margin=0.15)
        frames = [current.astype(np.float32)]

        while len(frames) < T:
            target = self._sample_point_with_margin(margin=0.15)
            v_max = float(self.rng.uniform(0.3, 0.8))
            distance = float(np.linalg.norm(target - current))
            walk_time = distance / max(v_max, 1e-8)
            n_walk = max(3, int(round(walk_time * fps)))
            walk_segment = self._generate_walk_segment(current, target, n_walk=n_walk)

            if not self.validate_trajectory(walk_segment):
                return None

            if len(frames) == 1:
                frames.extend(walk_segment[1:].astype(np.float32))
            else:
                frames.extend(walk_segment[1:].astype(np.float32))

            n_stop = int(self.rng.integers(0, 4))
            for _ in range(n_stop):
                frames.append(target.astype(np.float32))

            current = target.astype(np.float32)

        return np.asarray(frames[:T], dtype=np.float32)

    def _generate_walk_segment(self, start: np.ndarray, end: np.ndarray, n_walk: int) -> np.ndarray:
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        if n_walk < 2:
            raise ValueError("n_walk must be at least 2.")
        n_intervals = n_walk - 1
        weights = self._interval_weights(n_intervals)
        cumulative = np.concatenate([[0.0], np.cumsum(weights, axis=0)], axis=0)
        s = cumulative / cumulative[-1]
        return (start[None, :] + s[:, None].astype(np.float32) * (end - start)[None, :]).astype(np.float32)

    def _interval_weights(self, n_intervals: int) -> np.ndarray:
        if n_intervals <= 0:
            return np.ones((1,), dtype=np.float32)
        first_len = max(1, n_intervals // 3)
        second_len = max(1, n_intervals // 3)
        third_len = max(1, n_intervals - first_len - second_len)
        while first_len + second_len + third_len > n_intervals:
            third_len -= 1
        while first_len + second_len + third_len < n_intervals:
            second_len += 1

        accel = np.linspace(0.5, 1.0, num=first_len, dtype=np.float32)
        cruise = np.ones((second_len,), dtype=np.float32)
        decel = np.linspace(1.0, 0.5, num=third_len, dtype=np.float32)
        weights = np.concatenate([accel, cruise, decel], axis=0)
        if weights.shape[0] != n_intervals:
            raise RuntimeError(f"Weight profile length mismatch: expected {n_intervals}, got {weights.shape[0]}")
        return weights

    def _sample_point_with_margin(self, margin: float) -> np.ndarray:
        while True:
            point = np.array(
                [
                    self.rng.uniform(self.bbox.x_min, self.bbox.x_max),
                    self.rng.uniform(self.bbox.y_min, self.bbox.y_max),
                ],
                dtype=np.float32,
            )
            if not self.is_feasible_point(point):
                continue
            if self.distance_to_boundary(point) < (margin - 1e-9):
                continue
            return point

    def is_feasible_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=np.float32)
        in_feasible = any(rect.contains(point) for rect in self.feasible_rects)
        if not in_feasible:
            return False
        if any(rect.contains(point) for rect in self.blocked_rects):
            return False
        return True

    def distance_to_boundary(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float32)
        if not self.is_feasible_point(point):
            return 0.0
        return float(min(self._point_to_segment_distance(point, seg_start, seg_end) for seg_start, seg_end in self.boundary_segments))

    def segment_is_feasible(self, start: np.ndarray, end: np.ndarray, max_spacing: float = 0.01) -> bool:
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        seg_len = float(np.linalg.norm(end - start))
        n_steps = max(1, int(np.ceil(seg_len / max_spacing)))
        alphas = np.linspace(0.0, 1.0, num=n_steps + 1, dtype=np.float32)
        samples = start[None, :] + alphas[:, None] * (end - start)[None, :]
        return bool(np.all([self.is_feasible_point(sample) for sample in samples]))

    def _build_boundary_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        x_coords = sorted(
            {
                self.bbox.x_min,
                self.bbox.x_max,
                *[rect.x_min for rect in self.feasible_rects],
                *[rect.x_max for rect in self.feasible_rects],
                *[rect.x_min for rect in self.blocked_rects],
                *[rect.x_max for rect in self.blocked_rects],
            }
        )
        y_coords = sorted(
            {
                self.bbox.y_min,
                self.bbox.y_max,
                *[rect.y_min for rect in self.feasible_rects],
                *[rect.y_max for rect in self.feasible_rects],
                *[rect.y_min for rect in self.blocked_rects],
                *[rect.y_max for rect in self.blocked_rects],
            }
        )
        feasible_cells = {}
        for ix in range(len(x_coords) - 1):
            for iy in range(len(y_coords) - 1):
                cell = Rect(x_coords[ix], x_coords[ix + 1], y_coords[iy], y_coords[iy + 1])
                feasible_cells[(ix, iy)] = self.is_feasible_point(cell.center())

        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for ix in range(len(x_coords) - 1):
            for iy in range(len(y_coords) - 1):
                if not feasible_cells[(ix, iy)]:
                    continue

                x0, x1 = x_coords[ix], x_coords[ix + 1]
                y0, y1 = y_coords[iy], y_coords[iy + 1]

                neighbor_checks = [
                    ((ix - 1, iy), np.array([x0, y0], dtype=np.float32), np.array([x0, y1], dtype=np.float32)),
                    ((ix + 1, iy), np.array([x1, y0], dtype=np.float32), np.array([x1, y1], dtype=np.float32)),
                    ((ix, iy - 1), np.array([x0, y0], dtype=np.float32), np.array([x1, y0], dtype=np.float32)),
                    ((ix, iy + 1), np.array([x0, y1], dtype=np.float32), np.array([x1, y1], dtype=np.float32)),
                ]
                for neighbor_key, seg_start, seg_end in neighbor_checks:
                    if neighbor_key not in feasible_cells or not feasible_cells[neighbor_key]:
                        segments.append((seg_start, seg_end))
        if not segments:
            raise RuntimeError(f"Failed to derive boundary segments for map {self.map_type}")
        return segments

    def _point_to_segment_distance(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float32)
        seg_start = np.asarray(seg_start, dtype=np.float32)
        seg_end = np.asarray(seg_end, dtype=np.float32)
        vec = seg_end - seg_start
        denom = float(np.dot(vec, vec))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - seg_start))
        t = float(np.dot(point - seg_start, vec) / denom)
        t = min(1.0, max(0.0, t))
        proj = seg_start + t * vec
        return float(np.linalg.norm(point - proj))
