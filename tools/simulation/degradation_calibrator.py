from __future__ import annotations

import numpy as np


class GlobalDegrader:
    def apply_batch(self, trajs, deg_type, params, seed_base=0):
        trajs = np.asarray(trajs, dtype=np.float32)
        if trajs.ndim != 3 or trajs.shape[-1] != 2:
            raise ValueError(f"Expected trajs shape [N, T, 2], got {trajs.shape}")

        deg_type = str(deg_type)
        seed_base = int(seed_base)
        degraded = np.zeros_like(trajs, dtype=np.float32)
        for i in range(trajs.shape[0]):
            degraded[i] = self._apply_one(
                traj=trajs[i],
                deg_type=deg_type,
                params=params,
                seed=seed_base + i,
            )
        return degraded.astype(np.float32)

    def _apply_one(self, traj, deg_type, params, seed):
        traj = np.asarray(traj, dtype=np.float32)
        rng = np.random.default_rng(seed)

        if deg_type == "gaussian":
            sigma = float(params["sigma"])
            noise = rng.normal(0.0, sigma, size=traj.shape).astype(np.float32)
            return (traj + noise).astype(np.float32)

        if deg_type == "bias":
            sigma = float(params["sigma"])
            bias = rng.normal(0.0, sigma, size=(1, 2)).astype(np.float32)
            return (traj + bias).astype(np.float32)

        if deg_type == "drift":
            sigma_step = float(params["sigma_step"])
            increments = rng.normal(0.0, sigma_step, size=(traj.shape[0] - 1, 2)).astype(np.float32)
            drift = np.zeros_like(traj, dtype=np.float32)
            drift[1:] = np.cumsum(increments, axis=0).astype(np.float32)
            return (traj + drift).astype(np.float32)

        if deg_type == "jump":
            sigma = float(params["sigma"])
            jump_idx = int(rng.integers(1, traj.shape[0] - 1))
            jump = rng.normal(0.0, sigma, size=(2,)).astype(np.float32)
            norm = float(np.linalg.norm(jump))
            if norm > 0.8:
                jump = jump * np.float32(0.8 / norm)
            elif 0.0 < norm < 0.2:
                jump = jump * np.float32(0.2 / norm)
            elif norm == 0.0:
                jump = np.array([0.2, 0.0], dtype=np.float32)

            out = traj.copy()
            out[jump_idx:] += jump[None, :]
            return out.astype(np.float32)

        if deg_type == "burst":
            sigma = float(params["sigma"])
            burst_start = int(rng.integers(1, traj.shape[0] - 3))
            noise = rng.normal(0.0, sigma, size=traj.shape).astype(np.float32)
            burst_noise = rng.normal(0.0, 8.0 * sigma, size=(3, 2)).astype(np.float32)
            noise[burst_start : burst_start + 3] = burst_noise
            return (traj + noise).astype(np.float32)

        if deg_type == "combined":
            sigma_g = float(params["sigma_g"])
            sigma_d = float(params["sigma_d"])
            sigma_b = float(params["sigma_b"])

            out = self._apply_one(traj, "bias", {"sigma": sigma_b}, seed=seed)
            out = self._apply_one(out, "drift", {"sigma_step": sigma_d}, seed=seed + 100000)
            out = self._apply_one(out, "gaussian", {"sigma": sigma_g}, seed=seed + 200000)
            return out.astype(np.float32)

        raise ValueError(
            f"Unsupported deg_type={deg_type}. "
            "Supported: gaussian, bias, drift, jump, burst, combined"
        )


def compute_ade(clean, degraded):
    clean = np.asarray(clean, dtype=np.float32)
    degraded = np.asarray(degraded, dtype=np.float32)
    if clean.shape != degraded.shape:
        raise ValueError(f"Shape mismatch: clean {clean.shape} vs degraded {degraded.shape}")
    if clean.ndim != 3 or clean.shape[-1] != 2:
        raise ValueError(f"Expected shape [N, T, 2], got {clean.shape}")

    err = np.linalg.norm(degraded - clean, axis=-1)
    return float(err.mean())
