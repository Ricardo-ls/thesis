import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
CONFIG_PATH = DATA_DIR / "degradation_config.json"
FIGURE_PATH = DATA_DIR / "degradation_examples.png"

EXPERIMENT_N = 1000
T = 20
BASE_SEED = 42

DEGRADATION_ORDER = [
    "gaussian_medium",
    "bias_medium",
    "drift_medium",
    "jump_medium",
    "burst_medium",
    "combined_medium",
]


def compute_per_traj_ade(clean: np.ndarray, degraded: np.ndarray) -> np.ndarray:
    errors = np.linalg.norm(degraded - clean, axis=-1)
    return errors.mean(axis=1).astype(np.float32)


def compute_reported_ade(clean: np.ndarray, degraded: np.ndarray) -> float:
    return float(compute_per_traj_ade(clean, degraded).mean())


def apply_gaussian(clean: np.ndarray, sigma: float) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for i in range(clean.shape[0]):
        rng = np.random.default_rng(BASE_SEED + i)
        noise = rng.normal(0.0, sigma, size=clean.shape[1:]).astype(np.float32)
        degraded[i] = clean[i] + noise
    return degraded


def apply_bias(clean: np.ndarray, sigma: float) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for i in range(clean.shape[0]):
        rng = np.random.default_rng(BASE_SEED + i)
        bias = rng.normal(0.0, sigma, size=(2,)).astype(np.float32)
        degraded[i] = clean[i] + bias
    return degraded


def apply_drift(clean: np.ndarray, sigma_step: float) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for i in range(clean.shape[0]):
        rng = np.random.default_rng(BASE_SEED + i)
        increments = rng.normal(0.0, sigma_step, size=(T - 1, 2)).astype(np.float32)
        drift = np.zeros((T, 2), dtype=np.float32)
        drift[1:] = np.cumsum(increments, axis=0)
        degraded[i] = clean[i] + drift
    return degraded


def apply_jump(clean: np.ndarray) -> np.ndarray:
    degraded = clean.copy().astype(np.float32)
    jump_frame_candidates = np.arange(1, T - 1)
    for i in range(clean.shape[0]):
        rng = np.random.default_rng(BASE_SEED + i)
        n_jump = int(rng.integers(2, 5))
        jump_frames = np.sort(rng.choice(jump_frame_candidates, size=n_jump, replace=False))
        segment_ends = list(jump_frames[1:]) + [T]
        for jump_frame, segment_end in zip(jump_frames, segment_ends):
            magnitude = float(rng.uniform(0.2, 0.5))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            offset = np.array(
                [magnitude * np.cos(angle), magnitude * np.sin(angle)],
                dtype=np.float32,
            )
            degraded[i, jump_frame:segment_end] = clean[i, jump_frame:segment_end] + offset
    return degraded


def apply_burst(clean: np.ndarray, burst_sigma: float, background_sigma: float) -> np.ndarray:
    degraded = np.empty_like(clean, dtype=np.float32)
    for i in range(clean.shape[0]):
        rng = np.random.default_rng(BASE_SEED + i)
        noise = rng.normal(0.0, background_sigma, size=(T, 2)).astype(np.float32)
        burst_len = int(rng.integers(3, 6))
        burst_start = int(rng.integers(0, T - burst_len + 1))
        burst_end = burst_start + burst_len
        noise[burst_start:burst_end] = rng.normal(
            0.0, burst_sigma, size=(burst_len, 2)
        ).astype(np.float32)
        degraded[i] = clean[i] + noise
    return degraded


def apply_combined(clean: np.ndarray, sigma_g: float, sigma_b: float, sigma_d: float) -> np.ndarray:
    degraded = clean.copy().astype(np.float32)
    for i in range(clean.shape[0]):
        bias_rng = np.random.default_rng(42000 + i)
        drift_rng = np.random.default_rng(84000 + i)
        gaussian_rng = np.random.default_rng(42 + i)

        bias = bias_rng.normal(0.0, sigma_b, size=(2,)).astype(np.float32)
        degraded[i] = degraded[i] + bias

        increments = drift_rng.normal(0.0, sigma_d, size=(T - 1, 2)).astype(np.float32)
        drift = np.zeros((T, 2), dtype=np.float32)
        drift[1:] = np.cumsum(increments, axis=0)
        degraded[i] = degraded[i] + drift

        noise = gaussian_rng.normal(0.0, sigma_g, size=(T, 2)).astype(np.float32)
        degraded[i] = degraded[i] + noise
    return degraded


def save_array(path: Path, array: np.ndarray) -> None:
    np.save(path, array.astype(np.float32))
    print(f"Saved: {path}")


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


def plot_examples(clean: np.ndarray, degraded_map: dict[str, np.ndarray], ade_map: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(2026)
    example_indices = rng.choice(clean.shape[0], size=3, replace=False)

    fig, axes = plt.subplots(6, 3, figsize=(12, 20), constrained_layout=True)
    for row, deg_name in enumerate(DEGRADATION_ORDER):
        degraded = degraded_map[deg_name]
        per_traj_ade = ade_map[deg_name]
        for col, traj_idx in enumerate(example_indices):
            ax = axes[row, col]
            ax.plot(clean[traj_idx, :, 0], clean[traj_idx, :, 1], color="blue", linewidth=1.8)
            ax.plot(degraded[traj_idx, :, 0], degraded[traj_idx, :, 1], color="red", linewidth=1.8)
            ax.scatter(
                clean[traj_idx, 0, 0],
                clean[traj_idx, 0, 1],
                color="green",
                s=40,
                marker="o",
                zorder=3,
            )
            ax.scatter(
                degraded[traj_idx, 0, 0],
                degraded[traj_idx, 0, 1],
                color="red",
                s=45,
                marker="s",
                zorder=3,
            )
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 3.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                f"{deg_name}\nidx={traj_idx}  ADE={per_traj_ade[traj_idx]:.3f}",
                fontsize=10,
            )
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved: {FIGURE_PATH}")


def main() -> None:
    if not CLEAN_PATH.is_file():
        raise FileNotFoundError(f"Missing required input: {CLEAN_PATH}")

    clean_all = np.load(CLEAN_PATH)
    if clean_all.shape != (2000, 20, 2):
        raise ValueError(f"Expected clean_trajs shape (2000, 20, 2), got {clean_all.shape}")

    clean = clean_all[:EXPERIMENT_N].astype(np.float32)

    degraded_map = {}
    ade_map = {}

    gaussian = apply_gaussian(clean, sigma=0.05)
    degraded_map["gaussian_medium"] = gaussian
    ade_map["gaussian_medium"] = compute_per_traj_ade(clean, gaussian)
    save_array(DATA_DIR / "degraded_gaussian_medium.npy", gaussian)

    bias = apply_bias(clean, sigma=0.15)
    degraded_map["bias_medium"] = bias
    ade_map["bias_medium"] = compute_per_traj_ade(clean, bias)
    save_array(DATA_DIR / "degraded_bias_medium.npy", bias)

    drift = apply_drift(clean, sigma_step=0.010)
    degraded_map["drift_medium"] = drift
    ade_map["drift_medium"] = compute_per_traj_ade(clean, drift)
    save_array(DATA_DIR / "degraded_drift_medium.npy", drift)

    jump = apply_jump(clean)
    degraded_map["jump_medium"] = jump
    ade_map["jump_medium"] = compute_per_traj_ade(clean, jump)
    save_array(DATA_DIR / "degraded_jump_medium.npy", jump)

    burst = apply_burst(clean, burst_sigma=0.25, background_sigma=0.01)
    degraded_map["burst_medium"] = burst
    ade_map["burst_medium"] = compute_per_traj_ade(clean, burst)
    save_array(DATA_DIR / "degraded_burst_medium.npy", burst)

    combined = apply_combined(clean, sigma_g=0.05, sigma_b=0.15, sigma_d=0.010)
    degraded_map["combined_medium"] = combined
    ade_map["combined_medium"] = compute_per_traj_ade(clean, combined)
    save_array(DATA_DIR / "degraded_combined_medium.npy", combined)

    config = {
        "input_clean_path": str(CLEAN_PATH.relative_to(PROJECT_ROOT)),
        "experiment_subset": {
            "slice": "clean_trajs[:1000]",
            "shape": [1000, 20, 2],
            "dtype": "float32",
        },
        "gaussian_medium": {"sigma": 0.05},
        "bias_medium": {"sigma": 0.15},
        "drift_medium": {"sigma_step": 0.010, "drift_frame0_zero": True},
        "jump_medium": {
            "n_jump_range_inclusive": [2, 4],
            "jump_frame_range_inclusive": [1, T - 2],
            "magnitude_range_m": [0.2, 0.5],
            "offset_mode": "piecewise_constant_non_cumulative",
        },
        "burst_medium": {
            "burst_length_range_inclusive": [3, 5],
            "burst_sigma": 0.25,
            "background_sigma": 0.01,
        },
        "combined_medium": {
            "composition_order": ["bias_medium", "drift_medium", "gaussian_medium"],
            "independent_seeds": {
                "gaussian_seed": "42 + i",
                "bias_seed": "42000 + i",
                "drift_seed": "84000 + i",
            },
            "parameters": {
                "gaussian_sigma": 0.05,
                "bias_sigma": 0.15,
                "drift_sigma_step": 0.010,
            },
        },
        "per_trajectory_seed_rule": {
            "default": "42 + i",
            "combined": {
                "gaussian": "42 + i",
                "bias": "42000 + i",
                "drift": "84000 + i",
            },
        },
        "postprocess": {"clip_applied": False},
    }
    save_json(CONFIG_PATH, config)

    plot_examples(clean, degraded_map, ade_map)

    print("=== Step 2 验证输出 ===")
    for deg_name in DEGRADATION_ORDER:
        degraded = degraded_map[deg_name]
        ade = compute_reported_ade(clean, degraded)
        xmin, xmax = float(degraded[..., 0].min()), float(degraded[..., 0].max())
        ymin, ymax = float(degraded[..., 1].min()), float(degraded[..., 1].max())
        print(f"{deg_name:20s}  ADE={ade:.4f}  x=[{xmin:.2f},{xmax:.2f}]  y=[{ymin:.2f},{ymax:.2f}]")

    print("所有退化文件已保存至 data/stage3_indoor/")
    print("degradation_examples.png 已保存")
    print("=== 等待人工确认 ===")


if __name__ == "__main__":
    main()
