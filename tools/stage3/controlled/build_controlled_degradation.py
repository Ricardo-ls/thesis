from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3.data.build_imputation_dataset import (
    load_absolute_windows,
    normalize_to_room3,
    validate_window_array,
)
from utils.stage3.controlled_benchmark import (
    DEFAULT_DRIFT_AMP,
    DEFAULT_NOISE_STD,
    DEFAULT_SEED,
    DEFAULT_SPAN_MODE,
    DEFAULT_SPAN_RATIO,
    DEGRADATION_NAMES,
    clean_path,
    degraded_path,
    degradation_metadata_path,
    ensure_controlled_dirs,
    experiment_tag,
    mask_path,
)
from utils.stage3.io import save_json
from utils.stage3.paths import CLEAN_ABS_INPUT_PATH


def resolve_span_length(seq_len: int, span_ratio: float):
    if not 0.0 < span_ratio < 1.0:
        raise ValueError(f"span_ratio must be in (0, 1), got {span_ratio}")
    span_len = int(round(seq_len * span_ratio))
    span_len = max(1, span_len)
    span_len = min(span_len, seq_len - 2)
    if span_len <= 0:
        raise ValueError(f"Sequence length {seq_len} is too short for span_ratio={span_ratio}")
    return span_len


def sample_spans(num_samples: int, seq_len: int, span_len: int, span_mode: str, seed: int):
    rng = np.random.default_rng(seed)
    span_start = np.zeros(num_samples, dtype=np.int64)
    span_end = np.zeros(num_samples, dtype=np.int64)

    max_start = seq_len - span_len - 1
    if max_start < 1:
        raise ValueError(
            f"Cannot place an interior span of length {span_len} in sequence length {seq_len}"
        )

    if span_mode == "fixed":
        start = (seq_len - span_len) // 2
        end = start + span_len - 1
        span_start.fill(start)
        span_end.fill(end)
        return span_start, span_end

    if span_mode != "random":
        raise ValueError(f"Unsupported span_mode: {span_mode}")

    for i in range(num_samples):
        start = rng.integers(low=1, high=max_start + 1)
        end = start + span_len - 1
        span_start[i] = start
        span_end[i] = end

    return span_start, span_end


def build_obs_mask(num_samples: int, seq_len: int, span_start: np.ndarray, span_end: np.ndarray):
    obs_mask = np.ones((num_samples, seq_len), dtype=np.uint8)
    for i in range(num_samples):
        obs_mask[i, span_start[i] : span_end[i] + 1] = 0
    return obs_mask


def build_noise(num_samples: int, seq_len: int, noise_std: float, seed: int):
    rng = np.random.default_rng(seed + 101)
    return rng.normal(loc=0.0, scale=noise_std, size=(num_samples, seq_len, 2)).astype(np.float32)


def build_smooth_drift(num_samples: int, seq_len: int, drift_amp: float, seed: int):
    rng = np.random.default_rng(seed + 202)
    alpha = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)[None, :, None]
    start = rng.uniform(-drift_amp, drift_amp, size=(num_samples, 1, 2)).astype(np.float32)
    end = rng.uniform(-drift_amp, drift_amp, size=(num_samples, 1, 2)).astype(np.float32)
    return (1.0 - alpha) * start + alpha * end


def apply_degradation(clean: np.ndarray, obs_mask: np.ndarray, noise: np.ndarray | None, drift: np.ndarray | None):
    degraded = clean.copy()
    observed = np.repeat((obs_mask[:, :, None] == 1), repeats=clean.shape[-1], axis=2)
    if noise is not None:
        degraded[observed] += noise[observed]
    if drift is not None:
        degraded[observed] += drift[observed]
    degraded[~observed] = np.nan
    return degraded.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build controlled Stage 3 coarse reconstruction degradations."
    )
    parser.add_argument("--input_path", type=str, default=str(CLEAN_ABS_INPUT_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--span_ratio", type=float, default=DEFAULT_SPAN_RATIO)
    parser.add_argument("--span_mode", type=str, default=DEFAULT_SPAN_MODE, choices=["fixed", "random"])
    parser.add_argument("--noise_std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--drift_amp", type=float, default=DEFAULT_DRIFT_AMP)
    args = parser.parse_args()

    ensure_controlled_dirs()
    tag = experiment_tag(args.span_ratio, args.span_mode, args.seed)
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    raw = load_absolute_windows(input_path)
    clean = normalize_to_room3(validate_window_array(raw))
    num_samples, seq_len, _ = clean.shape

    span_len = resolve_span_length(seq_len, args.span_ratio)
    span_start, span_end = sample_spans(num_samples, seq_len, span_len, args.span_mode, args.seed)
    obs_mask = build_obs_mask(num_samples, seq_len, span_start, span_end)

    noise = build_noise(num_samples, seq_len, args.noise_std, args.seed)
    drift = build_smooth_drift(num_samples, seq_len, args.drift_amp, args.seed)

    degradation_arrays = {
        "missing_only": apply_degradation(clean, obs_mask, noise=None, drift=None),
        "missing_noise": apply_degradation(clean, obs_mask, noise=noise, drift=None),
        "missing_drift": apply_degradation(clean, obs_mask, noise=None, drift=drift),
        "missing_noise_drift": apply_degradation(clean, obs_mask, noise=noise, drift=drift),
    }

    np.save(clean_path(), clean)
    np.save(mask_path(tag), obs_mask)
    for degradation_name in DEGRADATION_NAMES:
        np.save(degraded_path(degradation_name, tag), degradation_arrays[degradation_name])

    metadata = {
        "source_dataset": str(input_path),
        "seed": args.seed,
        "span_ratio": args.span_ratio,
        "span_mode": args.span_mode,
        "noise_std": args.noise_std,
        "drift_amp": args.drift_amp,
        "experiment_tag": tag,
        "num_samples": int(num_samples),
        "sequence_length": int(seq_len),
        "output_shapes": {
            "clean": list(clean.shape),
            "mask": list(obs_mask.shape),
            **{name: list(array.shape) for name, array in degradation_arrays.items()},
        },
    }
    save_json(degradation_metadata_path(tag), metadata)

    print("=" * 60)
    print("Controlled Stage 3 degradations generated")
    print(f"input_path      = {input_path}")
    print(f"experiment_tag  = {tag}")
    print(f"clean_path      = {clean_path()}")
    print(f"mask_path       = {mask_path(tag)}")
    for degradation_name in DEGRADATION_NAMES:
        print(f"{degradation_name:16s}= {degraded_path(degradation_name, tag)}")
    print(f"metadata_path   = {degradation_metadata_path(tag)}")
    print(f"clean_shape     = {clean.shape}")
    print(f"mask_shape      = {obs_mask.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
