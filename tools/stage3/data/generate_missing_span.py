from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import load_npz, save_npz
from utils.stage3.paths import DATA_OUT_DIR, ensure_stage3_dirs


def resolve_span_length(seq_len: int, span_ratio: float):
    if not 0.0 < span_ratio < 1.0:
        raise ValueError(f"span_ratio must be in (0, 1), got {span_ratio}")
    span_len = int(round(seq_len * span_ratio))
    span_len = max(1, span_len)
    span_len = min(span_len, seq_len - 2)
    if span_len <= 0:
        raise ValueError(
            f"Sequence length {seq_len} is too short for an interior missing span."
        )
    return span_len


def sample_spans(num_samples: int, seq_len: int, span_len: int, span_mode: str, seed: int):
    if seq_len < 3:
        raise ValueError(f"Sequence length must be at least 3, got {seq_len}")

    rng = np.random.default_rng(seed)
    span_start = np.zeros(num_samples, dtype=np.int64)
    span_end = np.zeros(num_samples, dtype=np.int64)

    max_start = seq_len - span_len - 1
    if max_start < 1:
        raise ValueError(
            f"Cannot place an interior missing span of length {span_len} in sequence length {seq_len}"
        )

    if span_mode == "fixed":
        start = (seq_len - span_len) // 2
        end = start + span_len - 1
        span_start.fill(start)
        span_end.fill(end)
        return span_start, span_end

    for i in range(num_samples):
        start = rng.integers(low=1, high=max_start + 1)
        end = start + span_len - 1
        span_start[i] = start
        span_end[i] = end

    return span_start, span_end


def main():
    parser = argparse.ArgumentParser(
        description="Generate the minimal Stage 3 degraded dataset with one contiguous missing span."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(DATA_OUT_DIR / "clean_windows.npz"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DATA_OUT_DIR / "missing_span_windows.npz"),
    )
    parser.add_argument("--span_ratio", type=float, default=0.2)
    parser.add_argument("--span_mode", type=str, default="fixed", choices=["random", "fixed"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_stage3_dirs()

    data = load_npz(args.input_path)
    if "traj_abs" not in data:
        raise KeyError(f"'traj_abs' not found in {args.input_path}")

    traj_abs = np.asarray(data["traj_abs"], dtype=np.float32)
    if traj_abs.ndim != 3 or traj_abs.shape[-1] != 2:
        raise ValueError(
            f"Expected traj_abs with shape [N, T, 2], got {traj_abs.shape}"
        )

    num_samples, seq_len, _ = traj_abs.shape
    span_len = resolve_span_length(seq_len=seq_len, span_ratio=args.span_ratio)
    span_start, span_end = sample_spans(
        num_samples=num_samples,
        seq_len=seq_len,
        span_len=span_len,
        span_mode=args.span_mode,
        seed=args.seed,
    )

    obs_mask = np.ones((num_samples, seq_len), dtype=np.uint8)
    traj_obs = traj_abs.copy()

    for i in range(num_samples):
        start = int(span_start[i])
        end = int(span_end[i])
        obs_mask[i, start : end + 1] = 0
        traj_obs[i, start : end + 1, :] = np.nan

    output_path = Path(args.output_path)
    save_npz(
        output_path,
        traj_abs=traj_abs,
        traj_obs=traj_obs,
        obs_mask=obs_mask,
        span_start=span_start,
        span_end=span_end,
    )

    missing_lengths = span_end - span_start + 1
    total_points = num_samples * seq_len
    total_missing = int((obs_mask == 0).sum())

    print("=" * 60)
    print("Stage 3 missing-span dataset built")
    print(f"input_path           = {args.input_path}")
    print(f"output_path          = {output_path}")
    print(f"N                    = {num_samples}")
    print(f"T                    = {seq_len}")
    print(f"span_ratio           = {args.span_ratio}")
    print(f"span_mode            = {args.span_mode}")
    print(f"seed                 = {args.seed}")
    print(f"average_missing_len  = {missing_lengths.mean():.4f}")
    print(f"missing_ratio        = {total_missing / total_points:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
