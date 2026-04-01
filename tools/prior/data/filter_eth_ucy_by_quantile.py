import argparse
from pathlib import Path

import numpy as np


def compute_avg_speed(rel_data: np.ndarray) -> np.ndarray:
    """
    rel_data: [N, 19, 2]
    return: [N]
    """
    if rel_data.ndim != 3 or rel_data.shape[-1] != 2:
        raise ValueError(f"Expected rel_data shape [N, T, 2], got {rel_data.shape}")
    step_norm = np.linalg.norm(rel_data, axis=-1)   # [N, 19]
    avg_speed = step_norm.mean(axis=1)              # [N]
    return avg_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantile", type=int, choices=[10, 20, 30], required=True)
    parser.add_argument(
        "--abs_path",
        type=str,
        default="datasets/processed/data_eth_ucy_20.npy"
    )
    parser.add_argument(
        "--rel_path",
        type=str,
        default="datasets/processed/data_eth_ucy_20_rel.npy"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="datasets/processed"
    )
    args = parser.parse_args()

    abs_path = Path(args.abs_path)
    rel_path = Path(args.rel_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_data = np.load(abs_path)
    rel_data = np.load(rel_path)

    if len(abs_data) != len(rel_data):
        raise ValueError(
            f"abs/rel sample count mismatch: {len(abs_data)} vs {len(rel_data)}"
        )

    avg_speed = compute_avg_speed(rel_data)
    threshold = np.quantile(avg_speed, args.quantile / 100.0)
    keep_mask = avg_speed >= threshold

    abs_keep = abs_data[keep_mask]
    rel_keep = rel_data[keep_mask]

    suffix = f"q{args.quantile}"
    abs_out = out_dir / f"data_eth_ucy_20_{suffix}.npy"
    rel_out = out_dir / f"data_eth_ucy_20_rel_{suffix}.npy"

    np.save(abs_out, abs_keep)
    np.save(rel_out, rel_keep)

    print("=== ETH+UCY quantile filtering ===")
    print(f"quantile       : q{args.quantile}")
    print(f"threshold      : {threshold:.8f}")
    print(f"before         : {len(abs_data)}")
    print(f"after          : {len(abs_keep)}")
    print(f"kept_ratio     : {len(abs_keep) / len(abs_data):.6f}")
    print(f"avg_speed_mean : {avg_speed.mean():.8f} -> {compute_avg_speed(rel_keep).mean():.8f}")

    if rel_keep.shape[0] > 0:
        step_norm_before = np.linalg.norm(rel_data, axis=-1)
        step_norm_after = np.linalg.norm(rel_keep, axis=-1)
        acc_before = np.diff(rel_data, axis=1)
        acc_after = np.diff(rel_keep, axis=1)

        acc_rms_before = np.sqrt((acc_before ** 2).sum(axis=-1)).mean()
        acc_rms_after = np.sqrt((acc_after ** 2).sum(axis=-1)).mean()

        print(f"acc_rms_mean   : {acc_rms_before:.8f} -> {acc_rms_after:.8f}")
        print(f"saved abs      : {abs_out}")
        print(f"saved rel      : {rel_out}")


if __name__ == "__main__":
    main()