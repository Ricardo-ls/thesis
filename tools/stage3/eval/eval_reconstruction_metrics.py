from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import load_npz, save_json
from utils.stage3.paths import DATA_OUT_DIR, EVAL_OUT_DIR, ensure_stage3_dirs


def infer_method_name(pred_path: Path):
    stem = pred_path.stem
    if stem.endswith("_results"):
        return stem[: -len("_results")]
    return stem


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate minimal Stage 3 reconstruction metrics."
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=str(DATA_OUT_DIR / "missing_span_windows.npz"),
    )
    parser.add_argument("--target_key", type=str, default="traj_abs")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--pred_key", type=str, default="traj_hat")
    parser.add_argument("--method_name", type=str, default=None)
    args = parser.parse_args()

    ensure_stage3_dirs()

    target_data = load_npz(args.target_path)
    pred_data = load_npz(args.pred_path)
    if args.target_key not in target_data:
        raise KeyError(f"'{args.target_key}' not found in {args.target_path}")
    if args.pred_key not in pred_data:
        raise KeyError(f"'{args.pred_key}' not found in {args.pred_path}")

    target = np.asarray(target_data[args.target_key], dtype=np.float32)
    pred = np.asarray(pred_data[args.pred_key], dtype=np.float32)
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape} vs pred {pred.shape}")
    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError(f"Expected [N, T, 2] trajectories, got {target.shape}")

    diff = pred - target
    point_error = np.linalg.norm(diff, axis=-1)

    metrics = {
        "method_name": args.method_name or infer_method_name(Path(args.pred_path)),
        "num_samples": int(target.shape[0]),
        "sequence_length": int(target.shape[1]),
        "ADE": float(point_error.mean()),
        "FDE": float(point_error[:, -1].mean()),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
    }

    output_path = EVAL_OUT_DIR / f"{metrics['method_name']}_reconstruction_metrics.json"
    save_json(output_path, metrics)

    print("=" * 60)
    print("Reconstruction metrics")
    print(f"method_name   = {metrics['method_name']}")
    print(f"target_path   = {args.target_path}")
    print(f"pred_path     = {args.pred_path}")
    print(f"ADE           = {metrics['ADE']:.6f}")
    print(f"FDE           = {metrics['FDE']:.6f}")
    print(f"RMSE          = {metrics['RMSE']:.6f}")
    print(f"output_path   = {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
