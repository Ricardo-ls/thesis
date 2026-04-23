from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import load_npz, save_json
from utils.stage3.paths import (
    DEFAULT_EXPERIMENT_ID,
    LINEAR_METHOD_TAG,
    baseline_results_path,
    ensure_stage3_dirs,
    missing_span_path,
    reconstruction_metrics_path,
)


def infer_method_name(pred_path: Path):
    if pred_path.name == "results.npz":
        return pred_path.parent.name
    stem = pred_path.stem
    if stem.endswith("_results"):
        return stem[: -len("_results")]
    return stem


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate minimal Stage 3 reconstruction metrics."
    )
    parser.add_argument("--experiment_id", type=str, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--method_tag", type=str, default=None)
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
    )
    parser.add_argument("--target_key", type=str, default="traj_abs")
    parser.add_argument("--pred_path", type=str, default=None)
    parser.add_argument("--pred_key", type=str, default="traj_hat")
    parser.add_argument("--method_name", type=str, default=None)
    args = parser.parse_args()

    ensure_stage3_dirs()
    method_tag = args.method_tag
    if method_tag is None:
        method_tag = infer_method_name(Path(args.pred_path)) if args.pred_path else LINEAR_METHOD_TAG
    target_path = Path(args.target_path) if args.target_path else missing_span_path(args.experiment_id)
    pred_path = Path(args.pred_path) if args.pred_path else baseline_results_path(args.experiment_id, method_tag)
    method_name = args.method_name or method_tag

    target_data = load_npz(target_path)
    pred_data = load_npz(pred_path)
    if args.target_key not in target_data:
        raise KeyError(f"'{args.target_key}' not found in {target_path}")
    if args.pred_key not in pred_data:
        raise KeyError(f"'{args.pred_key}' not found in {pred_path}")

    target = np.asarray(target_data[args.target_key], dtype=np.float32)
    pred = np.asarray(pred_data[args.pred_key], dtype=np.float32)
    if "obs_mask" not in target_data:
        raise KeyError(f"'obs_mask' not found in {target_path}")
    obs_mask = np.asarray(target_data["obs_mask"], dtype=np.uint8)
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape} vs pred {pred.shape}")
    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError(f"Expected [N, T, 2] trajectories, got {target.shape}")
    if obs_mask.shape != target.shape[:2]:
        raise ValueError(
            f"obs_mask shape {obs_mask.shape} does not match target shape {target.shape[:2]}"
        )

    diff = pred - target
    point_error = np.linalg.norm(diff, axis=-1)
    missing_mask = obs_mask == 0
    if np.any(missing_mask):
        masked_point_error = point_error[missing_mask]
        masked_diff = diff[missing_mask]
        masked_ade = float(masked_point_error.mean())
        masked_rmse = float(np.sqrt(np.mean(masked_diff ** 2)))
    else:
        masked_ade = 0.0
        masked_rmse = 0.0

    metrics = {
        "method_name": method_name,
        "experiment_id": args.experiment_id,
        "num_samples": int(target.shape[0]),
        "sequence_length": int(target.shape[1]),
        "ADE": float(point_error.mean()),
        "FDE": float(point_error[:, -1].mean()),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "masked_ADE": masked_ade,
        "masked_RMSE": masked_rmse,
    }

    output_path = reconstruction_metrics_path(args.experiment_id, method_tag)
    save_json(output_path, metrics)

    print("=" * 60)
    print("Reconstruction metrics")
    print(f"experiment_id = {args.experiment_id}")
    print(f"method_name   = {metrics['method_name']}")
    print(f"target_path   = {target_path}")
    print(f"pred_path     = {pred_path}")
    print(f"ADE           = {metrics['ADE']:.6f}")
    print(f"FDE           = {metrics['FDE']:.6f}")
    print(f"RMSE          = {metrics['RMSE']:.6f}")
    print(f"masked_ADE    = {metrics['masked_ADE']:.6f}")
    print(f"masked_RMSE   = {metrics['masked_RMSE']:.6f}")
    print(f"output_path   = {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
