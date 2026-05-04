from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from degradation_calibrator import GlobalDegrader, compute_ade


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "simulated"
TEST_TRAJS_PATH = DATA_DIR / "test_trajs.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
OUTPUT_PATH = DATA_DIR / "degradation_params.json"

SEED_BASE = 0
CALIB_N = 500

TARGETS = {
    "low": {"target": 0.05, "tol_low": 0.035, "tol_high": 0.065},
    "medium": {"target": 0.15, "tol_low": 0.125, "tol_high": 0.175},
    "high": {"target": 0.30, "tol_low": 0.260, "tol_high": 0.340},
}

CALIBRATION_ORDER = [
    ("gaussian_low", "gaussian", "low"),
    ("gaussian_medium", "gaussian", "medium"),
    ("gaussian_high", "gaussian", "high"),
    ("bias_low", "bias", "low"),
    ("bias_medium", "bias", "medium"),
    ("bias_high", "bias", "high"),
    ("drift_low", "drift", "low"),
    ("drift_medium", "drift", "medium"),
    ("drift_high", "drift", "high"),
    ("jump_medium", "jump", "medium"),
    ("burst_medium", "burst", "medium"),
]


def load_inputs():
    if not TEST_TRAJS_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {TEST_TRAJS_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {METADATA_PATH}")

    clean = np.load(TEST_TRAJS_PATH, allow_pickle=False).astype(np.float32)
    if clean.shape != (5000, 20, 2):
        raise RuntimeError(f"Expected clean shape (5000, 20, 2), got {clean.shape}")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if "mean_step_size" not in metadata:
        raise KeyError(f"mean_step_size missing from {METADATA_PATH}")

    return clean, metadata


def make_params(deg_type: str, mid: float):
    if deg_type == "drift":
        return {"sigma_step": float(mid)}
    return {"sigma": float(mid)}


def run_bisection(degrader: GlobalDegrader, clean_calib: np.ndarray, deg_type: str, target_key: str):
    cfg = TARGETS[target_key]
    lo = 1e-5
    hi = 2.0
    actual_ade = None
    best_mid = None

    for _ in range(40):
        mid = (lo + hi) / 2.0
        params = make_params(deg_type, mid)
        degraded = degrader.apply_batch(clean_calib, deg_type, params, seed_base=SEED_BASE)
        actual_ade = compute_ade(clean_calib, degraded)
        best_mid = mid
        if abs(actual_ade - cfg["target"]) < 0.005:
            break
        if actual_ade < cfg["target"]:
            lo = mid
        else:
            hi = mid

    if best_mid is None or actual_ade is None:
        raise RuntimeError(f"Bisection failed for {deg_type}/{target_key}")

    in_tolerance = bool(cfg["tol_low"] <= actual_ade <= cfg["tol_high"])
    if not in_tolerance:
        print(
            f"warning: {deg_type}_{target_key} actual_ade={actual_ade:.4f} "
            f"outside tolerance [{cfg['tol_low']:.3f}, {cfg['tol_high']:.3f}]"
        )

    result = {
        "actual_ade": float(actual_ade),
        "target_ade": float(cfg["target"]),
        "tolerance_low": float(cfg["tol_low"]),
        "tolerance_high": float(cfg["tol_high"]),
        "in_tolerance": bool(in_tolerance),
        "out_of_tolerance": bool(not in_tolerance),
    }
    if deg_type == "drift":
        result["sigma_step"] = float(best_mid)
    else:
        result["sigma"] = float(best_mid)
    return result


def build_combined_medium(degrader: GlobalDegrader, clean_calib: np.ndarray, params: dict):
    sigma_g = float(params["gaussian_medium"]["sigma"])
    sigma_d = float(params["drift_medium"]["sigma_step"])
    sigma_b = float(params["bias_medium"]["sigma"])
    combined_params = {
        "sigma_g": sigma_g,
        "sigma_d": sigma_d,
        "sigma_b": sigma_b,
    }
    degraded = degrader.apply_batch(clean_calib, "combined", combined_params, seed_base=SEED_BASE)
    actual_ade = compute_ade(clean_calib, degraded)
    return {
        "sigma_g": sigma_g,
        "sigma_d": sigma_d,
        "sigma_b": sigma_b,
        "actual_ade": float(actual_ade),
        "target_ade": None,
        "in_tolerance": None,
        "diagnostic_only": True,
    }


def save_json(path: Path, payload: dict):
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to save {path}: {exc}") from exc
    print(f"Saved: {path}")


def main():
    clean, metadata = load_inputs()
    clean_calib = clean[:CALIB_N]
    degrader = GlobalDegrader()

    params = {
        "calibration_note": (
            f"mean_step_size={float(metadata['mean_step_size']):.6f}m at 3Hz. "
            "Parameters are calibrated on the first 500 simulated test trajectories "
            "to target full-trajectory ADE levels."
        )
    }

    for key, deg_type, target_key in CALIBRATION_ORDER:
        params[key] = run_bisection(degrader, clean_calib, deg_type, target_key)

    params["combined_medium"] = build_combined_medium(degrader, clean_calib, params)

    save_json(OUTPUT_PATH, params)

    print("=== 指令2 验证输出 ===")
    print(f"{'退化类型':<25} {'actual_ade':>10} {'target':>8} {'容差':>17} {'通过':>10}")
    print("-" * 78)

    for key, val in params.items():
        if key == "calibration_note":
            continue

        actual = float(val["actual_ade"])
        target = val.get("target_ade", None)

        if val.get("diagnostic_only", False):
            status = "诊断项"
            target_str = "-"
            tol_str = "-"
        else:
            status = "✅" if val.get("in_tolerance") else "⚠️ 超出"
            target_str = f"{float(target):.3f}"
            tol_str = f"[{float(val['tolerance_low']):.3f},{float(val['tolerance_high']):.3f}]"

        print(f"{key:<25} {actual:>10.4f} {target_str:>8} {tol_str:>17} {status:>10}")

    print()

    all_pass = all(
        v.get("in_tolerance", True)
        for k, v in params.items()
        if k != "calibration_note" and not v.get("diagnostic_only", False)
    )

    print(f"全部通过：{'是 ✅' if all_pass else '否 ⚠️ 请检查超出容差的条目'}")
    print("=== 等待人工确认 ===")


if __name__ == "__main__":
    main()
