from __future__ import annotations

from pathlib import Path
import json
import math
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


DATA_DIR = PROJECT_ROOT / "data" / "stage3_indoor"
CLEAN_PATH = DATA_DIR / "clean_trajs.npy"
FORMAL_E1_DIR = PROJECT_ROOT / "outputs" / "stage4" / "e1_oracle_residual_gating_6conditions"
PROTOCOL_PATH = FORMAL_E1_DIR / "e1_protocol_validation_summary.csv"
FULL_METRICS_PATH = FORMAL_E1_DIR / "e1_full_metrics.csv"
BIN_METRICS_PATH = FORMAL_E1_DIR / "e1_confidence_bin_metrics.csv"
PER_TRAJ_PATH = FORMAL_E1_DIR / "e1_per_trajectory_metrics.csv"

OUT_DIR = FORMAL_E1_DIR / "confidence_cache"
SUMMARY_CSV_PATH = OUT_DIR / "confidence_cache_validation_summary.csv"
SUMMARY_MD_PATH = OUT_DIR / "confidence_cache_validation_summary.md"

DEGRADATION_ORDER = [
    "gaussian_medium",
    "drift_medium",
    "burst_medium",
    "bias_medium",
    "jump_medium",
    "combined_medium",
]
ATOL = 1e-8


def require_inputs() -> None:
    missing = [
        path
        for path in [CLEAN_PATH, PROTOCOL_PATH, FULL_METRICS_PATH, BIN_METRICS_PATH, PER_TRAJ_PATH]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required Formal E1 inputs:\n" + "\n".join(str(path) for path in missing))


def frame_error(pred: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - clean, axis=-1).astype(np.float32)


def confidence_from_delta(degraded: np.ndarray, clean: np.ndarray, delta0: float) -> tuple[np.ndarray, np.ndarray]:
    delta = frame_error(degraded, clean)
    confidence = np.exp(-delta / delta0).astype(np.float32)
    return confidence, delta


def load_arrays(protocol_df: pd.DataFrame, degradation: str) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = protocol_df[protocol_df["degradation"] == degradation]
    if rows.empty:
        raise RuntimeError(f"Missing protocol row for {degradation}")
    row = rows.iloc[0]
    degraded_path = Path(str(row["degraded_path"]))
    if not degraded_path.is_file():
        raise FileNotFoundError(f"Missing degraded input for {degradation}: {degraded_path}")
    degraded = np.load(degraded_path).astype(np.float32)
    clean_all = np.load(CLEAN_PATH).astype(np.float32)
    clean = clean_all[: degraded.shape[0]].astype(np.float32)
    if clean.shape != degraded.shape:
        raise ValueError(f"Shape mismatch for {degradation}: clean={clean.shape}, degraded={degraded.shape}")
    return clean, degraded, row.to_dict()


def get_delta0(full_df: pd.DataFrame, degradation: str) -> float:
    rows = full_df[(full_df["degradation"] == degradation) & (full_df["method"] == "e1_oracle_gated_residual")]
    if rows.empty:
        rows = full_df[full_df["degradation"] == degradation]
    if rows.empty:
        raise RuntimeError(f"Missing Formal E1 delta0 row for {degradation}")
    values = rows["delta0_auto"].dropna().unique()
    if len(values) != 1:
        raise RuntimeError(f"Ambiguous delta0_auto values for {degradation}: {values}")
    return float(values[0])


def expected_bin_counts(bin_df: pd.DataFrame, degradation: str) -> dict[str, int]:
    rows = bin_df[(bin_df["degradation"] == degradation) & (bin_df["method"] == "noisy_input")]
    out: dict[str, int] = {}
    for bin_name in ["high", "mid", "low"]:
        match = rows[rows["confidence_bin"] == bin_name]
        if not match.empty:
            out[bin_name] = int(match.iloc[0]["N_frames"])
    return out


def validate_per_traj_stats(confidence: np.ndarray, per_df: pd.DataFrame, degradation: str) -> dict:
    rows = per_df[per_df["degradation"] == degradation].sort_values("trajectory_id")
    if rows.empty:
        return {
            "per_traj_stats_available": False,
            "per_traj_min_match": False,
            "per_traj_mean_match": False,
            "per_traj_max_match": False,
            "per_traj_max_abs_diff_min": math.nan,
            "per_traj_max_abs_diff_mean": math.nan,
            "per_traj_max_abs_diff_max": math.nan,
        }
    if len(rows) != confidence.shape[0]:
        raise ValueError(f"Per-trajectory row count mismatch for {degradation}: {len(rows)} vs {confidence.shape[0]}")
    calc_min = confidence.min(axis=1)
    calc_mean = confidence.mean(axis=1)
    calc_max = confidence.max(axis=1)
    saved_min = rows["confidence_min"].to_numpy(dtype=np.float64)
    saved_mean = rows["confidence_mean"].to_numpy(dtype=np.float64)
    saved_max = rows["confidence_max"].to_numpy(dtype=np.float64)
    diff_min = np.abs(calc_min - saved_min)
    diff_mean = np.abs(calc_mean - saved_mean)
    diff_max = np.abs(calc_max - saved_max)
    return {
        "per_traj_stats_available": True,
        "per_traj_min_match": bool(np.all(diff_min <= ATOL)),
        "per_traj_mean_match": bool(np.all(diff_mean <= ATOL)),
        "per_traj_max_match": bool(np.all(diff_max <= ATOL)),
        "per_traj_max_abs_diff_min": float(diff_min.max()),
        "per_traj_max_abs_diff_mean": float(diff_mean.max()),
        "per_traj_max_abs_diff_max": float(diff_max.max()),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[FILE] {path} written")


def markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    for col in table.columns:
        table[col] = table[col].map(
            lambda value: f"{float(value):.9f}"
            if isinstance(value, (float, np.floating)) and np.isfinite(value)
            else str(value)
        )
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for row in table.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary_md(summary_df: pd.DataFrame) -> None:
    compact_cols = [
        "degradation",
        "shape",
        "delta0",
        "n_high_conf_frames",
        "n_mid_conf_frames",
        "n_low_conf_frames",
        "bin_counts_match",
        "per_traj_confidence_stats_match",
        "range_valid",
        "validation_pass",
    ]
    lines = [
        "# Formal E1 Confidence Cache Validation Summary",
        "",
        "Formal E1 did not originally save per-frame c_t arrays. This cache is reconstructed from the locked Formal E1 formula and validated against Formal E1 confidence-bin/statistical summaries. It is now the audited c_t source for E2-DPS.",
        "",
        "Cache type: validated reconstructed Formal E1 c_t cache.",
        "",
        "Formula:",
        "",
        "`c_t = exp(- ||y_t - x_star_t|| / delta_0)`",
        "",
        "Bins:",
        "",
        "- high: c_t > 0.7",
        "- low: c_t < 0.3",
        "- mid: 0.3 <= c_t <= 0.7",
        "",
        "## Validation Table",
        markdown_table(summary_df[compact_cols]),
        "",
        "## Output Files",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"- {row['degradation']}: {row['confidence_path']} ; {row['validation_json_path']}")
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[FILE] {SUMMARY_MD_PATH} written")


def main() -> None:
    require_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_df = pd.read_csv(PROTOCOL_PATH)
    full_df = pd.read_csv(FULL_METRICS_PATH)
    bin_df = pd.read_csv(BIN_METRICS_PATH)
    per_df = pd.read_csv(PER_TRAJ_PATH)

    summary_rows: list[dict] = []
    for degradation in DEGRADATION_ORDER:
        clean, degraded, protocol_row = load_arrays(protocol_df, degradation)
        delta0 = get_delta0(full_df, degradation)
        confidence, delta = confidence_from_delta(degraded, clean, delta0)

        shape_valid = confidence.shape == clean.shape[:2] == degraded.shape[:2]
        range_valid = bool(np.all(np.isfinite(confidence)) and np.all(confidence >= 0.0) and np.all(confidence <= 1.0))

        high = confidence > 0.7
        low = confidence < 0.3
        mid = (confidence >= 0.3) & (confidence <= 0.7)
        counts = {
            "high": int(high.sum()),
            "mid": int(mid.sum()),
            "low": int(low.sum()),
            "total": int(confidence.size),
        }
        expected = expected_bin_counts(bin_df, degradation)
        expected_high = expected.get("high")
        expected_low = expected.get("low")
        expected_mid = expected.get("mid", counts["total"] - int(expected_high or 0) - int(expected_low or 0))
        high_match = expected_high is not None and counts["high"] == expected_high
        low_match = expected_low is not None and counts["low"] == expected_low
        mid_match = counts["mid"] == expected_mid
        bin_counts_match = bool(high_match and low_match and mid_match)

        traj_stats = validate_per_traj_stats(confidence, per_df, degradation)
        per_traj_stats_match = bool(
            traj_stats["per_traj_stats_available"]
            and traj_stats["per_traj_min_match"]
            and traj_stats["per_traj_mean_match"]
            and traj_stats["per_traj_max_match"]
        )
        validation_pass = bool(shape_valid and range_valid and bin_counts_match and per_traj_stats_match)

        confidence_path = OUT_DIR / f"{degradation}_confidence.npy"
        np.save(confidence_path, confidence.astype(np.float32))
        print(f"[FILE] {confidence_path} written")

        validation = {
            "degradation": degradation,
            "cache_type": "validated reconstructed Formal E1 c_t cache",
            "not_archived_original": True,
            "formula": "c_t = exp(- ||y_t - x_star_t|| / delta_0)",
            "delta0": delta0,
            "conditional_source": protocol_row.get("conditional_source", ""),
            "clean_path": str(CLEAN_PATH),
            "degraded_path": protocol_row.get("degraded_path", ""),
            "confidence_path": str(confidence_path),
            "shape_clean": list(clean.shape),
            "shape_degraded": list(degraded.shape),
            "shape_confidence": list(confidence.shape),
            "shape_valid": shape_valid,
            "range_valid": range_valid,
            "confidence_min": float(confidence.min()),
            "confidence_mean": float(confidence.mean()),
            "confidence_max": float(confidence.max()),
            "delta_min": float(delta.min()),
            "delta_mean": float(delta.mean()),
            "delta_max": float(delta.max()),
            "n_high_conf_frames": counts["high"],
            "n_mid_conf_frames": counts["mid"],
            "n_low_conf_frames": counts["low"],
            "n_total_frames": counts["total"],
            "formal_e1_expected_high": expected_high,
            "formal_e1_expected_mid": expected_mid,
            "formal_e1_expected_low": expected_low,
            "high_count_match": high_match,
            "mid_count_match": mid_match,
            "low_count_match": low_match,
            "bin_counts_match": bin_counts_match,
            **traj_stats,
            "per_traj_confidence_stats_match": per_traj_stats_match,
            "validation_pass": validation_pass,
        }
        validation_path = OUT_DIR / f"{degradation}_confidence_validation.json"
        write_json(validation_path, validation)

        summary_rows.append(
            {
                "degradation": degradation,
                "confidence_path": str(confidence_path),
                "validation_json_path": str(validation_path),
                "shape": str(tuple(confidence.shape)),
                "delta0": delta0,
                "confidence_min": float(confidence.min()),
                "confidence_mean": float(confidence.mean()),
                "confidence_max": float(confidence.max()),
                "n_high_conf_frames": counts["high"],
                "n_mid_conf_frames": counts["mid"],
                "n_low_conf_frames": counts["low"],
                "formal_e1_expected_high": expected_high,
                "formal_e1_expected_mid": expected_mid,
                "formal_e1_expected_low": expected_low,
                "bin_counts_match": bin_counts_match,
                "per_traj_confidence_stats_match": per_traj_stats_match,
                "max_abs_diff_confidence_min": traj_stats["per_traj_max_abs_diff_min"],
                "max_abs_diff_confidence_mean": traj_stats["per_traj_max_abs_diff_mean"],
                "max_abs_diff_confidence_max": traj_stats["per_traj_max_abs_diff_max"],
                "range_valid": range_valid,
                "shape_valid": shape_valid,
                "validation_pass": validation_pass,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"[FILE] {SUMMARY_CSV_PATH} written")
    write_summary_md(summary_df)

    print("FORMAL_E1_CONFIDENCE_CACHE_REBUILD_COMPLETE")
    print(f"conditions_generated: {int(summary_df['validation_pass'].notna().sum())}/6")
    print(f"all_validation_pass: {bool(summary_df['validation_pass'].all())}")
    for _, row in summary_df.iterrows():
        print(
            f"{row['degradation']}: shape={row['shape']} "
            f"high={row['n_high_conf_frames']} mid={row['n_mid_conf_frames']} low={row['n_low_conf_frames']} "
            f"bin_match={row['bin_counts_match']} stats_match={row['per_traj_confidence_stats_match']} "
            f"validation={row['validation_pass']}"
        )
    print(f"cache_dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
