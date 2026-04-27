from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3.controlled.evaluate_coarse_reconstruction import (
    compute_geometry_metrics,
    compute_reconstruction_metrics,
    load_array,
    room3_map_meta,
)
from tools.stage3.refinement.refiners import REFINER_LABELS, REFINER_NAMES
from tools.stage3.refinement.run_refinement_interface import (
    REFINEMENT_EVAL_DIR,
    REFINEMENT_ROOT_DIR,
    ensure_refinement_dirs,
    refined_path,
)
from utils.stage3.controlled_benchmark import (
    DEGRADATION_NAMES,
    METHODS,
    METHOD_LABELS,
    DEFAULT_SEED,
    DEFAULT_SPAN_MODE,
    DEFAULT_SPAN_RATIO,
    clean_path,
    experiment_tag,
    mask_path,
    reconstruction_path,
)
from utils.stage3.io import save_json


def safe_improvement(coarse_value: float, refined_value: float):
    if abs(coarse_value) < 1e-12:
        return 0.0
    return float((coarse_value - refined_value) / coarse_value)


def metrics_csv_path():
    return REFINEMENT_EVAL_DIR / "refinement_metrics.csv"


def metrics_json_path():
    return REFINEMENT_EVAL_DIR / "refinement_metrics.json"


def report_path():
    return REFINEMENT_ROOT_DIR / "refinement_report.md"


def compute_geometry_metrics_fast(pred: np.ndarray, map_meta: dict):
    occupancy = map_meta["occupancy"]
    if np.any(occupancy == 1):
        return compute_geometry_metrics(pred, map_meta)

    x = pred[:, :, 0]
    y = pred[:, :, 1]
    off_map = (
        (x < map_meta["x_min"])
        | (x > map_meta["x_max"])
        | (y < map_meta["y_min"])
        | (y > map_meta["y_max"])
    )
    total_points = pred.shape[0] * pred.shape[1]
    return {
        "off_map_ratio": float(off_map.sum() / total_points) if total_points > 0 else 0.0,
        "wall_crossing_count": 0,
    }


def write_csv(rows: list[dict], output_path: Path):
    fieldnames = [
        "degradation",
        "coarse_method",
        "refiner",
        "ADE",
        "FDE",
        "RMSE",
        "masked_ADE",
        "masked_RMSE",
        "improvement_ADE",
        "improvement_masked_ADE",
        "off_map_ratio",
        "wall_crossing_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict], output_path: Path):
    focus_metrics = ["ADE", "RMSE", "masked_ADE", "masked_RMSE"]
    lines = [
        "# Stage 3 Refinement Report",
        "",
        "## Purpose",
        "",
        "Test whether a lightweight refinement stage improves coarse reconstruction in the current controlled benchmark.",
        "",
        "## Setup",
        "",
        "- Degradation types: missing_only, missing_noise, missing_drift, missing_noise_drift",
        "- Coarse methods: Linear, SG, Kalman",
        "- Refiners: Identity, Light SG, DDPM prior interface v0",
        "",
        "## Metric Interpretation",
        "",
        "Full-trajectory metrics measure overall consistency, while masked metrics measure reconstruction quality on the missing segment itself.",
        "When the two views differ, both should be reported explicitly.",
        "",
        "## Mean Results By Refiner",
        "",
        "| coarse_method | refiner | ADE | RMSE | masked_ADE | masked_RMSE | improvement_ADE | improvement_masked_ADE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for coarse_method in METHODS:
        for refiner in REFINER_NAMES:
            subset = [row for row in rows if row["coarse_method"] == coarse_method and row["refiner"] == refiner]
            means = {metric: float(np.mean([row[metric] for row in subset])) for metric in focus_metrics}
            imp_ade = float(np.mean([row["improvement_ADE"] for row in subset]))
            imp_masked = float(np.mean([row["improvement_masked_ADE"] for row in subset]))
            lines.append(
                f"| {METHOD_LABELS[coarse_method]} | {REFINER_LABELS[refiner]} | "
                f"{means['ADE']:.6f} | {means['RMSE']:.6f} | {means['masked_ADE']:.6f} | "
                f"{means['masked_RMSE']:.6f} | {imp_ade:.6f} | {imp_masked:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report keeps full-trajectory metrics and masked metrics as complementary views.",
            "The key question for refinement is whether masked_ADE improves, because that directly reflects missing-segment reconstruction quality.",
            "",
            "## First learned-prior refinement interface",
            "",
        "The `ddpm_prior_interface_v0` refiner is the first Stage 3 interface-level connection to the Stage 2 learned prior.",
        "It is not yet a fully optimized conditional diffusion refinement model.",
        "Instead, it uses a one-shot prior projection in relative displacement space and then maps the result back to absolute trajectories.",
        "Its purpose is to verify that Stage 2 prior checkpoints can be connected cleanly to Stage 3 coarse reconstructions.",
        "",
        "In the current benchmark, the interface is operational but the v0 prior refinement does not improve over the simple baselines.",
        "Its mean ADE and mean masked_ADE are both worse than Identity and Light SG across all three coarse-method families.",
        "This is still a useful result because it validates the integration path without overclaiming performance.",
        "",
        "## Figures",
        "",
        "- `figures/coarse_vs_refined_ADE.png`",
        "- `figures/coarse_vs_refined_masked_ADE.png`",
        "- `figures/improvement_bar_chart.png`",
        "- `figures/full_vs_masked_refinement_improvement.png`",
        "- `figures/ddpm_vs_naive_refinement_improvement.png`",
        "- `figures/trajectory_example_coarse_refined.png`",
    ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 refinement outputs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--span_ratio", type=float, default=DEFAULT_SPAN_RATIO)
    parser.add_argument("--span_mode", type=str, default=DEFAULT_SPAN_MODE, choices=["fixed", "random"])
    args = parser.parse_args()

    ensure_refinement_dirs()
    tag = experiment_tag(args.span_ratio, args.span_mode, args.seed)
    clean = load_array(clean_path()).astype(np.float32)
    obs_mask = load_array(mask_path(tag)).astype(np.uint8)
    map_meta = room3_map_meta()

    if clean.ndim != 3 or clean.shape[-1] != 2:
        raise ValueError(f"Expected clean trajectory shape [N, T, 2], got {clean.shape}")
    if obs_mask.shape != clean.shape[:2]:
        raise ValueError(f"obs_mask shape {obs_mask.shape} does not match clean shape {clean.shape[:2]}")

    rows = []
    grouped = {}
    for degradation in DEGRADATION_NAMES:
        grouped[degradation] = {}
        for coarse_method in METHODS:
            coarse = load_array(reconstruction_path(degradation, coarse_method, tag)).astype(np.float32)
            coarse_metrics = compute_reconstruction_metrics(clean, coarse, obs_mask)
            for refiner in REFINER_NAMES:
                refined = load_array(refined_path(degradation, coarse_method, refiner)).astype(np.float32)
                recon_metrics = compute_reconstruction_metrics(clean, refined, obs_mask)
                geom_metrics = compute_geometry_metrics_fast(refined, map_meta)
                row = {
                    "degradation": degradation,
                    "coarse_method": coarse_method,
                    "refiner": refiner,
                    **recon_metrics,
                    "improvement_ADE": safe_improvement(coarse_metrics["ADE"], recon_metrics["ADE"]),
                    "improvement_masked_ADE": safe_improvement(
                        coarse_metrics["masked_ADE"], recon_metrics["masked_ADE"]
                    ),
                    **geom_metrics,
                }
                rows.append(row)
                grouped[degradation].setdefault(coarse_method, []).append(row)

    write_csv(rows, metrics_csv_path())
    save_json(
        metrics_json_path(),
        {
            "config": {
                "seed": args.seed,
                "span_ratio": args.span_ratio,
                "span_mode": args.span_mode,
                "experiment_tag": tag,
            },
            "rows": rows,
            "by_degradation": grouped,
        },
    )
    build_report(rows, report_path())

    print("=" * 60)
    print("Refinement metrics summary")
    for row in rows:
        print(
            f"{row['degradation']:20s} {row['coarse_method']:30s} {row['refiner']:22s} "
            f"ADE={row['ADE']:.6f} masked_ADE={row['masked_ADE']:.6f} "
            f"improve_ADE={row['improvement_ADE']:.6f} "
            f"improve_masked_ADE={row['improvement_masked_ADE']:.6f}"
        )
    print(f"metrics_csv = {metrics_csv_path()}")
    print(f"metrics_json = {metrics_json_path()}")
    print(f"report_path = {report_path()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
