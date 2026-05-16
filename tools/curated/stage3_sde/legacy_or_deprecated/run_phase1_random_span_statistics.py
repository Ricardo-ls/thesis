from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage3.io import load_json, load_npz, save_json
from utils.stage3.paths import (
    CLEAN_ROOM3_PATH,
    KALMAN_METHOD_TAG,
    LINEAR_METHOD_TAG,
    PHASE1_OUTPUT_DIR,
    RANDOM_SPAN_STATS_DIR,
    SAVGOL_METHOD_TAG,
    ensure_stage3_dirs,
)


METHOD_TAGS = [
    LINEAR_METHOD_TAG,
    SAVGOL_METHOD_TAG,
    KALMAN_METHOD_TAG,
]

RECONSTRUCTION_METRICS = ["ADE", "FDE", "RMSE", "masked_ADE", "masked_RMSE"]
GEOMETRY_METRICS = ["off_map_ratio", "wall_crossing_count"]
SUMMARY_METRICS = RECONSTRUCTION_METRICS + GEOMETRY_METRICS


def run_command(cmd: list[str]):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def experiment_id_for(seed: int, span_ratio: float):
    span_percent = int(round(span_ratio * 100))
    return f"span{span_percent}_random_seed{seed}"


def load_num_trajectories():
    data = load_npz(CLEAN_ROOM3_PATH)
    traj_abs = np.asarray(data["traj_abs"], dtype=np.float32)
    if traj_abs.ndim != 3 or traj_abs.shape[-1] != 2:
        raise ValueError(f"Expected traj_abs with shape [N, T, 2], got {traj_abs.shape}")
    return int(traj_abs.shape[0]), int(traj_abs.shape[1])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_summary_rows(rows: list[dict], span_ratio: float, span_mode: str):
    summary_rows = []
    for method_tag in METHOD_TAGS:
        method_rows = [row for row in rows if row["method"] == method_tag]
        if not method_rows:
            raise ValueError(f"No rows collected for method {method_tag}")
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in method_rows], dtype=np.float64)
            summary_rows.append(
                {
                    "span_ratio": span_ratio,
                    "span_mode": span_mode,
                    "method": method_tag,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return summary_rows


def build_report(summary_rows: list[dict], output_path: Path):
    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row["method"], {})[row["metric"]] = row

    lines = [
        "# Stage 3 Phase 1 Random-Span Statistics",
        "",
        "## Purpose",
        "",
        "Random-span statistics test whether the current canonical_room3 benchmark behavior remains stable across many missing-span positions instead of depending on one fixed gap placement.",
        "",
        "## Protocol",
        "",
        "- `span_ratio = 0.2`",
        "- `span_mode = random`",
        "- `seeds = 0..19`",
        f"- baselines: `{LINEAR_METHOD_TAG}`, `{SAVGOL_METHOD_TAG}`, `{KALMAN_METHOD_TAG}`",
        "",
        "## Metric Interpretation",
        "",
        "This benchmark reports two complementary evaluation views. Full-trajectory metrics measure overall trajectory consistency, while masked metrics measure reconstruction quality on the missing segment. When the two views rank methods differently, both rankings are reported explicitly.",
        "",
        "Full-trajectory metrics, including ADE, FDE, and RMSE, measure overall trajectory consistency over the full window. Masked metrics, including masked_ADE and masked_RMSE, measure reconstruction quality on the removed segment itself. Since the task is missing-segment reconstruction, masked metrics are emphasized when discussing reconstruction quality on the missing span. When the two views rank methods differently, both rankings are reported explicitly rather than collapsed into a single overall ranking.",
        "",
        "## Mean +- Std Results",
        "",
        "| method | ADE | RMSE | masked_ADE | masked_RMSE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method_tag in METHOD_TAGS:
        method_metrics = grouped[method_tag]
        lines.append(
            "| "
            f"{method_tag} | "
            f"{method_metrics['ADE']['mean']:.6f} +- {method_metrics['ADE']['std']:.6f} | "
            f"{method_metrics['RMSE']['mean']:.6f} +- {method_metrics['RMSE']['std']:.6f} | "
            f"{method_metrics['masked_ADE']['mean']:.6f} +- {method_metrics['masked_ADE']['std']:.6f} | "
            f"{method_metrics['masked_RMSE']['mean']:.6f} +- {method_metrics['masked_RMSE']['std']:.6f} |"
        )

    ade_ranking = sorted(METHOD_TAGS, key=lambda m: grouped[m]["ADE"]["mean"])
    masked_ade_ranking = sorted(METHOD_TAGS, key=lambda m: grouped[m]["masked_ADE"]["mean"])
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            f"- Full-trajectory view by mean ADE: `{', '.join(ade_ranking)}`",
            f"- Missing-segment view by mean masked_ADE: `{', '.join(masked_ade_ranking)}`",
            "",
        ]
    )
    if ade_ranking[:2] != masked_ade_ranking[:2]:
        lines.append(
            "- The full-trajectory view and the missing-segment view rank Linear and Savitzky-Golay differently, so both rankings should be retained in the interpretation."
        )
    else:
        lines.append(
            "- The full-trajectory view and the missing-segment view agree on the top ordering under the current random-span sweep."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "This random-span sweep strengthens statistical reliability relative to a single fixed missing-span position. It should be read as a stability check for the current Phase 1 benchmark, not as a reason to collapse the two metric views into one overall winner.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 3 Phase 1 random-span statistics over seeds."
    )
    parser.add_argument("--span_ratio", type=float, default=0.2)
    parser.add_argument("--span_mode", type=str, default="random", choices=["random"])
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=19)
    args = parser.parse_args()

    ensure_stage3_dirs()
    RANDOM_SPAN_STATS_DIR.mkdir(parents=True, exist_ok=True)

    num_samples, sequence_length = load_num_trajectories()
    print("=" * 60)
    print("Stage 3 Phase 1 random-span statistics")
    print(f"num_seeds         = {args.seed_end - args.seed_start + 1}")
    print(f"num_trajectories  = {num_samples}")
    print(f"sequence_length   = {sequence_length}")
    print(f"output_dir        = {RANDOM_SPAN_STATS_DIR}")
    print("=" * 60)

    rows = []
    python_exe = sys.executable

    for seed in range(args.seed_start, args.seed_end + 1):
        experiment_id = experiment_id_for(seed=seed, span_ratio=args.span_ratio)
        run_command(
            [
                python_exe,
                "-m",
                "tools.stage3.data.generate_missing_span",
                "--experiment_id",
                experiment_id,
                "--span_ratio",
                str(args.span_ratio),
                "--span_mode",
                args.span_mode,
                "--seed",
                str(seed),
            ]
        )
        for module_name, method_tag, extra_args in [
            ("tools.stage3.baselines.run_linear_interp", LINEAR_METHOD_TAG, []),
            (
                "tools.stage3.baselines.run_savgol",
                SAVGOL_METHOD_TAG,
                ["--window_length", "5", "--polyorder", "2"],
            ),
            (
                "tools.stage3.baselines.run_kalman",
                KALMAN_METHOD_TAG,
                ["--dt", "1.0", "--process_var", "1e-3", "--measure_var", "1e-2"],
            ),
        ]:
            run_command(
                [python_exe, "-m", module_name, "--experiment_id", experiment_id, *extra_args]
            )
            run_command(
                [
                    python_exe,
                    "-m",
                    "tools.stage3.eval.eval_reconstruction_metrics",
                    "--experiment_id",
                    experiment_id,
                    "--method_tag",
                    method_tag,
                ]
            )
            run_command(
                [
                    python_exe,
                    "-m",
                    "tools.stage3.eval.eval_geometry_metrics",
                    "--experiment_id",
                    experiment_id,
                    "--method_tag",
                    method_tag,
                ]
            )
            recon_path = (
                PHASE1_OUTPUT_DIR
                / "eval"
                / experiment_id
                / method_tag
                / "reconstruction_metrics.json"
            )
            geom_path = (
                PHASE1_OUTPUT_DIR
                / "eval"
                / experiment_id
                / method_tag
                / "geometry_metrics.json"
            )
            recon_metrics = load_json(recon_path)
            geom_metrics = load_json(geom_path)
            rows.append(
                {
                    "seed": seed,
                    "span_ratio": args.span_ratio,
                    "span_mode": args.span_mode,
                    "method": method_tag,
                    "ADE": recon_metrics["ADE"],
                    "FDE": recon_metrics["FDE"],
                    "RMSE": recon_metrics["RMSE"],
                    "masked_ADE": recon_metrics["masked_ADE"],
                    "masked_RMSE": recon_metrics["masked_RMSE"],
                    "off_map_ratio": geom_metrics["off_map_ratio"],
                    "wall_crossing_count": geom_metrics["wall_crossing_count"],
                }
            )

    metrics_csv_path = RANDOM_SPAN_STATS_DIR / "metrics_by_seed.csv"
    metrics_json_path = RANDOM_SPAN_STATS_DIR / "metrics_by_seed.json"
    summary_csv_path = RANDOM_SPAN_STATS_DIR / "metrics_summary_mean_std.csv"
    summary_json_path = RANDOM_SPAN_STATS_DIR / "metrics_summary_mean_std.json"
    report_path = RANDOM_SPAN_STATS_DIR / "random_span_statistics_report.md"

    write_csv(
        metrics_csv_path,
        rows,
        [
            "method",
            "seed",
            "span_ratio",
            "span_mode",
            "ADE",
            "FDE",
            "RMSE",
            "masked_ADE",
            "masked_RMSE",
            "off_map_ratio",
            "wall_crossing_count",
        ],
    )
    save_json(metrics_json_path, rows)

    summary_rows = compute_summary_rows(
        rows=rows,
        span_ratio=args.span_ratio,
        span_mode=args.span_mode,
    )
    write_csv(
        summary_csv_path,
        summary_rows,
        ["span_ratio", "span_mode", "method", "metric", "mean", "std", "min", "max"],
    )
    save_json(summary_json_path, summary_rows)
    build_report(summary_rows=summary_rows, output_path=report_path)

    print("=" * 60)
    print("Random-span statistics finished")
    print(f"metrics_by_seed          = {metrics_csv_path}")
    print(f"metrics_summary_mean_std = {summary_csv_path}")
    print(f"report                   = {report_path}")
    print("-" * 60)
    for row in summary_rows:
        if row["metric"] in {"ADE", "RMSE", "masked_ADE", "masked_RMSE"}:
            print(
                f"{row['method']:>32s} | {row['metric']:>11s} | "
                f"mean={row['mean']:.6f} std={row['std']:.6f}"
            )
    print("=" * 60)


if __name__ == "__main__":
    main()
