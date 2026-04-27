from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3.geometry_extension.wall_door_geometry import (
    canonical_room3_wall_door_v1,
    compute_wall_door_metrics,
)
from tools.stage3.refinement.run_refinement_interface import load_array
from utils.stage3.controlled_benchmark import METHODS

OUT_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "geometry_extension" / "wall_door_v1"
FIG_DIR = OUT_ROOT / "figures"

PHASE1_BASELINES_DIR = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "baselines"
CONTROLLED_RECON_DIR = PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark" / "reconstruction"
REFINEMENT_REFINED_DIR = PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "refined"
ALPHA_SWEEP_REFINED_DIR = PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "alpha_sweep" / "refined"

DEGRADATION_NAMES = [
    "missing_only",
    "missing_noise",
    "missing_drift",
    "missing_noise_drift",
]
CONTROLLED_METHODS = ["input_degraded", *METHODS]


def ensure_dirs():
    for path in [OUT_ROOT, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def geometry_metrics_csv_path() -> Path:
    return OUT_ROOT / "geometry_metrics.csv"


def geometry_summary_csv_path() -> Path:
    return OUT_ROOT / "geometry_summary.csv"


def report_path() -> Path:
    return OUT_ROOT / "geometry_extension_report.md"


def figure_path() -> Path:
    return FIG_DIR / "geometry_violation_summary.png"


def parse_controlled_reconstruction(path: Path) -> tuple[str, str, str] | None:
    stem = path.stem
    for degradation in sorted(DEGRADATION_NAMES, key=len, reverse=True):
        prefix = f"recon_{degradation}_"
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix) :]
        for method in sorted(CONTROLLED_METHODS, key=len, reverse=True):
            method_prefix = f"{method}_"
            if suffix.startswith(method_prefix):
                tag = suffix[len(method_prefix) :]
                return degradation, method, tag
    return None


def parse_refined_name(path: Path) -> tuple[str, str, str] | None:
    stem = path.stem
    for degradation in sorted(DEGRADATION_NAMES, key=len, reverse=True):
        prefix = f"refined_{degradation}_"
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix) :]
        for coarse_method in sorted(METHODS, key=len, reverse=True):
            coarse_prefix = f"{coarse_method}_"
            if suffix.startswith(coarse_prefix):
                refiner = suffix[len(coarse_prefix) :]
                return degradation, coarse_method, refiner
    return None


def alpha_from_refiner(refiner_name: str) -> float | None:
    match = re.search(r"alpha([0-9]+(?:\.[0-9]+)?)$", refiner_name)
    if match is None:
        return None
    return float(match.group(1))


def collect_phase1_rows(profile_name: str) -> list[dict]:
    rows = []
    if not PHASE1_BASELINES_DIR.exists():
        return rows

    for results_path in sorted(PHASE1_BASELINES_DIR.glob("*/ */results.npz".replace(" ", ""))):
        experiment_id = results_path.parent.parent.name
        method_name = results_path.parent.name
        data = np.load(results_path, allow_pickle=False)
        if "traj_hat" not in data:
            continue
        metrics = compute_wall_door_metrics(np.asarray(data["traj_hat"], dtype=np.float32), canonical_room3_wall_door_v1())
        rows.append(
            {
                "geometry_profile": profile_name,
                "source_family": "phase1_baseline",
                "experiment_id": experiment_id,
                "degradation": "",
                "coarse_method": "",
                "method_name": method_name,
                "refiner_name": "",
                "alpha": np.nan,
                "output_path": str(results_path),
                **metrics,
            }
        )
    return rows


def collect_controlled_rows(profile_name: str) -> list[dict]:
    rows = []
    if not CONTROLLED_RECON_DIR.exists():
        return rows

    for path in sorted(CONTROLLED_RECON_DIR.glob("recon_*.npy")):
        parsed = parse_controlled_reconstruction(path)
        if parsed is None:
            continue
        degradation, method_name, experiment_id = parsed
        traj = load_array(path).astype(np.float32)
        metrics = compute_wall_door_metrics(traj, canonical_room3_wall_door_v1())
        rows.append(
            {
                "geometry_profile": profile_name,
                "source_family": "controlled_benchmark",
                "experiment_id": experiment_id,
                "degradation": degradation,
                "coarse_method": "",
                "method_name": method_name,
                "refiner_name": "",
                "alpha": np.nan,
                "output_path": str(path),
                **metrics,
            }
        )
    return rows


def collect_refinement_rows(profile_name: str, source_dir: Path, source_family: str) -> list[dict]:
    rows = []
    if not source_dir.exists():
        return rows

    for path in sorted(source_dir.glob("refined_*.npy")):
        parsed = parse_refined_name(path)
        if parsed is None:
            continue
        degradation, coarse_method, refiner_name = parsed
        traj = load_array(path).astype(np.float32)
        metrics = compute_wall_door_metrics(traj, canonical_room3_wall_door_v1())
        rows.append(
            {
                "geometry_profile": profile_name,
                "source_family": source_family,
                "experiment_id": "",
                "degradation": degradation,
                "coarse_method": coarse_method,
                "method_name": refiner_name,
                "refiner_name": refiner_name,
                "alpha": alpha_from_refiner(refiner_name),
                "output_path": str(path),
                **metrics,
            }
        )
    return rows


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "geometry_profile",
        "source_family",
        "experiment_id",
        "degradation",
        "coarse_method",
        "method_name",
        "refiner_name",
        "alpha",
    ]
    grouped = (
        metrics_df.groupby(group_cols, dropna=False, as_index=False)[
            [
                "boundary_violation_count",
                "internal_wall_crossing_count",
                "door_valid_crossing_count",
                "infeasible_transition_count",
                "total_points",
                "total_segments",
            ]
        ]
        .sum()
    )
    grouped["off_map_ratio"] = grouped["boundary_violation_count"] / grouped["total_points"].clip(lower=1)
    grouped["door_crossing_valid_ratio"] = grouped["door_valid_crossing_count"] / grouped[
        "internal_wall_crossing_count"
    ].clip(lower=1)
    grouped = grouped[
        [
            "geometry_profile",
            "source_family",
            "experiment_id",
            "degradation",
            "coarse_method",
            "method_name",
            "refiner_name",
            "alpha",
            "off_map_ratio",
            "boundary_violation_count",
            "internal_wall_crossing_count",
            "door_valid_crossing_count",
            "door_crossing_valid_ratio",
            "infeasible_transition_count",
            "total_points",
            "total_segments",
        ]
    ]
    return grouped.sort_values(group_cols).reset_index(drop=True)


def plot_summary(summary_df: pd.DataFrame, output_path: Path):
    plot_df = (
        summary_df.groupby(["source_family", "method_name"], dropna=False, as_index=False)["infeasible_transition_count"]
        .sum()
        .sort_values(["source_family", "infeasible_transition_count"], ascending=[True, False])
    )
    families = ["phase1_baseline", "controlled_benchmark", "refinement", "refinement_alpha_sweep"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    for axis, family in zip(axes, families):
        subset = plot_df[plot_df["source_family"] == family].head(10)
        if subset.empty:
            axis.set_visible(False)
            continue
        axis.barh(subset["method_name"], subset["infeasible_transition_count"], color="#4C78A8")
        axis.set_title(family)
        axis.set_xlabel("infeasible_transition_count")
        axis.grid(axis="x", alpha=0.25)
        axis.invert_yaxis()

    fig.suptitle("Wall-door v1 geometry violations")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in df.to_dict(orient="records"):
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)) and pd.notna(val):
                vals.append(f"{float(val):.6f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_report(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path):
    by_family = (
        summary_df.groupby("source_family", dropna=False, as_index=False)[
            [
                "boundary_violation_count",
                "internal_wall_crossing_count",
                "door_valid_crossing_count",
                "infeasible_transition_count",
                "total_points",
                "total_segments",
            ]
        ]
        .sum()
    )
    by_family["off_map_ratio"] = by_family["boundary_violation_count"] / by_family["total_points"].clip(lower=1)
    by_family["door_crossing_valid_ratio"] = by_family["door_valid_crossing_count"] / by_family[
        "internal_wall_crossing_count"
    ].clip(lower=1)

    worst_rows = summary_df.sort_values(
        ["infeasible_transition_count", "internal_wall_crossing_count"],
        ascending=[False, False],
    ).head(12)[
        [
            "source_family",
            "experiment_id",
            "degradation",
            "coarse_method",
            "method_name",
            "alpha",
            "infeasible_transition_count",
            "internal_wall_crossing_count",
            "door_valid_crossing_count",
        ]
    ]

    lines = [
        "# Stage 3 Geometry Extension Report",
        "",
        "## Scope",
        "",
        "The fixed `canonical_room3` benchmark remains unchanged.",
        "This report adds a separate geometry extension profile, `canonical_room3_wall_door_v1`, to test indoor feasibility constraints on already-generated trajectories.",
        "",
        "## Geometry Profile",
        "",
        "- Room boundary: `[0, 3] x [0, 3]`",
        "- Internal wall: vertical wall at `x = 1.5`",
        "- Door opening: `y in [1.2, 1.8]`",
        "- Side-to-side crossings of the internal wall are feasible only if the crossing point lies inside the door interval.",
        "",
        "## Metric Interpretation",
        "",
        "- `off_map_ratio`: fraction of trajectory points outside the room boundary.",
        "- `boundary_violation_count`: count of points outside the room boundary.",
        "- `internal_wall_crossing_count`: count of side-to-side crossings of the internal wall.",
        "- `door_valid_crossing_count`: subset of wall crossings that pass through the door opening.",
        "- `infeasible_transition_count`: count of transitions that either use an invalid wall crossing or have an off-boundary endpoint.",
        "",
        "## Source Coverage",
        "",
        f"- Evaluated rows: `{len(metrics_df)}`",
        f"- Source families covered: `{', '.join(sorted(metrics_df['source_family'].unique()))}`",
        "",
        "## Summary By Source Family",
        "",
        markdown_table(
            by_family[
                [
                    "source_family",
                    "off_map_ratio",
                    "boundary_violation_count",
                    "internal_wall_crossing_count",
                    "door_valid_crossing_count",
                    "door_crossing_valid_ratio",
                    "infeasible_transition_count",
                ]
            ]
        ),
        "",
        "## Highest-Violation Rows",
        "",
        markdown_table(worst_rows),
        "",
        "## Interpretation",
        "",
        "This extension does not redefine the Stage 3 reconstruction task and does not replace the canonical empty-room protocol.",
        "It adds a stricter indoor feasibility lens so that future reconstruction or refinement methods can be checked for physically valid transitions under a simple wall-and-door layout.",
        "",
        "## Output Files",
        "",
        "- `geometry_metrics.csv`",
        "- `geometry_summary.csv`",
        "- `geometry_extension_report.md`",
        "- `figures/geometry_violation_summary.png`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(summary_df: pd.DataFrame):
    print("=" * 72)
    print("Geometry extension summary by source family")
    family_df = (
        summary_df.groupby("source_family", dropna=False, as_index=False)[
            ["boundary_violation_count", "internal_wall_crossing_count", "door_valid_crossing_count", "infeasible_transition_count"]
        ]
        .sum()
        .sort_values("source_family")
    )
    for _, row in family_df.iterrows():
        print(
            f"{row['source_family']:24s} "
            f"boundary={int(row['boundary_violation_count'])} "
            f"wall_cross={int(row['internal_wall_crossing_count'])} "
            f"door_valid={int(row['door_valid_crossing_count'])} "
            f"infeasible={int(row['infeasible_transition_count'])}"
        )
    print(f"metrics_csv = {geometry_metrics_csv_path()}")
    print(f"summary_csv = {geometry_summary_csv_path()}")
    print(f"report_path = {report_path()}")
    print(f"figure_path = {figure_path()}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 wall-door geometry extension evaluation.")
    parser.parse_args()

    ensure_dirs()
    profile = canonical_room3_wall_door_v1()

    rows = []
    rows.extend(collect_phase1_rows(profile.profile_name))
    rows.extend(collect_controlled_rows(profile.profile_name))
    rows.extend(collect_refinement_rows(profile.profile_name, REFINEMENT_REFINED_DIR, "refinement"))
    rows.extend(collect_refinement_rows(profile.profile_name, ALPHA_SWEEP_REFINED_DIR, "refinement_alpha_sweep"))

    if not rows:
        raise RuntimeError("No Stage 3 outputs found for geometry extension evaluation.")

    metrics_df = pd.DataFrame(rows)
    summary_df = build_summary(metrics_df)

    metrics_df.to_csv(geometry_metrics_csv_path(), index=False)
    summary_df.to_csv(geometry_summary_csv_path(), index=False)
    plot_summary(summary_df, figure_path())
    build_report(metrics_df, summary_df, report_path())
    print_console_summary(summary_df)


if __name__ == "__main__":
    main()
