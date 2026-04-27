from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
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
    compute_wall_door_window_diagnostics,
)
from tools.stage3.refinement.run_refinement_interface import load_array
from utils.stage3.controlled_benchmark import METHODS

OUT_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "geometry_extension" / "wall_door_v1"
FIG_DIR = OUT_ROOT / "figures"
DOCS_ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "stage3"
FIGURE_ASSET_MANIFEST = PROJECT_ROOT / "outputs" / "stage3" / "figure_assets" / "figure_manifest.md"
DOCS_FIGURE_MANIFEST = DOCS_ASSET_ROOT / "figure_manifest.md"

PHASE1_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3"
PHASE1_BASELINES_DIR = PHASE1_ROOT / "baselines"
PHASE1_EXPERIMENTS_DIR = PHASE1_ROOT / "data" / "experiments"
PHASE1_CLEAN_PATH = PHASE1_ROOT / "data" / "clean_windows_room3.npz"
CONTROLLED_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark"
CONTROLLED_RECON_DIR = CONTROLLED_ROOT / "reconstruction"
CONTROLLED_CLEAN_PATH = CONTROLLED_ROOT / "degradation" / "clean.npy"
CONTROLLED_MASK_PATH = CONTROLLED_ROOT / "degradation" / "mask_span20_fixed_seed42.npy"
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


def feasible_indices_path() -> Path:
    return OUT_ROOT / "feasible_indices.npy"


def filter_summary_csv_path() -> Path:
    return OUT_ROOT / "geometry_filter_summary.csv"


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


def phase1_mask_lookup() -> dict[str, np.ndarray]:
    lookup = {}
    for path in sorted(PHASE1_EXPERIMENTS_DIR.glob("*/missing_span_windows.npz")):
        data = np.load(path, allow_pickle=False)
        lookup[path.parent.name] = np.asarray(data["obs_mask"], dtype=np.uint8)
    return lookup


def compute_filter_summary(clean: np.ndarray, profile) -> tuple[np.ndarray, pd.DataFrame, dict]:
    diagnostics = compute_wall_door_window_diagnostics(clean, profile)
    clean_off_map_ratio = diagnostics["clean_off_map_ratio_by_window"]
    clean_internal = diagnostics["internal_wall_crossing_count_by_window"]
    clean_door_valid = diagnostics["door_valid_crossing_count_by_window"]
    clean_infeasible = diagnostics["infeasible_transition_count_by_window"]

    feasible_mask = (clean_off_map_ratio == 0.0) & (clean_infeasible == 0)
    feasible_indices = np.flatnonzero(feasible_mask).astype(np.int64)
    same_side = int(np.sum(feasible_mask & (clean_internal == 0)))
    door_transition = int(np.sum(feasible_mask & (clean_internal > 0) & (clean_infeasible == 0)))
    total_windows = int(clean.shape[0])
    feasible_windows = int(feasible_mask.sum())
    discarded_windows = total_windows - feasible_windows
    retention_rate = float(feasible_windows / total_windows) if total_windows > 0 else 0.0

    summary = pd.DataFrame(
        [
            {
                "total_windows": total_windows,
                "feasible_windows": feasible_windows,
                "discarded_windows": discarded_windows,
                "retention_rate": retention_rate,
                "same_side_windows": same_side,
                "door_transition_windows": door_transition,
            }
        ]
    )
    detail = {
        "clean_off_map_ratio_by_window": clean_off_map_ratio,
        "clean_internal_wall_crossing_count_by_window": clean_internal,
        "clean_door_valid_crossing_count_by_window": clean_door_valid,
        "clean_infeasible_transition_count_by_window": clean_infeasible,
        "feasible_mask": feasible_mask,
    }
    return feasible_indices, summary, detail


def subset_mask(mask: np.ndarray | None, feasible_indices: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    return np.asarray(mask, dtype=np.uint8)[feasible_indices]


def build_row(
    *,
    profile_name: str,
    source_family: str,
    experiment_id: str,
    degradation: str,
    coarse_method: str,
    method_name: str,
    refiner_name: str,
    alpha: float | None,
    output_path: Path,
    traj: np.ndarray,
    feasible_indices: np.ndarray,
    profile,
    obs_mask: np.ndarray | None,
) -> dict:
    traj = np.asarray(traj, dtype=np.float32)[feasible_indices]
    obs_mask = subset_mask(obs_mask, feasible_indices)
    diagnostics = compute_wall_door_window_diagnostics(traj, profile, obs_mask=obs_mask)

    num_windows = int(traj.shape[0])
    total_points = int(diagnostics["total_points"])
    total_segments = int(diagnostics["total_segments"])
    boundary_violation_count = int(diagnostics["boundary_violation_count_by_window"].sum())
    internal_wall_crossing_count = int(diagnostics["internal_wall_crossing_count_by_window"].sum())
    door_valid_crossing_count = int(diagnostics["door_valid_crossing_count_by_window"].sum())
    infeasible_transition_count = int(diagnostics["infeasible_transition_count_by_window"].sum())
    window_violation_count = int(np.sum(diagnostics["window_has_violation"]))
    masked_segment_count = int(diagnostics["masked_segment_count"])
    masked_infeasible_transition_count = int(diagnostics["masked_infeasible_transition_count"])

    off_map_ratio = float(boundary_violation_count / total_points) if total_points > 0 else 0.0
    door_crossing_valid_ratio = (
        float(door_valid_crossing_count / internal_wall_crossing_count) if internal_wall_crossing_count > 0 else 0.0
    )
    window_violation_rate = float(window_violation_count / num_windows) if num_windows > 0 else 0.0
    infeasible_transition_rate = float(infeasible_transition_count / total_segments) if total_segments > 0 else 0.0
    mean_infeasible_transitions_per_window = float(infeasible_transition_count / num_windows) if num_windows > 0 else 0.0
    masked_infeasible_transition_rate = (
        float(masked_infeasible_transition_count / masked_segment_count) if masked_segment_count > 0 else np.nan
    )

    return {
        "geometry_profile": profile_name,
        "source_family": source_family,
        "experiment_id": experiment_id,
        "degradation": degradation,
        "coarse_method": coarse_method,
        "method_name": method_name,
        "refiner_name": refiner_name,
        "alpha": np.nan if alpha is None else float(alpha),
        "output_path": str(output_path),
        "evaluated_windows": num_windows,
        "off_map_ratio": off_map_ratio,
        "boundary_violation_count": boundary_violation_count,
        "internal_wall_crossing_count": internal_wall_crossing_count,
        "door_valid_crossing_count": door_valid_crossing_count,
        "door_crossing_valid_ratio": door_crossing_valid_ratio,
        "infeasible_transition_count": infeasible_transition_count,
        "window_violation_count": window_violation_count,
        "window_violation_rate": window_violation_rate,
        "infeasible_transition_rate": infeasible_transition_rate,
        "mean_infeasible_transitions_per_window": mean_infeasible_transitions_per_window,
        "masked_infeasible_transition_count": masked_infeasible_transition_count,
        "masked_segment_count": masked_segment_count,
        "masked_infeasible_transition_rate": masked_infeasible_transition_rate,
        "total_points": total_points,
        "total_segments": total_segments,
    }


def collect_phase1_rows(profile_name: str, profile, feasible_indices: np.ndarray, mask_lookup: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    if not PHASE1_BASELINES_DIR.exists():
        return rows

    for results_path in sorted(PHASE1_BASELINES_DIR.glob("*/*/results.npz")):
        experiment_id = results_path.parent.parent.name
        method_name = results_path.parent.name
        data = np.load(results_path, allow_pickle=False)
        if "traj_hat" not in data:
            continue
        traj = np.asarray(data["traj_hat"], dtype=np.float32)
        row = build_row(
            profile_name=profile_name,
            source_family="phase1_baseline",
            experiment_id=experiment_id,
            degradation="",
            coarse_method="",
            method_name=method_name,
            refiner_name="",
            alpha=None,
            output_path=results_path,
            traj=traj,
            feasible_indices=feasible_indices,
            profile=profile,
            obs_mask=mask_lookup.get(experiment_id),
        )
        rows.append(row)
    return rows


def collect_controlled_rows(profile_name: str, profile, feasible_indices: np.ndarray, controlled_mask: np.ndarray) -> list[dict]:
    rows = []
    if not CONTROLLED_RECON_DIR.exists():
        return rows

    for path in sorted(CONTROLLED_RECON_DIR.glob("recon_*.npy")):
        parsed = parse_controlled_reconstruction(path)
        if parsed is None:
            continue
        degradation, method_name, experiment_id = parsed
        traj = load_array(path).astype(np.float32)
        row = build_row(
            profile_name=profile_name,
            source_family="controlled_benchmark",
            experiment_id=experiment_id,
            degradation=degradation,
            coarse_method="",
            method_name=method_name,
            refiner_name="",
            alpha=None,
            output_path=path,
            traj=traj,
            feasible_indices=feasible_indices,
            profile=profile,
            obs_mask=controlled_mask,
        )
        rows.append(row)
    return rows


def collect_refinement_rows(
    profile_name: str,
    profile,
    feasible_indices: np.ndarray,
    controlled_mask: np.ndarray,
    source_dir: Path,
    source_family: str,
) -> list[dict]:
    rows = []
    if not source_dir.exists():
        return rows

    for path in sorted(source_dir.glob("refined_*.npy")):
        parsed = parse_refined_name(path)
        if parsed is None:
            continue
        degradation, coarse_method, refiner_name = parsed
        traj = load_array(path).astype(np.float32)
        row = build_row(
            profile_name=profile_name,
            source_family=source_family,
            experiment_id="span20_fixed_seed42" if source_family != "phase1_baseline" else "",
            degradation=degradation,
            coarse_method=coarse_method,
            method_name=refiner_name,
            refiner_name=refiner_name,
            alpha=alpha_from_refiner(refiner_name),
            output_path=path,
            traj=traj,
            feasible_indices=feasible_indices,
            profile=profile,
            obs_mask=controlled_mask,
        )
        rows.append(row)
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
    sum_cols = [
        "evaluated_windows",
        "boundary_violation_count",
        "internal_wall_crossing_count",
        "door_valid_crossing_count",
        "infeasible_transition_count",
        "window_violation_count",
        "masked_infeasible_transition_count",
        "masked_segment_count",
        "total_points",
        "total_segments",
    ]
    grouped = metrics_df.groupby(group_cols, dropna=False, as_index=False)[sum_cols].sum()
    grouped["off_map_ratio"] = grouped["boundary_violation_count"] / grouped["total_points"].clip(lower=1)
    grouped["door_crossing_valid_ratio"] = grouped["door_valid_crossing_count"] / grouped[
        "internal_wall_crossing_count"
    ].clip(lower=1)
    grouped["window_violation_rate"] = grouped["window_violation_count"] / grouped["evaluated_windows"].clip(lower=1)
    grouped["infeasible_transition_rate"] = grouped["infeasible_transition_count"] / grouped["total_segments"].clip(lower=1)
    grouped["mean_infeasible_transitions_per_window"] = grouped["infeasible_transition_count"] / grouped[
        "evaluated_windows"
    ].clip(lower=1)
    grouped["masked_infeasible_transition_rate"] = grouped["masked_infeasible_transition_count"] / grouped[
        "masked_segment_count"
    ].replace(0, np.nan)
    return grouped.sort_values(group_cols).reset_index(drop=True)


def plot_summary(summary_df: pd.DataFrame, output_path: Path):
    family_df = (
        summary_df.groupby("source_family", dropna=False, as_index=False)[
            [
                "evaluated_windows",
                "window_violation_count",
                "infeasible_transition_count",
                "total_segments",
            ]
        ]
        .sum()
    )
    family_df["window_violation_rate"] = family_df["window_violation_count"] / family_df["evaluated_windows"].clip(lower=1)
    family_df["infeasible_transition_rate"] = family_df["infeasible_transition_count"] / family_df["total_segments"].clip(lower=1)
    family_df["mean_infeasible_transitions_per_window"] = family_df["infeasible_transition_count"] / family_df[
        "evaluated_windows"
    ].clip(lower=1)

    metrics = [
        ("window_violation_rate", "window_violation_rate"),
        ("infeasible_transition_rate", "infeasible_transition_rate"),
        ("mean_infeasible_transitions_per_window", "mean_infeasible_transitions_per_window"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(family_df["source_family"], family_df[col], color="#4C78A8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("wall_door_v1 geometry violations on the feasible clean-target subset")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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


def build_report(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    filter_summary: pd.DataFrame,
    retention_rate: float,
    output_path: Path,
):
    by_family = (
        summary_df.groupby("source_family", dropna=False, as_index=False)[
            [
                "evaluated_windows",
                "boundary_violation_count",
                "internal_wall_crossing_count",
                "door_valid_crossing_count",
                "infeasible_transition_count",
                "window_violation_count",
                "masked_infeasible_transition_count",
                "masked_segment_count",
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
    by_family["window_violation_rate"] = by_family["window_violation_count"] / by_family["evaluated_windows"].clip(lower=1)
    by_family["infeasible_transition_rate"] = by_family["infeasible_transition_count"] / by_family["total_segments"].clip(lower=1)
    by_family["mean_infeasible_transitions_per_window"] = by_family["infeasible_transition_count"] / by_family[
        "evaluated_windows"
    ].clip(lower=1)
    by_family["masked_infeasible_transition_rate"] = by_family["masked_infeasible_transition_count"] / by_family[
        "masked_segment_count"
    ].replace(0, np.nan)

    worst_rows = summary_df.sort_values(
        ["window_violation_rate", "infeasible_transition_rate", "infeasible_transition_count"],
        ascending=[False, False, False],
    ).head(12)[
        [
            "source_family",
            "experiment_id",
            "degradation",
            "coarse_method",
            "method_name",
            "alpha",
            "window_violation_rate",
            "infeasible_transition_rate",
            "mean_infeasible_transitions_per_window",
        ]
    ]

    lines = [
        "# Stage 3 Geometry Extension Report",
        "",
        "## Scope",
        "",
        "The fixed `canonical_room3` benchmark remains unchanged.",
        "This report keeps `wall_door_v1` as a separate geometry feasibility extension rather than a replacement benchmark.",
        "It does not introduce a new reconstruction method and does not change the existing reconstruction metrics.",
        "",
        "## Geometry Profile",
        "",
        "- Room boundary: `[0, 3] x [0, 3]`",
        "- Internal wall: vertical wall at `x = 1.5`",
        "- Door opening: `y in [1.2, 1.8]`",
        "- A side-to-side crossing of `x = 1.5` is feasible only if the crossing point lies within the door interval.",
        "",
        "## Feasible Clean-Target Filtering",
        "",
        "Before evaluating reconstructed or refined trajectories, clean target windows are filtered under the synthetic wall-door layout.",
        "A clean window is retained only if:",
        "",
        "- `clean_off_map_ratio == 0`",
        "- `clean_infeasible_transition_count == 0`",
        "",
        "Infeasible clean windows are excluded from the main geometry evaluation but are not deleted from the repository.",
        "All main geometry metrics below are interpreted only on the feasible subset.",
        "",
        "## Filter Summary",
        "",
        markdown_table(filter_summary),
        "",
        "## Main Reported Metrics",
        "",
        "Large raw counts are secondary in this revised experiment.",
        "The primary geometry diagnostics are:",
        "",
        "- `window_violation_rate`",
        "- `infeasible_transition_rate`",
        "- `mean_infeasible_transitions_per_window`",
        "- `masked_infeasible_transition_rate` when a missing mask is available",
        "",
        "## Summary By Source Family",
        "",
        markdown_table(
            by_family[
                [
                    "source_family",
                    "evaluated_windows",
                    "window_violation_rate",
                    "infeasible_transition_rate",
                    "mean_infeasible_transitions_per_window",
                    "masked_infeasible_transition_rate",
                    "off_map_ratio",
                ]
            ]
        ),
        "",
        "## Highest-Violation Rows By Normalized Rate",
        "",
        markdown_table(worst_rows),
        "",
        "## Interpretation",
        "",
        "The revised `wall_door_v1` experiment is a feasibility diagnostic on a feasible clean-target subset, not a new reconstruction benchmark and not a new trajectory generator.",
        "Its purpose is to test whether existing Stage 3 outputs remain physically plausible under a simple wall-and-door indoor layout.",
        "",
        "## Limitation",
        "",
        "`wall_door_v1` is a synthetic feasibility stress test applied after scaling ETH+UCY trajectories into `canonical_room3`.",
        "Its violation counts should be interpreted as feasibility diagnostics under this artificial layout, not as direct evidence of real-room navigation failure.",
    ]

    if retention_rate < 0.10:
        lines.extend(
            [
                "",
                "## Warning",
                "",
                "The scaled ETH+UCY trajectories are poorly matched to the synthetic wall-door layout. The current `wall_door_v1` experiment should be treated as a preliminary feasibility stress test. A future geometry-controlled synthetic trajectory set may be needed.",
            ]
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `feasible_indices.npy`",
            "- `geometry_filter_summary.csv`",
            "- `geometry_metrics.csv`",
            "- `geometry_summary.csv`",
            "- `geometry_extension_report.md`",
            "- `figures/geometry_violation_summary.png`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manifest_limitation(manifest_path: Path):
    if not manifest_path.exists():
        return
    text = manifest_path.read_text(encoding="utf-8")
    old = (
        "## Geometry Extension Limitation\n\n"
        "`wall_door_v1` is a synthetic feasibility stress test applied after scaling ETH+UCY trajectories into `canonical_room3`.\n"
        "Its violation counts should be interpreted as feasibility diagnostics under this artificial layout, not as direct evidence of real-room navigation failure.\n"
    )
    new = (
        "## Geometry Extension Limitation\n\n"
        "`wall_door_v1` is a synthetic feasibility stress test applied after scaling ETH+UCY trajectories into `canonical_room3`.\n"
        "Its violation counts should be interpreted as feasibility diagnostics under this artificial layout, not as direct evidence of real-room navigation failure.\n"
        "In the revised experiment, the main geometry evaluation is computed only on the feasible clean-target subset, so normalized violation rates should be read as conditional diagnostics on that retained subset.\n"
    )
    if old in text:
        text = text.replace(old, new)
    elif "## Geometry Extension Limitation" not in text:
        text = text.rstrip() + "\n\n" + new
    manifest_path.write_text(text, encoding="utf-8")


def print_console_summary(
    summary_df: pd.DataFrame,
    filter_summary: pd.DataFrame,
    retention_rate: float,
):
    print("=" * 72)
    print("geometry_filter_summary.csv")
    print(filter_summary.to_csv(index=False).strip())
    print("-" * 72)
    print("geometry_summary.csv head")
    print(summary_df.head(12).to_csv(index=False).strip())
    print("-" * 72)
    print("Generated files")
    for path in [
        feasible_indices_path(),
        filter_summary_csv_path(),
        geometry_metrics_csv_path(),
        geometry_summary_csv_path(),
        report_path(),
        figure_path(),
    ]:
        print(path)
    print("-" * 72)
    print(f"retention_rate acceptable: {retention_rate >= 0.10}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 wall-door geometry extension evaluation on feasible clean targets.")
    parser.parse_args()

    ensure_dirs()
    profile = canonical_room3_wall_door_v1()

    clean_npz = np.load(PHASE1_CLEAN_PATH, allow_pickle=False)
    clean = np.asarray(clean_npz["traj_abs"], dtype=np.float32)

    feasible_indices, filter_summary, _ = compute_filter_summary(clean, profile)
    np.save(feasible_indices_path(), feasible_indices)
    filter_summary.to_csv(filter_summary_csv_path(), index=False)

    controlled_mask = np.load(CONTROLLED_MASK_PATH, allow_pickle=False).astype(np.uint8)
    mask_lookup = phase1_mask_lookup()

    rows = []
    rows.extend(collect_phase1_rows(profile.profile_name, profile, feasible_indices, mask_lookup))
    rows.extend(collect_controlled_rows(profile.profile_name, profile, feasible_indices, controlled_mask))
    rows.extend(
        collect_refinement_rows(
            profile.profile_name,
            profile,
            feasible_indices,
            controlled_mask,
            REFINEMENT_REFINED_DIR,
            "refinement",
        )
    )
    rows.extend(
        collect_refinement_rows(
            profile.profile_name,
            profile,
            feasible_indices,
            controlled_mask,
            ALPHA_SWEEP_REFINED_DIR,
            "refinement_alpha_sweep",
        )
    )

    if not rows:
        raise RuntimeError("No Stage 3 outputs found for geometry extension evaluation.")

    metrics_df = pd.DataFrame(rows)
    summary_df = build_summary(metrics_df)

    metrics_df.to_csv(geometry_metrics_csv_path(), index=False)
    summary_df.to_csv(geometry_summary_csv_path(), index=False)
    plot_summary(summary_df, figure_path())
    build_report(
        metrics_df,
        summary_df,
        filter_summary,
        float(filter_summary["retention_rate"].iloc[0]),
        report_path(),
    )

    DOCS_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path(), DOCS_ASSET_ROOT / "geometry_violation_summary.png")
    update_manifest_limitation(FIGURE_ASSET_MANIFEST)
    update_manifest_limitation(DOCS_FIGURE_MANIFEST)

    print_console_summary(summary_df, filter_summary, float(filter_summary["retention_rate"].iloc[0]))


if __name__ == "__main__":
    main()
