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
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3.geometry_extension.geometry_profiles import (
    GeometryProfile,
    all_geometry_profiles,
    compute_geometry_window_diagnostics,
)
from tools.stage3.refinement.run_refinement_interface import load_array
from utils.stage3.controlled_benchmark import METHODS

DOCS_ROOT = PROJECT_ROOT / "docs"
DOCS_ASSET_ROOT = DOCS_ROOT / "assets" / "stage3"
GEOMETRY_OUT_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "geometry_extension"
FIGURE_ASSET_MANIFEST = PROJECT_ROOT / "outputs" / "stage3" / "figure_assets" / "figure_manifest.md"
DOCS_FIGURE_MANIFEST = DOCS_ASSET_ROOT / "figure_manifest.md"
PROTOCOL_DOC_PATH = DOCS_ROOT / "stage3_geometry_extension_protocol.md"

PHASE1_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3"
PHASE1_BASELINES_DIR = PHASE1_ROOT / "baselines"
PHASE1_EXPERIMENTS_DIR = PHASE1_ROOT / "data" / "experiments"
PHASE1_CLEAN_PATH = PHASE1_ROOT / "data" / "clean_windows_room3.npz"
CONTROLLED_ROOT = PROJECT_ROOT / "outputs" / "stage3" / "controlled_benchmark"
CONTROLLED_RECON_DIR = CONTROLLED_ROOT / "reconstruction"
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
PROFILE_MAIN_CONSTRAINT = {
    "obstacle_v1": "central rectangular obstacle [1.2, 1.8] x [1.2, 1.8]",
    "two_room_v1": "internal wall with narrow opening y in [1.35, 1.65]",
}
PROFILE_LAYOUT_DOC_TARGETS = {
    "obstacle_v1": DOCS_ASSET_ROOT / "obstacle_v1_layout.png",
    "two_room_v1": DOCS_ASSET_ROOT / "two_room_v1_layout.png",
}
PROFILE_SUMMARY_DOC_TARGETS = {
    "obstacle_v1": DOCS_ASSET_ROOT / "obstacle_v1_geometry_violation_summary.png",
    "two_room_v1": DOCS_ASSET_ROOT / "two_room_v1_geometry_violation_summary.png",
}


def ensure_dirs():
    GEOMETRY_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    for profile in all_geometry_profiles():
        (GEOMETRY_OUT_ROOT / profile.profile_name / "figures").mkdir(parents=True, exist_ok=True)


def profile_out_root(profile_name: str) -> Path:
    return GEOMETRY_OUT_ROOT / profile_name


def profile_fig_dir(profile_name: str) -> Path:
    return profile_out_root(profile_name) / "figures"


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


def subset_mask(mask: np.ndarray | None, feasible_indices: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    return np.asarray(mask, dtype=np.uint8)[feasible_indices]


def compute_filter_summary(clean: np.ndarray, profile: GeometryProfile) -> tuple[np.ndarray, pd.DataFrame]:
    diagnostics = compute_geometry_window_diagnostics(clean, profile)
    clean_off_map_ratio = np.asarray(diagnostics["clean_off_map_ratio_by_window"])
    clean_infeasible = np.asarray(diagnostics["infeasible_transition_count_by_window"])
    total_windows = int(clean.shape[0])

    if profile.profile_name == "obstacle_v1":
        clean_obstacle_points = np.asarray(diagnostics["obstacle_point_violation_count_by_window"])
        clean_obstacle_segments = np.asarray(diagnostics["obstacle_segment_crossing_count_by_window"])
        feasible_mask = (
            (clean_off_map_ratio == 0.0)
            & (clean_obstacle_points == 0)
            & (clean_obstacle_segments == 0)
            & (clean_infeasible == 0)
        )
        x = clean[:, :, 0]
        stays_left = np.all(x <= float(profile.obstacle_x_min), axis=1)
        stays_right = np.all(x >= float(profile.obstacle_x_max), axis=1)
        stays_below = np.all(clean[:, :, 1] <= float(profile.obstacle_y_min), axis=1)
        stays_above = np.all(clean[:, :, 1] >= float(profile.obstacle_y_max), axis=1)
        same_region = feasible_mask & (stays_left | stays_right | stays_below | stays_above)
        bypass = feasible_mask & ~same_region
        summary = pd.DataFrame(
            [
                {
                    "profile_name": profile.profile_name,
                    "total_windows": total_windows,
                    "feasible_windows": int(feasible_mask.sum()),
                    "discarded_windows": int(total_windows - feasible_mask.sum()),
                    "retention_rate": float(feasible_mask.mean()) if total_windows > 0 else 0.0,
                    "obstacle_free_same_region_windows": int(same_region.sum()),
                    "obstacle_bypass_windows": int(bypass.sum()),
                }
            ]
        )
    else:
        clean_internal = np.asarray(diagnostics["internal_wall_crossing_count_by_window"])
        clean_door_valid = np.asarray(diagnostics["door_valid_crossing_count_by_window"])
        feasible_mask = (clean_off_map_ratio == 0.0) & (clean_infeasible == 0)
        same_room = feasible_mask & (clean_internal == 0)
        transition = feasible_mask & (clean_internal > 0) & (clean_infeasible == 0)
        summary = pd.DataFrame(
            [
                {
                    "profile_name": profile.profile_name,
                    "total_windows": total_windows,
                    "feasible_windows": int(feasible_mask.sum()),
                    "discarded_windows": int(total_windows - feasible_mask.sum()),
                    "retention_rate": float(feasible_mask.mean()) if total_windows > 0 else 0.0,
                    "same_room_windows": int(same_room.sum()),
                    "room_transition_windows": int(transition.sum()),
                    "valid_crossing_windows": int((feasible_mask & (clean_door_valid > 0)).sum()),
                }
            ]
        )

    feasible_indices = np.flatnonzero(feasible_mask).astype(np.int64)
    return feasible_indices, summary


def build_row(
    *,
    profile: GeometryProfile,
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
    obs_mask: np.ndarray | None,
) -> dict:
    traj = np.asarray(traj, dtype=np.float32)[feasible_indices]
    obs_mask = subset_mask(obs_mask, feasible_indices)
    diagnostics = compute_geometry_window_diagnostics(traj, profile, obs_mask=obs_mask)

    num_windows = int(traj.shape[0])
    total_points = int(diagnostics["total_points"])
    total_segments = int(diagnostics["total_segments"])
    boundary_violation_count = int(np.asarray(diagnostics["boundary_violation_count_by_window"]).sum())
    infeasible_transition_count = int(np.asarray(diagnostics["infeasible_transition_count_by_window"]).sum())
    window_violation_count = int(np.sum(np.asarray(diagnostics["window_has_violation"])))
    masked_segment_count = int(diagnostics["masked_segment_count"])
    masked_infeasible_transition_count = int(diagnostics["masked_infeasible_transition_count"])

    row = {
        "geometry_profile": profile.profile_name,
        "source_family": source_family,
        "experiment_id": experiment_id,
        "degradation": degradation,
        "coarse_method": coarse_method,
        "method_name": method_name,
        "refiner_name": refiner_name,
        "alpha": np.nan if alpha is None else float(alpha),
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "evaluated_windows": num_windows,
        "off_map_ratio": float(boundary_violation_count / total_points) if total_points > 0 else 0.0,
        "boundary_violation_count": boundary_violation_count,
        "infeasible_transition_count": infeasible_transition_count,
        "window_violation_count": window_violation_count,
        "window_violation_rate": float(window_violation_count / num_windows) if num_windows > 0 else 0.0,
        "infeasible_transition_rate": float(infeasible_transition_count / total_segments) if total_segments > 0 else 0.0,
        "mean_infeasible_transitions_per_window": float(infeasible_transition_count / num_windows) if num_windows > 0 else 0.0,
        "masked_infeasible_transition_count": masked_infeasible_transition_count,
        "masked_segment_count": masked_segment_count,
        "masked_infeasible_transition_rate": (
            float(masked_infeasible_transition_count / masked_segment_count) if masked_segment_count > 0 else np.nan
        ),
        "total_points": total_points,
        "total_segments": total_segments,
    }

    if profile.profile_name == "two_room_v1":
        internal = int(np.asarray(diagnostics["internal_wall_crossing_count_by_window"]).sum())
        door_valid = int(np.asarray(diagnostics["door_valid_crossing_count_by_window"]).sum())
        row.update(
            {
                "internal_wall_crossing_count": internal,
                "door_valid_crossing_count": door_valid,
                "door_crossing_valid_ratio": float(door_valid / internal) if internal > 0 else 0.0,
            }
        )
    if profile.profile_name == "obstacle_v1":
        point_viol = int(np.asarray(diagnostics["obstacle_point_violation_count_by_window"]).sum())
        seg_viol = int(np.asarray(diagnostics["obstacle_segment_crossing_count_by_window"]).sum())
        row.update(
            {
                "obstacle_point_violation_count": point_viol,
                "obstacle_segment_crossing_count": seg_viol,
                "obstacle_point_violation_rate": float(point_viol / total_points) if total_points > 0 else 0.0,
                "obstacle_segment_crossing_rate": float(seg_viol / total_segments) if total_segments > 0 else 0.0,
            }
        )

    return row


def collect_phase1_rows(profile: GeometryProfile, feasible_indices: np.ndarray, mask_lookup: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    if not PHASE1_BASELINES_DIR.exists():
        return rows
    for results_path in sorted(PHASE1_BASELINES_DIR.glob("*/*/results.npz")):
        experiment_id = results_path.parent.parent.name
        method_name = results_path.parent.name
        data = np.load(results_path, allow_pickle=False)
        if "traj_hat" not in data:
            continue
        rows.append(
            build_row(
                profile=profile,
                source_family="phase1_baseline",
                experiment_id=experiment_id,
                degradation="",
                coarse_method="",
                method_name=method_name,
                refiner_name="",
                alpha=None,
                output_path=results_path,
                traj=np.asarray(data["traj_hat"], dtype=np.float32),
                feasible_indices=feasible_indices,
                obs_mask=mask_lookup.get(experiment_id),
            )
        )
    return rows


def collect_controlled_rows(profile: GeometryProfile, feasible_indices: np.ndarray, controlled_mask: np.ndarray) -> list[dict]:
    rows = []
    if not CONTROLLED_RECON_DIR.exists():
        return rows
    for path in sorted(CONTROLLED_RECON_DIR.glob("recon_*.npy")):
        parsed = parse_controlled_reconstruction(path)
        if parsed is None:
            continue
        degradation, method_name, experiment_id = parsed
        rows.append(
            build_row(
                profile=profile,
                source_family="controlled_benchmark",
                experiment_id=experiment_id,
                degradation=degradation,
                coarse_method="",
                method_name=method_name,
                refiner_name="",
                alpha=None,
                output_path=path,
                traj=load_array(path).astype(np.float32),
                feasible_indices=feasible_indices,
                obs_mask=controlled_mask,
            )
        )
    return rows


def collect_refinement_rows(
    profile: GeometryProfile,
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
        rows.append(
            build_row(
                profile=profile,
                source_family=source_family,
                experiment_id="span20_fixed_seed42",
                degradation=degradation,
                coarse_method=coarse_method,
                method_name=refiner_name,
                refiner_name=refiner_name,
                alpha=alpha_from_refiner(refiner_name),
                output_path=path,
                traj=load_array(path).astype(np.float32),
                feasible_indices=feasible_indices,
                obs_mask=controlled_mask,
            )
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
    sum_cols = [
        "evaluated_windows",
        "boundary_violation_count",
        "infeasible_transition_count",
        "window_violation_count",
        "masked_infeasible_transition_count",
        "masked_segment_count",
        "total_points",
        "total_segments",
    ]
    optional_sum_cols = [
        "internal_wall_crossing_count",
        "door_valid_crossing_count",
        "obstacle_point_violation_count",
        "obstacle_segment_crossing_count",
    ]
    present_cols = [col for col in optional_sum_cols if col in metrics_df.columns]
    grouped = metrics_df.groupby(group_cols, dropna=False, as_index=False)[sum_cols + present_cols].sum()
    grouped["off_map_ratio"] = grouped["boundary_violation_count"] / grouped["total_points"].clip(lower=1)
    grouped["window_violation_rate"] = grouped["window_violation_count"] / grouped["evaluated_windows"].clip(lower=1)
    grouped["infeasible_transition_rate"] = grouped["infeasible_transition_count"] / grouped["total_segments"].clip(lower=1)
    grouped["mean_infeasible_transitions_per_window"] = grouped["infeasible_transition_count"] / grouped["evaluated_windows"].clip(lower=1)
    grouped["masked_infeasible_transition_rate"] = grouped["masked_infeasible_transition_count"] / grouped["masked_segment_count"].replace(0, np.nan)
    if "internal_wall_crossing_count" in grouped.columns:
        grouped["door_crossing_valid_ratio"] = grouped["door_valid_crossing_count"] / grouped["internal_wall_crossing_count"].clip(lower=1)
    if "obstacle_point_violation_count" in grouped.columns:
        grouped["obstacle_point_violation_rate"] = grouped["obstacle_point_violation_count"] / grouped["total_points"].clip(lower=1)
        grouped["obstacle_segment_crossing_rate"] = grouped["obstacle_segment_crossing_count"] / grouped["total_segments"].clip(lower=1)
    return grouped.sort_values(group_cols).reset_index(drop=True)


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


def plot_layout(profile: GeometryProfile, output_path: Path):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.add_patch(
        Rectangle(
            (profile.x_min, profile.y_min),
            profile.x_max - profile.x_min,
            profile.y_max - profile.y_min,
            fill=False,
            linewidth=2.2,
            edgecolor="#1f2937",
        )
    )
    if profile.profile_name == "two_room_v1":
        wall_color = "#7c2d12"
        door_color = "#059669"
        ax.plot([profile.wall_x, profile.wall_x], [profile.y_min, profile.door_y_min], color="#9a3412", linewidth=4.0)
        ax.plot([profile.wall_x, profile.wall_x], [profile.door_y_max, profile.y_max], color="#9a3412", linewidth=4.0)
        ax.plot([profile.wall_x, profile.wall_x], [profile.door_y_min, profile.door_y_max], color="#10b981", linewidth=4.0)
        ax.plot([profile.wall_x - 0.08, profile.wall_x + 0.08], [profile.door_y_min, profile.door_y_min], color=door_color, linewidth=2.0)
        ax.plot([profile.wall_x - 0.08, profile.wall_x + 0.08], [profile.door_y_max, profile.door_y_max], color=door_color, linewidth=2.0)
        ax.annotate(
            "",
            xy=(float(profile.wall_x) + 0.18, float(profile.door_y_max)),
            xytext=(float(profile.wall_x) + 0.18, float(profile.door_y_min)),
            arrowprops=dict(arrowstyle="<->", color=door_color, linewidth=1.8),
        )
        ax.text(float(profile.wall_x) + 0.23, 0.5 * float(profile.door_y_min + profile.door_y_max), "opening", color=door_color, fontsize=9, va="center")
        ax.text(float(profile.wall_x) + 0.06, float(profile.door_y_max) + 0.04, f"y in [{profile.door_y_min:.2f}, {profile.door_y_max:.2f}]", fontsize=8.5, color=door_color)
        ax.text(float(profile.wall_x) - 0.62, 0.18, "wall x = 1.5", fontsize=8.5, color=wall_color)
        ax.text(0.18, 2.82, "narrower valid passage", fontsize=9, color=door_color)
    if profile.profile_name == "obstacle_v1":
        ax.add_patch(
            Rectangle(
                (float(profile.obstacle_x_min), float(profile.obstacle_y_min)),
                float(profile.obstacle_x_max - profile.obstacle_x_min),
                float(profile.obstacle_y_max - profile.obstacle_y_min),
                facecolor="#ef4444",
                alpha=0.55,
                edgecolor="#991b1b",
                linewidth=2.0,
            )
        )
        ax.text(float(profile.obstacle_x_min), float(profile.obstacle_y_max) + 0.06, "blocked obstacle", fontsize=9)
    ax.set_xlim(profile.x_min - 0.05, profile.x_max + 0.05)
    ax.set_ylim(profile.y_min - 0.05, profile.y_max + 0.05)
    ax.set_aspect("equal")
    ax.set_title(profile.profile_name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_profile_summary(profile_name: str, summary_df: pd.DataFrame, output_path: Path):
    family_df = (
        summary_df.groupby("source_family", dropna=False, as_index=False)[
            ["evaluated_windows", "window_violation_count", "infeasible_transition_count", "total_segments"]
        ]
        .sum()
    )
    family_df["window_violation_rate"] = family_df["window_violation_count"] / family_df["evaluated_windows"].clip(lower=1)
    family_df["infeasible_transition_rate"] = family_df["infeasible_transition_count"] / family_df["total_segments"].clip(lower=1)
    family_df["mean_infeasible_transitions_per_window"] = family_df["infeasible_transition_count"] / family_df["evaluated_windows"].clip(lower=1)

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
    fig.suptitle(f"{profile_name} geometry violations on the feasible clean-target subset")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cross_profile_summary(summary_df: pd.DataFrame, output_path: Path):
    family_order = ["phase1_baseline", "controlled_benchmark", "refinement", "refinement_alpha_sweep"]
    profile_order = ["obstacle_v1", "two_room_v1"]
    grouped = (
        summary_df.groupby(["geometry_profile", "source_family"], as_index=False)[["evaluated_windows", "window_violation_count"]].sum()
    )
    grouped["window_violation_rate"] = grouped["window_violation_count"] / grouped["evaluated_windows"].clip(lower=1)

    x = np.arange(len(profile_order))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for idx, family in enumerate(family_order):
        vals = []
        for profile_name in profile_order:
            row = grouped[(grouped["geometry_profile"] == profile_name) & (grouped["source_family"] == family)]
            vals.append(float(row["window_violation_rate"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=family)
    ax.set_xticks(x)
    ax.set_xticklabels(profile_order)
    ax.set_ylabel("window_violation_rate")
    ax.set_xlabel("geometry profile")
    ax.set_title("Geometry feasibility comparison across source families")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_profile_report(profile: GeometryProfile, summary_df: pd.DataFrame, filter_summary: pd.DataFrame, output_path: Path):
    by_family = (
        summary_df.groupby("source_family", dropna=False, as_index=False)[
            [
                "evaluated_windows",
                "window_violation_count",
                "infeasible_transition_count",
                "masked_infeasible_transition_count",
                "masked_segment_count",
                "total_points",
                "total_segments",
                "boundary_violation_count",
            ]
        ]
        .sum()
    )
    by_family["window_violation_rate"] = by_family["window_violation_count"] / by_family["evaluated_windows"].clip(lower=1)
    by_family["infeasible_transition_rate"] = by_family["infeasible_transition_count"] / by_family["total_segments"].clip(lower=1)
    by_family["mean_infeasible_transitions_per_window"] = by_family["infeasible_transition_count"] / by_family["evaluated_windows"].clip(lower=1)
    by_family["masked_infeasible_transition_rate"] = by_family["masked_infeasible_transition_count"] / by_family["masked_segment_count"].replace(0, np.nan)
    by_family["off_map_ratio"] = by_family["boundary_violation_count"] / by_family["total_points"].clip(lower=1)

    cols = [
        "source_family",
        "evaluated_windows",
        "window_violation_rate",
        "infeasible_transition_rate",
        "mean_infeasible_transitions_per_window",
        "masked_infeasible_transition_rate",
        "off_map_ratio",
    ]
    if profile.profile_name == "obstacle_v1":
        extra = (
            summary_df.groupby("source_family", dropna=False, as_index=False)[
                ["obstacle_point_violation_count", "obstacle_segment_crossing_count", "total_points", "total_segments"]
            ]
            .sum()
        )
        extra["obstacle_point_violation_rate"] = extra["obstacle_point_violation_count"] / extra["total_points"].clip(lower=1)
        extra["obstacle_segment_crossing_rate"] = extra["obstacle_segment_crossing_count"] / extra["total_segments"].clip(lower=1)
        by_family = by_family.merge(
            extra[["source_family", "obstacle_point_violation_rate", "obstacle_segment_crossing_rate"]],
            on="source_family",
            how="left",
        )
        cols.extend(["obstacle_point_violation_rate", "obstacle_segment_crossing_rate"])

    warning = ""
    if float(filter_summary["retention_rate"].iloc[0]) < 0.10:
        warning = (
            "\n## Warning\n\n"
            "The scaled ETH+UCY trajectories are poorly matched to this synthetic geometry profile. "
            "This profile should be treated only as a preliminary feasibility stress test. "
            "A future geometry-controlled synthetic trajectory set may be needed.\n"
        )

    lines = [
        "# Stage 3 Geometry Extension Report",
        "",
        "## Scope",
        "",
        "These profiles are geometry feasibility extensions.",
        "They do not replace `canonical_room3`.",
        "They do not define new reconstruction methods.",
        "They are synthetic feasibility stress tests.",
        "Clean target windows are first filtered for each geometry profile.",
        "Geometry metrics are interpreted only on the clean-target feasible subset.",
        "Normalized rates are emphasized over raw counts.",
        "",
        "## Geometry Profile",
        "",
        f"- `profile_name`: `{profile.profile_name}`",
        f"- Main constraint: {PROFILE_MAIN_CONSTRAINT[profile.profile_name]}",
        "- Room boundary: `[0, 3] x [0, 3]`",
        "",
        "## Filter Summary",
        "",
        markdown_table(filter_summary),
        "",
        "## Summary By Source Family",
        "",
        markdown_table(by_family[cols]),
        warning,
        "",
        "## Output Files",
        "",
        "- `feasible_indices.npy`",
        "- `geometry_filter_summary.csv`",
        "- `geometry_metrics.csv`",
        "- `geometry_summary.csv`",
        "- `geometry_extension_report.md`",
        f"- `figures/{profile.profile_name}_layout.png`",
        "- `figures/geometry_violation_summary.png`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_protocol_doc():
    text = """# Stage 3 Geometry Extension Protocol

## Purpose

This document defines Stage 3 geometry feasibility extensions for evaluation.

These profiles do not replace `canonical_room3`.
They do not define new reconstruction methods.
They are synthetic feasibility stress tests layered on top of already-generated trajectories.

## Relationship To Existing Stage 3 Assets

- `canonical_room3` remains the fixed Stage 3 Phase 1 reference benchmark.
- Stage 2 prior checkpoints remain unchanged.
- Controlled benchmark degradation definitions remain unchanged.
- The DDPM sampling formula remains unchanged.
- Existing reconstruction metrics remain unchanged.
- This extension adds geometry-feasibility evaluation layers only.

## Shared Evaluation Principle

For each geometry profile:

1. Clean target windows are first filtered for feasibility under that profile.
2. Existing reconstructed and refined outputs are evaluated only on the clean-target feasible subset.
3. Normalized rates are emphasized over raw counts.

Primary normalized metrics:

- `window_violation_rate`
- `infeasible_transition_rate`
- `mean_infeasible_transitions_per_window`
- `masked_infeasible_transition_rate` when a missing mask is available

## Geometry Profiles

### 1. `obstacle_v1`

- Room boundary: `[0, 3] x [0, 3]`
- Central blocked obstacle: `[1.2, 1.8] x [1.2, 1.8]`

### 2. `two_room_v1`

- Room boundary: `[0, 3] x [0, 3]`
- Internal wall: vertical wall at `x = 1.5`
- Narrow transition opening: `y in [1.35, 1.65]`

## Interpretation

- These profiles are geometry feasibility extensions.
- They do not replace `canonical_room3`.
- They do not define new reconstruction methods.
- They are synthetic feasibility stress tests.
- Clean target windows are first filtered for each geometry profile.
- Geometry metrics are interpreted only on the clean-target feasible subset.
- Normalized rates are emphasized over raw counts.

## Limitation

If a profile has `retention_rate < 0.10`, do not make strong conclusions.

Use this warning:

\"The scaled ETH+UCY trajectories are poorly matched to this synthetic geometry profile. This profile should be treated only as a preliminary feasibility stress test. A future geometry-controlled synthetic trajectory set may be needed.\"
"""
    PROTOCOL_DOC_PATH.write_text(text, encoding="utf-8")


def build_figure_manifest():
    text = """# Figure Manifest

## Figures for first full review

### 1. Problem and protocol

- `overall_stage3_objective.png`
- `missing_reconstruction_task.png`
- `refinement_interface_v0_v1_v2.png`
- `obstacle_v1_layout.png`
- `two_room_v1_layout.png`

### 2. Benchmark evidence

- `full_vs_masked_comparison.png`
- `random_span_masked_ADE_mean_std.png`

### 3. Controlled coarse reconstruction

- `controlled_degradation_examples.png`
- `controlled_benchmark_metric_summary.png`

### 4. DDPM prior refinement

- `refinement_v0_v1_v2_comparison.png`
- `alpha_sweep_masked_ADE.png`
- `alpha_sweep_improvement_masked_ADE.png`

### 5. Geometry extension

- `obstacle_v1_geometry_violation_summary.png`
- `two_room_v1_geometry_violation_summary.png`
- `geometry_profiles_comparison.png`

| Figure filename | Type | Source files used | Purpose | Status | Short interpretation |
| --- | --- | --- | --- | --- | --- |
| overall_stage3_objective.png | conceptual | programmatic schematic | Show the Stage 3 reconstruction/refinement objective. | generated | Stage 3 is missing indoor trajectory reconstruction, not generic forecasting. |
| missing_reconstruction_task.png | conceptual | programmatic schematic | Show the one contiguous missing-segment task definition. | generated | Missing-segment reconstruction quality should be read against the clean target. |
| refinement_interface_v0_v1_v2.png | conceptual | programmatic schematic | Explain v0/v1/v2 DDPM refinement interfaces. | generated | v0 changes the whole trajectory; v1/v2 protect observed points. |
| obstacle_v1_layout.png | conceptual | `tools/stage3/geometry_extension/run_geometry_extension.py` | Explain the obstacle geometry extension layout. | generated | `obstacle_v1` adds a central blocked region as a feasibility stress test. |
| two_room_v1_layout.png | conceptual | `tools/stage3/geometry_extension/run_geometry_extension.py` | Explain the narrow-opening room-transition layout. | generated | `two_room_v1` tightens room-transition feasibility without changing the benchmark itself. |
| full_vs_masked_comparison.png | data-result | `outputs/stage3/phase1/canonical_room3/random_span_statistics/figures/full_vs_masked_comparison.png` | Show full vs masked metric ranking differences. | copied | Full-trajectory consistency and missing-segment reconstruction quality can rank methods differently. |
| random_span_masked_ADE_mean_std.png | data-result | `outputs/stage3/phase1/canonical_room3/random_span_statistics/metrics_summary_mean_std.csv` | Show mean +- std masked_ADE over random span positions. | generated | Masked metrics are the direct view of missing-segment reconstruction quality. |
| controlled_degradation_examples.png | data-result | `outputs/stage3/controlled_benchmark/degradation/clean.npy`<br>`outputs/stage3/controlled_benchmark/degradation/mask_span20_fixed_seed42.npy`<br>`outputs/stage3/controlled_benchmark/degradation/degraded_missing_only_span20_fixed_seed42.npy`<br>`outputs/stage3/controlled_benchmark/degradation/degraded_missing_noise_span20_fixed_seed42.npy`<br>`outputs/stage3/controlled_benchmark/degradation/degraded_missing_drift_span20_fixed_seed42.npy`<br>`outputs/stage3/controlled_benchmark/degradation/degraded_missing_noise_drift_span20_fixed_seed42.npy` | Show the four controlled degradation settings. | generated | The controlled benchmark stresses reconstruction under missingness, noise, drift, and combined degradation. |
| controlled_benchmark_metric_summary.png | data-result | `outputs/stage3/controlled_benchmark/eval/metrics_summary.csv` | Summarize controlled coarse reconstruction metrics. | generated | Baseline behavior changes across degradation types; masked_ADE emphasizes the missing segment. |
| refinement_v0_v1_v2_comparison.png | data-result | `outputs/stage3/refinement/eval/refinement_metrics.csv` | Compare identity, Light SG, DDPM v0, DDPM v1, and DDPM v2 alpha=0.25. | generated | v0 proves integration, v1 protects observed points, and v2 blends the DDPM candidate into the missing span. |
| alpha_sweep_masked_ADE.png | data-result | `outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv` | Show alpha sensitivity by coarse method. | generated | Linear and Savitzky-Golay prefer alpha=0.00, while Kalman benefits only from very small alpha. |
| alpha_sweep_improvement_masked_ADE.png | data-result | `outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv` | Show masked_ADE improvement relative to alpha=0.00. | generated | Large alpha usually hurts, which indicates the unconditional DDPM prior is not a reliable direct refiner. |
| obstacle_v1_geometry_violation_summary.png | data-result | `outputs/stage3/geometry_extension/obstacle_v1/geometry_summary.csv` | Summarize geometry feasibility violations under `obstacle_v1`. | generated | `obstacle_v1` checks whether trajectories remain feasible around a blocked central region. |
| two_room_v1_geometry_violation_summary.png | data-result | `outputs/stage3/geometry_extension/two_room_v1/geometry_summary.csv` | Summarize geometry feasibility violations under `two_room_v1`. | generated | `two_room_v1` tests whether outputs can pass through a narrower valid transition opening. |
| geometry_profiles_comparison.png | data-result | `outputs/stage3/geometry_extension/geometry_profiles_summary.csv` | Compare profile-level window violation rates by source family. | generated | Cross-profile normalized rates show how geometry strictness changes feasibility diagnostics. |

## Missing data sources

- None

## TODO

- `input_degraded` is not present in `metrics_summary.csv`, so it remains omitted from plots that depend on that table.

## Geometry Extension Limitation

These profiles are synthetic feasibility stress tests applied after scaling ETH+UCY trajectories into `canonical_room3`.
Their violation counts should be interpreted as feasibility diagnostics under artificial layouts, not as direct evidence of real-room navigation failure.
The main geometry evaluation is computed only on the feasible clean-target subset, so normalized violation rates should be read as conditional diagnostics on that retained subset.
"""
    FIGURE_ASSET_MANIFEST.write_text(text, encoding="utf-8")
    DOCS_FIGURE_MANIFEST.write_text(text, encoding="utf-8")


def build_cross_profile_summary(profile_summaries: list[pd.DataFrame], filter_summaries: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_summary = pd.concat(profile_summaries, ignore_index=True)
    filters = pd.concat(filter_summaries, ignore_index=True)
    filter_cols = ["profile_name", "total_windows", "feasible_windows", "discarded_windows", "retention_rate"]
    filter_lookup = filters[filter_cols].drop_duplicates()
    family_summary = (
        combined_summary.groupby(["geometry_profile", "source_family"], as_index=False)[
            ["evaluated_windows", "window_violation_count", "infeasible_transition_count", "total_segments"]
        ]
        .sum()
    )
    family_summary["window_violation_rate"] = family_summary["window_violation_count"] / family_summary["evaluated_windows"].clip(lower=1)
    family_summary["infeasible_transition_rate"] = family_summary["infeasible_transition_count"] / family_summary["total_segments"].clip(lower=1)
    family_summary["mean_infeasible_transitions_per_window"] = family_summary["infeasible_transition_count"] / family_summary["evaluated_windows"].clip(lower=1)
    family_summary = family_summary.merge(filter_lookup, left_on="geometry_profile", right_on="profile_name", how="left").drop(columns=["profile_name"])
    family_summary["main_constraint_type"] = family_summary["geometry_profile"].map(PROFILE_MAIN_CONSTRAINT)
    family_summary = family_summary[
        [
            "geometry_profile",
            "total_windows",
            "feasible_windows",
            "discarded_windows",
            "retention_rate",
            "main_constraint_type",
            "source_family",
            "window_violation_rate",
            "infeasible_transition_rate",
            "mean_infeasible_transitions_per_window",
        ]
    ].sort_values(["geometry_profile", "source_family"]).reset_index(drop=True)
    profile_level = filters.assign(main_constraint_type=filters["profile_name"].map(PROFILE_MAIN_CONSTRAINT))
    return family_summary, profile_level


def build_cross_profile_markdown(summary_df: pd.DataFrame, output_path: Path):
    lines = [
        "# Geometry Profiles Summary",
        "",
        "This table compares the three Stage 3 geometry feasibility extensions on the clean-target feasible subset.",
        "",
        markdown_table(summary_df),
        "",
        "Normalized rates are the main interpretation layer. Raw counts remain in per-profile CSV outputs.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_final_console_summary(
    obstacle_filter_summary: pd.DataFrame,
    two_room_filter_summary: pd.DataFrame,
    cross_profile_summary: pd.DataFrame,
    generated_figure_paths: list[Path],
    missing_sources: list[str],
):
    print("1. obstacle_v1 geometry_filter_summary.csv")
    print(obstacle_filter_summary.to_csv(index=False).strip())
    print("2. two_room_v1 geometry_filter_summary.csv")
    print(two_room_filter_summary.to_csv(index=False).strip())
    print("3. geometry_profiles_summary.csv")
    print(cross_profile_summary.to_csv(index=False).strip())
    print("4. generated figure paths")
    for path in generated_figure_paths:
        print(path.as_posix())
    print("5. missing sources or TODOs")
    if missing_sources:
        for item in missing_sources:
            print(item)
    else:
        print("None")
    print("6. git status --short")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 geometry feasibility extension evaluation across profiles.")
    parser.parse_args()

    ensure_dirs()
    build_protocol_doc()

    clean = np.asarray(np.load(PHASE1_CLEAN_PATH, allow_pickle=False)["traj_abs"], dtype=np.float32)
    controlled_mask = np.load(CONTROLLED_MASK_PATH, allow_pickle=False).astype(np.uint8)
    mask_lookup = phase1_mask_lookup()

    missing_sources: list[str] = []
    profile_summary_frames: list[pd.DataFrame] = []
    filter_summary_frames: list[pd.DataFrame] = []
    generated_figure_paths: list[Path] = []
    retained_filter_summaries: dict[str, pd.DataFrame] = {}

    for profile in all_geometry_profiles():
        out_root = profile_out_root(profile.profile_name)
        fig_dir = profile_fig_dir(profile.profile_name)

        feasible_indices, filter_summary = compute_filter_summary(clean, profile)
        filter_summary_frames.append(filter_summary)
        retained_filter_summaries[profile.profile_name] = filter_summary
        np.save(out_root / "feasible_indices.npy", feasible_indices)
        filter_summary.to_csv(out_root / "geometry_filter_summary.csv", index=False)

        rows = []
        rows.extend(collect_phase1_rows(profile, feasible_indices, mask_lookup))
        rows.extend(collect_controlled_rows(profile, feasible_indices, controlled_mask))
        rows.extend(collect_refinement_rows(profile, feasible_indices, controlled_mask, REFINEMENT_REFINED_DIR, "refinement"))
        rows.extend(
            collect_refinement_rows(profile, feasible_indices, controlled_mask, ALPHA_SWEEP_REFINED_DIR, "refinement_alpha_sweep")
        )

        if not rows:
            missing_sources.append(f"{profile.profile_name}: no Stage 3 outputs found for evaluation")
            continue

        metrics_df = pd.DataFrame(rows)
        summary_df = build_summary(metrics_df)
        profile_summary_frames.append(summary_df)

        metrics_df.to_csv(out_root / "geometry_metrics.csv", index=False)
        summary_df.to_csv(out_root / "geometry_summary.csv", index=False)

        layout_path = fig_dir / f"{profile.profile_name}_layout.png"
        summary_fig_path = fig_dir / "geometry_violation_summary.png"
        plot_layout(profile, layout_path)
        plot_profile_summary(profile.profile_name, summary_df, summary_fig_path)
        build_profile_report(profile, summary_df, filter_summary, out_root / "geometry_extension_report.md")

        shutil.copy2(layout_path, PROFILE_LAYOUT_DOC_TARGETS[profile.profile_name])
        shutil.copy2(summary_fig_path, PROFILE_SUMMARY_DOC_TARGETS[profile.profile_name])
        generated_figure_paths.extend(
            [
                layout_path,
                summary_fig_path,
                PROFILE_LAYOUT_DOC_TARGETS[profile.profile_name],
                PROFILE_SUMMARY_DOC_TARGETS[profile.profile_name],
            ]
        )

    if not profile_summary_frames:
        raise RuntimeError("No geometry profile outputs were generated.")

    geometry_profiles_summary_csv = GEOMETRY_OUT_ROOT / "geometry_profiles_summary.csv"
    geometry_profiles_summary_md = GEOMETRY_OUT_ROOT / "geometry_profiles_summary.md"
    family_summary_df, _ = build_cross_profile_summary(profile_summary_frames, filter_summary_frames)
    family_summary_df.to_csv(geometry_profiles_summary_csv, index=False)
    build_cross_profile_markdown(family_summary_df, geometry_profiles_summary_md)
    cross_profile_fig = DOCS_ASSET_ROOT / "geometry_profiles_comparison.png"
    plot_cross_profile_summary(pd.concat(profile_summary_frames, ignore_index=True), cross_profile_fig)
    generated_figure_paths.append(cross_profile_fig)

    build_figure_manifest()

    print_final_console_summary(
        retained_filter_summaries["obstacle_v1"],
        retained_filter_summaries["two_room_v1"],
        family_summary_df,
        generated_figure_paths,
        missing_sources,
    )


if __name__ == "__main__":
    main()
