from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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

from tools.stage3.controlled.evaluate_coarse_reconstruction import (
    compute_geometry_metrics,
    compute_reconstruction_metrics,
    room3_map_meta,
)
from tools.stage3.refinement.ddpm_refiner import (
    ddpm_prior_interface_v0,
    ddpm_prior_masked_blend_v2,
)
from tools.stage3.refinement.run_refinement_interface import load_array
from utils.stage3.controlled_benchmark import (
    DEFAULT_SEED,
    DEFAULT_SPAN_MODE,
    DEFAULT_SPAN_RATIO,
    DEGRADATION_NAMES,
    METHODS,
    METHOD_LABELS,
    clean_path,
    experiment_tag,
    mask_path,
    reconstruction_path,
)

ALPHAS = [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

OUT_DIR = PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "alpha_sweep"
REFINED_DIR = OUT_DIR / "refined"
FIG_DIR = OUT_DIR / "figures"


def ensure_dirs():
    for path in [OUT_DIR, REFINED_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def alpha_slug(alpha: float) -> str:
    return f"alpha{alpha:.2f}"


def safe_improvement(coarse_value: float, refined_value: float) -> float:
    if abs(coarse_value) < 1e-12:
        return 0.0
    return float((coarse_value - refined_value) / coarse_value)


def metrics_csv_path() -> Path:
    return OUT_DIR / "alpha_sweep_metrics.csv"


def metrics_json_path() -> Path:
    return OUT_DIR / "alpha_sweep_metrics.json"


def summary_csv_path() -> Path:
    return OUT_DIR / "alpha_sweep_summary.csv"


def report_path() -> Path:
    return OUT_DIR / "alpha_sweep_report.md"


def figure_path(name: str) -> Path:
    return FIG_DIR / name


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


def write_metrics_csv(rows: list[dict], output_path: Path):
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_metrics_json(rows: list[dict], output_path: Path, config: dict):
    payload = {
        "config": config,
        "alphas": ALPHAS,
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby(["degradation", "coarse_method", "alpha"], as_index=False)[
            [
                "ADE",
                "FDE",
                "RMSE",
                "masked_ADE",
                "masked_RMSE",
                "off_map_ratio",
                "wall_crossing_count",
                "coarse_ADE",
                "coarse_masked_ADE",
                "improvement_ADE",
                "improvement_masked_ADE",
                "improves_over_identity",
                "improves_over_light_savgol",
            ]
        ]
        .mean()
    )
    return summary.sort_values(["degradation", "coarse_method", "alpha"]).reset_index(drop=True)


def plot_metric(summary_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path, *, reference_metric: str | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.ravel()

    for axis, degradation in zip(axes, DEGRADATION_NAMES):
        subset = summary_df[summary_df["degradation"] == degradation]
        for coarse_method in METHODS:
            method_subset = subset[subset["coarse_method"] == coarse_method].sort_values("alpha")
            axis.plot(
                method_subset["alpha"],
                method_subset[metric],
                marker="o",
                linewidth=2.0,
                label=METHOD_LABELS[coarse_method],
            )
            if reference_metric is not None and not method_subset.empty:
                axis.axhline(
                    float(method_subset[reference_metric].iloc[0]),
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.6,
                )

        axis.set_title(degradation)
        axis.set_xlabel("alpha")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_bool(value: bool) -> str:
    return "Yes" if value else "No"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for record in df.to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def build_report(metrics_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path):
    best_by_coarse = (
        summary_df.groupby(["coarse_method", "alpha"], as_index=False)["masked_ADE"]
        .mean()
        .sort_values(["coarse_method", "masked_ADE", "alpha"])
        .groupby("coarse_method", as_index=False)
        .first()
    )

    best_by_pair = (
        summary_df.sort_values(["degradation", "coarse_method", "masked_ADE", "alpha"])
        .groupby(["degradation", "coarse_method"], as_index=False)
        .first()
    )

    any_improve_over_identity = bool(metrics_df["improves_over_identity"].any())
    any_improve_over_light_savgol = bool(metrics_df["improves_over_light_savgol"].any())

    alpha_trend_lines = []
    for coarse_method in METHODS:
        coarse_subset = (
            summary_df.groupby(["coarse_method", "alpha"], as_index=False)["masked_ADE"]
            .mean()
        )
        coarse_subset = coarse_subset[coarse_subset["coarse_method"] == coarse_method].sort_values("alpha")
        diffs = np.diff(coarse_subset["masked_ADE"].to_numpy())
        is_non_decreasing = bool(np.all(diffs >= -1e-9))
        alpha_trend_lines.append(
            f"- {METHOD_LABELS[coarse_method]}: larger alpha consistently degrades masked_ADE = `{is_non_decreasing}`"
        )

    lines = [
        "# Stage 3 DDPM Masked Blend v2 Alpha Sweep",
        "",
        "## Purpose",
        "",
        "This sweep tests whether the fixed alpha=0.25 used by `ddpm_prior_masked_blend_v2` is appropriate.",
        "Observed points are preserved, while missing points are blended with the cached `ddpm_prior_interface_v0` candidate.",
        "",
        "## Questions",
        "",
        "1. Which alpha gives the best masked_ADE for each coarse method?",
        "2. Which alpha gives the best masked_ADE for each degradation/coarse_method pair?",
        "3. Does any alpha improve over identity/coarse?",
        "4. Does any alpha improve over light_savgol_refiner?",
        "5. Does larger alpha consistently degrade performance?",
        "6. What does this imply about the unconditional DDPM prior?",
        "",
        "## Best Alpha By Coarse Method",
        "",
        dataframe_to_markdown(best_by_coarse),
        "",
        "## Best Alpha By Degradation And Coarse Method",
        "",
        dataframe_to_markdown(
            best_by_pair[["degradation", "coarse_method", "alpha", "masked_ADE", "improvement_masked_ADE"]]
        ),
        "",
        "## Answers",
        "",
        "### 1. Best alpha by coarse method",
        "",
    ]

    for _, row in best_by_coarse.iterrows():
        lines.append(
            f"- {METHOD_LABELS[row['coarse_method']]}: best alpha = `{row['alpha']:.2f}` with mean masked_ADE = `{row['masked_ADE']:.6f}`"
        )

    lines.extend(
        [
            "",
            "### 2. Best alpha by degradation/coarse_method pair",
            "",
        ]
    )
    for _, row in best_by_pair.iterrows():
        lines.append(
            f"- {row['degradation']} / {METHOD_LABELS[row['coarse_method']]}: best alpha = `{row['alpha']:.2f}` with masked_ADE = `{row['masked_ADE']:.6f}`"
        )

    lines.extend(
        [
            "",
            "### 3. Does any alpha improve over identity/coarse?",
            "",
            f"- {format_bool(any_improve_over_identity)}",
            "",
            "### 4. Does any alpha improve over light_savgol_refiner?",
            "",
            f"- {format_bool(any_improve_over_light_savgol)}",
            "",
            "### 5. Does larger alpha consistently degrade performance?",
            "",
        ]
    )
    lines.extend(alpha_trend_lines)

    implication = (
        "The unconditional DDPM prior is not acting like a reliable direct missing-segment refiner here. "
        "If the best alpha stays near 0 and larger alpha worsens masked_ADE, the prior candidate is adding harmful trajectory content unless it is conditioned more tightly on the observed context and missing-mask structure."
    )
    lines.extend(
        [
            "",
            "### 6. Implication for the unconditional DDPM prior",
            "",
            implication,
            "",
            "## Output Files",
            "",
            "- `alpha_sweep_metrics.csv`",
            "- `alpha_sweep_metrics.json`",
            "- `alpha_sweep_summary.csv`",
            "- `alpha_sweep_report.md`",
            "- `figures/alpha_sweep_ADE.png`",
            "- `figures/alpha_sweep_masked_ADE.png`",
            "- `figures/alpha_sweep_improvement_masked_ADE.png`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(metrics_df: pd.DataFrame):
    best_by_coarse = (
        metrics_df.groupby(["coarse_method", "alpha"], as_index=False)["masked_ADE"]
        .mean()
        .sort_values(["coarse_method", "masked_ADE", "alpha"])
        .groupby("coarse_method", as_index=False)
        .first()
    )
    best_by_pair = (
        metrics_df.groupby(["degradation", "coarse_method", "alpha"], as_index=False)["masked_ADE"]
        .mean()
        .sort_values(["degradation", "coarse_method", "masked_ADE", "alpha"])
        .groupby(["degradation", "coarse_method"], as_index=False)
        .first()
    )
    any_identity = bool(metrics_df["improves_over_identity"].any())
    any_light_sg = bool(metrics_df["improves_over_light_savgol"].any())

    print("=" * 72)
    print("Best alpha by coarse_method using mean masked_ADE")
    for _, row in best_by_coarse.iterrows():
        print(
            f"{row['coarse_method']:30s} alpha={row['alpha']:.2f} "
            f"masked_ADE={row['masked_ADE']:.6f}"
        )

    print("-" * 72)
    print("Best alpha by degradation/coarse_method using masked_ADE")
    for _, row in best_by_pair.iterrows():
        print(
            f"{row['degradation']:20s} {row['coarse_method']:30s} "
            f"alpha={row['alpha']:.2f} masked_ADE={row['masked_ADE']:.6f}"
        )

    print("-" * 72)
    print(f"Any alpha improves over identity/coarse: {any_identity}")
    print(f"Any alpha improves over Light SG: {any_light_sg}")
    print(f"metrics_csv = {metrics_csv_path()}")
    print(f"metrics_json = {metrics_json_path()}")
    print(f"summary_csv = {summary_csv_path()}")
    print(f"report_path = {report_path()}")
    print(f"figure_ADE = {figure_path('alpha_sweep_ADE.png')}")
    print(f"figure_masked_ADE = {figure_path('alpha_sweep_masked_ADE.png')}")
    print(f"figure_improvement_masked_ADE = {figure_path('alpha_sweep_improvement_masked_ADE.png')}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Run alpha sweep for ddpm_prior_masked_blend_v2.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--span_ratio", type=float, default=DEFAULT_SPAN_RATIO)
    parser.add_argument("--span_mode", type=str, default=DEFAULT_SPAN_MODE, choices=["fixed", "random"])
    args = parser.parse_args()

    ensure_dirs()
    tag = experiment_tag(args.span_ratio, args.span_mode, args.seed)
    clean = load_array(clean_path()).astype(np.float32)
    obs_mask = load_array(mask_path(tag)).astype(np.uint8)
    map_meta = room3_map_meta()

    rows: list[dict] = []

    for degradation in DEGRADATION_NAMES:
        for coarse_method in METHODS:
            coarse = load_array(reconstruction_path(degradation, coarse_method, tag)).astype(np.float32)
            coarse_recon_metrics = compute_reconstruction_metrics(clean, coarse, obs_mask)
            coarse_geom_metrics = compute_geometry_metrics_fast(coarse, map_meta)
            ddpm_candidate, base_meta = ddpm_prior_interface_v0(coarse)

            identity_masked_ade = float(coarse_recon_metrics["masked_ADE"])
            light_savgol_masked_ade = None

            light_savgol_refined_path = (
                PROJECT_ROOT
                / "outputs"
                / "stage3"
                / "refinement"
                / "refined"
                / f"refined_{degradation}_{coarse_method}_light_savgol_refiner.npy"
            )
            if light_savgol_refined_path.exists():
                light_savgol = load_array(light_savgol_refined_path).astype(np.float32)
                light_savgol_metrics = compute_reconstruction_metrics(clean, light_savgol, obs_mask)
                light_savgol_masked_ade = float(light_savgol_metrics["masked_ADE"])
            else:
                light_savgol_masked_ade = float("nan")

            for alpha in ALPHAS:
                refined, metadata = ddpm_prior_masked_blend_v2(
                    coarse,
                    obs_mask=obs_mask,
                    ddpm_candidate_abs=ddpm_candidate,
                    alpha=alpha,
                    base_metadata=base_meta,
                )
                refiner_name = f"ddpm_prior_masked_blend_v2_{alpha_slug(alpha)}"
                output_path = REFINED_DIR / f"refined_{degradation}_{coarse_method}_{refiner_name}.npy"
                np.save(output_path, refined)

                recon_metrics = compute_reconstruction_metrics(clean, refined, obs_mask)
                geom_metrics = compute_geometry_metrics_fast(refined, map_meta)

                row = {
                    "degradation": degradation,
                    "coarse_method": coarse_method,
                    "alpha": float(alpha),
                    "refiner": refiner_name,
                    "output_path": str(output_path),
                    **recon_metrics,
                    **geom_metrics,
                    "coarse_ADE": float(coarse_recon_metrics["ADE"]),
                    "coarse_masked_ADE": identity_masked_ade,
                    "identity_masked_ADE": identity_masked_ade,
                    "light_savgol_masked_ADE": light_savgol_masked_ade,
                    "improvement_ADE": safe_improvement(coarse_recon_metrics["ADE"], recon_metrics["ADE"]),
                    "improvement_masked_ADE": safe_improvement(
                        coarse_recon_metrics["masked_ADE"], recon_metrics["masked_ADE"]
                    ),
                    "improves_over_identity": bool(recon_metrics["masked_ADE"] < identity_masked_ade),
                    "improves_over_light_savgol": bool(
                        np.isfinite(light_savgol_masked_ade) and recon_metrics["masked_ADE"] < light_savgol_masked_ade
                    ),
                    "coarse_off_map_ratio": float(coarse_geom_metrics["off_map_ratio"]),
                    "coarse_wall_crossing_count": int(coarse_geom_metrics["wall_crossing_count"]),
                    **metadata,
                }
                rows.append(row)

                print(
                    f"[done] {degradation:20s} {coarse_method:30s} alpha={alpha:.2f} "
                    f"masked_ADE={row['masked_ADE']:.6f} improve_masked_ADE={row['improvement_masked_ADE']:.6f}"
                )

    metrics_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(metrics_df)

    write_metrics_csv(rows, metrics_csv_path())
    write_metrics_json(
        rows,
        metrics_json_path(),
        {
            "seed": args.seed,
            "span_ratio": args.span_ratio,
            "span_mode": args.span_mode,
            "experiment_tag": tag,
        },
    )
    summary_df.to_csv(summary_csv_path(), index=False)

    plot_metric(summary_df, "ADE", "ADE", figure_path("alpha_sweep_ADE.png"), reference_metric="coarse_ADE")
    plot_metric(
        summary_df,
        "masked_ADE",
        "masked_ADE",
        figure_path("alpha_sweep_masked_ADE.png"),
        reference_metric="coarse_masked_ADE",
    )
    plot_metric(
        summary_df,
        "improvement_masked_ADE",
        "improvement_masked_ADE",
        figure_path("alpha_sweep_improvement_masked_ADE.png"),
        reference_metric=None,
    )

    build_report(metrics_df, summary_df, report_path())
    print_summary(metrics_df)


if __name__ == "__main__":
    main()
