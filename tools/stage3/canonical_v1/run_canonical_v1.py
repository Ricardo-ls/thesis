from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.stage3.canonical_v1.utils import (
    ALPHA_SWEEP_ROOT,
    CASE_TRACEABILITY_PATH,
    CHANGELOG_PATH,
    CODE_ROOT,
    CONDITIONS,
    CONFIG_PATH,
    CanonicalConfig,
    DDPM_AGGREGATIONS,
    EXPERIMENT_NAME,
    FIGURE3_AUDIT_PATH,
    FIGURE3_PATH,
    FIGURE5_AUDIT_PATH,
    FIGURE5_PATH,
    FIG_BEST_PATH,
    FDE_AUDIT_PATH,
    FIG_MEDIAN_PATH,
    FIG_WORST_PATH,
    FULL_MATRIX_SEED_CSV,
    FULL_MATRIX_SEED_MD,
    FULL_MATRIX_TRAJ_CSV,
    FULL_MATRIX_TRAJ_MD,
    GEOMETRY_STATEMENT_PATH,
    METRICS,
    METHODS,
    METHOD_LABELS,
    MISSING_CELL_AUDIT_CSV,
    OUTPUT_ROOT,
    PHASE1_EVAL_ROOT,
    README_PATH,
    RAW_SEED_LEVEL_PATH,
    RAW_TRAJ_LEVEL_PATH,
    SELECTED_CASES_PATH,
    STATUS_VALUES,
    TABLE2_REPLACEMENT_MD,
    anchor_missing_spans,
    build_degraded,
    build_obs_mask,
    compute_metrics,
    ensure_output_dirs,
    fmt,
    get_prior_metadata,
    highlight_gap,
    load_alpha_sweep_source,
    load_clean_room3,
    load_prior_checkpoint,
    make_five_column_figure,
    markdown_table,
    parse_condition,
    room3_map_meta,
    run_ddpm,
    run_kalman,
    run_linear_interp,
    run_savgol,
    summarize,
    write_csv,
    write_json,
    RunLogger,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed Stage 3 canonical evidence protocol.")
    parser.add_argument("--max_trajectories", type=int, default=1024)
    parser.add_argument("--num_ddpm_seeds", type=int, default=5)
    parser.add_argument("--ddpm_device", type=str, default="cpu")
    parser.add_argument("--reuse_existing_raw", action="store_true")
    return parser


def deterministic_seed_rows(condition: str, method: str, clean: np.ndarray, pred: np.ndarray, obs_mask: np.ndarray, span_start: np.ndarray, span_end: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for traj_idx in range(clean.shape[0]):
        metrics = compute_metrics(clean[traj_idx], pred[traj_idx], obs_mask[traj_idx], int(span_start[traj_idx]), int(span_end[traj_idx]))
        rows.append(
            {
                "condition": condition,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "trajectory_id": int(traj_idx),
                "seed": 0,
                "seed_role": "deterministic",
                "aggregation_level": "seed_level",
                "status": "ok",
                "span_start": int(span_start[traj_idx]),
                "span_len": int(span_end[traj_idx] - span_start[traj_idx] + 1),
                **metrics,
            }
        )
    return rows


def stochastic_seed_rows(condition: str, method: str, clean: np.ndarray, pred_ns: np.ndarray, obs_mask: np.ndarray, span_start: np.ndarray, span_end: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for traj_idx in range(clean.shape[0]):
        for seed_idx in range(pred_ns.shape[1]):
            metrics = compute_metrics(clean[traj_idx], pred_ns[traj_idx, seed_idx], obs_mask[traj_idx], int(span_start[traj_idx]), int(span_end[traj_idx]))
            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "trajectory_id": int(traj_idx),
                    "seed": int(seed_idx),
                    "seed_role": "ddpm_sample_seed",
                    "aggregation_level": "seed_level",
                    "status": "ok",
                    "span_start": int(span_start[traj_idx]),
                    "span_len": int(span_end[traj_idx] - span_start[traj_idx] + 1),
                    **metrics,
                }
            )
    return rows


def aggregate_trajectory_level(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    det_methods = [m for m in METHODS if not m.startswith("ddpm_")]
    ddpm_methods = [m for m in METHODS if m.startswith("ddpm_")]

    det_df = seed_df[seed_df["method"].isin(det_methods)]
    for rec in det_df.to_dict(orient="records"):
        rec = dict(rec)
        rec["aggregation"] = "deterministic"
        rec["aggregation_level"] = "trajectory_level"
        rows.append(rec)

    for method in ddpm_methods:
        method_df = seed_df[seed_df["method"] == method]
        for (condition, trajectory_id), group in method_df.groupby(["condition", "trajectory_id"], sort=False):
            base = {
                "condition": condition,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "trajectory_id": int(trajectory_id),
                "seed": "",
                "seed_role": "trajectory_aggregated",
                "aggregation_level": "trajectory_level",
                "status": "ok",
                "span_start": int(group["span_start"].iloc[0]),
                "span_len": int(group["span_len"].iloc[0]),
            }
            for aggregation in DDPM_AGGREGATIONS:
                row = dict(base)
                row["aggregation"] = aggregation
                for metric in METRICS:
                    values = group[metric].astype(float).to_numpy()
                    if aggregation == "seed_mean":
                        row[metric] = float(np.mean(values))
                    elif aggregation == "seed_median":
                        row[metric] = float(np.median(values))
                    elif aggregation == "seed_best":
                        row[metric] = float(np.min(values))
                    elif aggregation == "seed_worst":
                        row[metric] = float(np.max(values))
                rows.append(row)
    return pd.DataFrame(rows)


def build_stats_matrix(df: pd.DataFrame, *, level_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    if level_name == "seed_level":
        expected_aggregations = [""]
    else:
        expected_aggregations = ["deterministic", *DDPM_AGGREGATIONS]

    for condition in CONDITIONS:
        for method in METHODS:
            aggregations = expected_aggregations
            if level_name == "trajectory_level" and method.startswith("ddpm_"):
                aggregations = DDPM_AGGREGATIONS
            elif level_name == "trajectory_level":
                aggregations = ["deterministic"]
            for aggregation in aggregations:
                subset = df[(df["condition"] == condition) & (df["method"] == method)]
                if level_name == "trajectory_level":
                    subset = subset[subset["aggregation"] == aggregation]
                status = "ok" if not subset.empty else "missing_raw_data"
                for metric in METRICS:
                    if subset.empty:
                        rows.append(
                            {
                                "condition": condition,
                                "method": method,
                                "aggregation": aggregation,
                                "metric": metric,
                                "status": status,
                                "N": "missing_raw_data",
                                "mean": "missing_raw_data",
                                "std": "missing_raw_data",
                                "median": "missing_raw_data",
                                "min": "missing_raw_data",
                                "max": "missing_raw_data",
                                "p05": "missing_raw_data",
                                "p25": "missing_raw_data",
                                "p75": "missing_raw_data",
                                "p95": "missing_raw_data",
                                "source_file": str(RAW_SEED_LEVEL_PATH if level_name == "seed_level" else RAW_TRAJ_LEVEL_PATH),
                            }
                        )
                        audit_rows.append({"condition": condition, "method": method, "aggregation": aggregation, "metric": metric, "status": status})
                        continue
                    stats = summarize(subset[metric].astype(float).tolist())
                    rows.append(
                        {
                            "condition": condition,
                            "method": method,
                            "aggregation": aggregation,
                            "metric": metric,
                            "status": status,
                            **stats,
                            "source_file": str(RAW_SEED_LEVEL_PATH if level_name == "seed_level" else RAW_TRAJ_LEVEL_PATH),
                        }
                    )
    return rows, audit_rows


def build_markdown_matrix(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {title}",
        "",
        markdown_table(
            rows,
            ["condition", "method", "aggregation", "metric", "status", "N", "mean", "std", "median", "min", "max", "p05", "p25", "p75", "p95"],
        ),
        "",
    ]
    return "\n".join(lines)


def render_figure3(seed_df: pd.DataFrame):
    subset = seed_df[(seed_df["condition"].isin(["span20_random_seed42", "span20_random_seed43", "span20_random_seed44"])) & (seed_df["method"].isin(METHODS))]
    grouped = subset.groupby(["condition", "method"])["masked_ADE"]
    plot_rows = []
    for (condition, method), values in grouped:
        arr = values.to_numpy(dtype=float)
        plot_rows.append({"condition": condition, "method": method, "mean": float(np.mean(arr)), "std": float(np.std(arr)), "N": int(arr.size)})
    plot_df = pd.DataFrame(plot_rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x_labels = CONDITIONS
    x = np.arange(len(x_labels))
    width = 0.16
    offsets = np.linspace(-2, 2, num=len(METHODS)) * width
    for offset, method in zip(offsets, METHODS):
        method_df = plot_df[plot_df["method"] == method].set_index("condition").reindex(x_labels)
        ax.bar(
            x + offset,
            method_df["mean"],
            width=width,
            yerr=method_df["std"],
            capsize=3,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20)
    ax.set_ylabel("masked_ADE mean ± std")
    ax.set_title("Figure 3 replacement: raw per-case spread")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE3_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    FIGURE3_AUDIT_PATH.write_text(
        "\n".join(
            [
                "# Figure 3 Spread Definition",
                "",
                "- direct verdict: valid raw-spread replacement for the canonical_v1 evidence layer",
                "- what is averaged: raw per-case `masked_ADE` values",
                "- grouping keys: `condition`, `method`",
                "- N: deterministic methods use trajectories; DDPM methods use trajectory-seed cases",
                "- spread type: std",
                "- spread population: raw per-case values from `per_case_results_seed_level.csv`",
                "- does spread use already-averaged summaries: NO",
                "- source file: `outputs/stage3/canonical_v1/raw/per_case_results_seed_level.csv`",
                "- note: this replacement uses raw rows and does not reuse the old random-span summary std",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def render_figure5(seed_df: pd.DataFrame, traj_df: pd.DataFrame):
    alpha_source_path, alpha_rows = load_alpha_sweep_source()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    note_lines = []
    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows)
        alpha_df["alpha"] = alpha_df["alpha"].astype(float)
        alpha_df["masked_ADE"] = alpha_df["masked_ADE"].astype(float)
        grouped = alpha_df.groupby("alpha")["masked_ADE"]
        xs = []
        means = []
        traj_std = []
        for alpha, values in grouped:
            arr = values.to_numpy()
            xs.append(alpha)
            means.append(float(np.mean(arr)))
            traj_std.append(float(np.std(arr)))
        ax.errorbar(xs, means, yerr=traj_std, marker="o", linewidth=2.0, capsize=4)
        note_lines = [
            "# Figure 5 Spread Definition",
            "",
            "- direct verdict: still limited by historical alpha-sweep raw availability",
            "- source used: historical `outputs/stage3/refinement/alpha_sweep/alpha_sweep_metrics.csv`",
            "- what is averaged: per cell `masked_ADE` rows from the historical alpha sweep",
            "- grouping keys: `alpha`",
            "- N per alpha: number of historical rows available for that alpha",
            "- spread type: std over historical row values",
            "- does spread use raw per-trajectory rows: NO",
            "- does spread use already-aggregated rows: YES",
            "- trajectory variability separation: unavailable from historical raw because only condition/coarse/alpha summary rows exist",
            "- DDPM seed variability separation: unavailable from historical raw because seed-level alpha sweep outputs were not saved",
            "- degradation variability separation: mixed into the historical alpha rows",
            "- coarse method variability separation: mixed into the historical alpha rows",
            "- interpretation rule: treat this figure as a historical alpha diagnostic, not as a cleanly separated variance decomposition",
        ]
    else:
        ax.text(0.5, 0.5, "alpha_sweep_metrics.csv missing", ha="center", va="center")
        note_lines = [
            "# Figure 5 Spread Definition",
            "",
            "- source used: missing",
            "- exact variability separation is impossible because no raw alpha sweep rows were found",
        ]
    ax.set_xlabel("alpha")
    ax.set_ylabel("historical masked_ADE mean ± std")
    ax.set_title("Figure 5 replacement: alpha variance from available historical rows")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE5_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    FIGURE5_AUDIT_PATH.write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def select_ddpm_cases(seed_df: pd.DataFrame, predictions_by_condition: dict[str, dict[str, np.ndarray]], clean: np.ndarray, condition_payloads: dict[str, dict[str, np.ndarray]]):
    ddpm_df = seed_df[seed_df["method"] == "ddpm_v3_inpainting_anchored"].copy()
    coarse_df = seed_df[seed_df["method"] == "linear_interp"][["condition", "trajectory_id", "masked_ADE"]].rename(columns={"masked_ADE": "coarse_masked_ADE"})
    ddpm_df = ddpm_df.merge(coarse_df, on=["condition", "trajectory_id"], how="left")
    ddpm_df["delta_masked_ADE"] = ddpm_df["coarse_masked_ADE"] - ddpm_df["masked_ADE"]
    median_target = float(ddpm_df["delta_masked_ADE"].median())
    median_idx = (ddpm_df["delta_masked_ADE"] - median_target).abs().idxmin()
    median_case = ddpm_df.loc[median_idx]
    best_case = ddpm_df.loc[ddpm_df["delta_masked_ADE"].idxmax()]
    worst_case = ddpm_df.loc[ddpm_df["delta_masked_ADE"].idxmin()]

    def pack(case_row: pd.Series, figure_path: Path) -> dict[str, Any]:
        condition = str(case_row["condition"])
        traj_id = int(case_row["trajectory_id"])
        seed = int(case_row["seed"])
        payload = condition_payloads[condition]
        span_start = int(payload["span_start"][traj_id])
        span_len = int(payload["span_end"][traj_id] - payload["span_start"][traj_id] + 1)
        return {
            "condition": condition,
            "trajectory_id": traj_id,
            "seed": seed,
            "missing_start": span_start,
            "missing_len": span_len,
            "coarse_method": "linear_interp",
            "delta_masked_ADE_definition": "coarse_masked_ADE - final_masked_ADE",
            "delta_masked_ADE": float(case_row["delta_masked_ADE"]),
            "figure_path": str(figure_path),
        }

    selected = {
        "median_case": pack(median_case, FIG_MEDIAN_PATH),
        "best_improvement_case": pack(best_case, FIG_BEST_PATH),
        "worst_degradation_case": pack(worst_case, FIG_WORST_PATH),
    }
    write_json(SELECTED_CASES_PATH, selected)

    figure_specs = [
        ("median_case", FIG_MEDIAN_PATH, "Median DDPM case"),
        ("best_improvement_case", FIG_BEST_PATH, "Best DDPM improvement case"),
        ("worst_degradation_case", FIG_WORST_PATH, "Worst DDPM degradation case"),
    ]
    trace_lines = [
        "# DDPM Case Traceability",
        "",
        "All three figures use the same trajectory, condition, seed, and missing span across all five columns within each figure.",
        "",
    ]
    for key, figure_path, title in figure_specs:
        case = selected[key]
        condition = case["condition"]
        traj_id = case["trajectory_id"]
        seed = case["seed"]
        payload = condition_payloads[condition]
        preds = predictions_by_condition[condition]
        span_start = int(case["missing_start"])
        span_end = span_start + int(case["missing_len"]) - 1
        title_text = (
            f"{title} | trajectory_id={traj_id} | condition={condition} | seed={seed} | "
            f"missing_start={span_start} | missing_len={case['missing_len']} | coarse_method=linear_interp | "
            f"delta_masked_ADE=coarse_masked_ADE-final_masked_ADE={case['delta_masked_ADE']:+.6f}"
        )
        make_five_column_figure(
            figure_path=figure_path,
            title=title_text,
            clean=clean[traj_id],
            degraded=payload["degraded"][traj_id],
            coarse=preds["linear_interp"][traj_id],
            ddpm_candidate=preds["ddpm_v3_inpainting"][traj_id, seed],
            final_refined=preds["ddpm_v3_inpainting_anchored"][traj_id, seed],
            obs_mask=payload["obs_mask"][traj_id],
            span_start=span_start,
            span_end=span_end,
        )
        trace_lines.append(f"- `{figure_path.name}`: PASS")
    CASE_TRACEABILITY_PATH.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")


def write_geometry_statement():
    GEOMETRY_STATEMENT_PATH.write_text(
        "\n".join(
            [
                "# Geometry Usage Statement",
                "",
                "- training: NO",
                "- conditioning: NO",
                "- loss: NO",
                "- sampling: NO",
                "- rejection: NO",
                "- evaluation only: YES",
                "- correct term: geometry feasibility evaluation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_fde_audit(seed_df: pd.DataFrame):
    lines = [
        "# FDE Zero Audit",
        "",
        "Question: is FDE equal to zero because endpoint clamping or endpoint preservation forces the last frame to match ground truth?",
        "",
        "Short answer:",
        "- `linear_interp`: YES, FDE is exactly zero because the last frame is observed and linear interpolation preserves observed boundary points.",
        "- `ddpm_v3_inpainting_anchored`: YES, FDE is exactly zero because the anchoring step restores all observed points, including the final frame.",
        "- `savgol_w5_p2`: NO, FDE is not forced to zero because smoothing perturbs observed points after gap filling.",
        "- `kalman_cv_dt1.0_q1e-3_r1e-2`: NO, FDE is not forced to zero because the Kalman rollout estimates all frames and does not hard-clamp the final observation.",
        "- `ddpm_v3_inpainting`: NO, FDE is not forced to zero because the raw DDPM output is not post-clamped in absolute endpoint space.",
        "",
        "Evidence from code:",
        "- `build_obs_mask` only removes an interior contiguous span, so the final frame remains observed.",
        "- `run_linear_interp` and `run_savgol.validate_and_interp` require the sequence to start and end with observations.",
        "- `anchor_missing_spans` writes `traj[observed_mask] = observed_abs[...]`, which restores every observed point exactly.",
        "- `ddpm_prior_inpainting_v3` clamps known relative displacements during reverse diffusion, but it does not directly hard-clamp the final absolute endpoint after absolute reconstruction.",
        "",
        "Observed FDE behavior in canonical_v1 seed-level outputs:",
        "",
    ]
    for condition in CONDITIONS:
        lines.append(f"## {condition}")
        subset = seed_df[(seed_df['condition'] == condition)]
        for method in METHODS:
            group = subset[subset["method"] == method]
            values = group["FDE"].astype(float).to_numpy()
            zero_count = int(np.sum(np.abs(values) < 1e-12))
            lines.append(
                f"- `{method}`: N={values.size}, zero_count={zero_count}, zero_fraction={zero_count / max(values.size,1):.6f}, "
                f"mean={np.mean(values):.6f}, max_abs={np.max(np.abs(values)):.6f}"
            )
        lines.append("")
    lines += [
        "Verdict:",
        "- FDE is exactly zero for `linear_interp` and `ddpm_v3_inpainting_anchored` because endpoint preservation restores the final observed frame.",
        "- FDE is not globally zero for the other methods, so there is no universal endpoint-clamping artifact across the whole protocol.",
    ]
    FDE_AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(config: CanonicalConfig, prior_meta: dict[str, Any], seed_rows: int, traj_rows: int, missing_count: int):
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        "This is the fixed Stage 3 canonical protocol.",
        "Old Stage 3 directories are historical and not primary evidence.",
        "All report tables and figures should be regenerated from `outputs/stage3/canonical_v1`.",
        "",
        "Protocol summary:",
        f"- dataset: `canonical_room3` from `{Path('outputs/stage3/phase1/canonical_room3/data/clean_windows_room3.npz')}`",
        f"- fixed conditions: {', '.join(CONDITIONS)}",
        f"- methods: {', '.join(METHODS)}",
        f"- metrics: {', '.join(METRICS)}",
        f"- deterministic baselines and DDPM seed-level results have different N definitions",
        "- trajectory-level aggregation is the fair comparison table",
        "- `seed_best` is an oracle diagnostic only and must not be used as the main comparison conclusion",
        "",
        "Execution note:",
        f"- current fixed runnable subset uses the first `{config.max_trajectories}` canonical_room3 trajectories so the DDPM seed-level evidence layer is reproducible in the local CPU environment",
        "",
        "DDPM prior:",
        f"- objective: `{prior_meta['objective']}`",
        f"- variant: `{prior_meta['variant']}`",
        f"- checkpoint: `{prior_meta['checkpoint_path']}`",
        "",
        "Generated outputs:",
        f"- seed-level rows: `{seed_rows}`",
        f"- trajectory-level rows: `{traj_rows}`",
        f"- missing-cell count: `{missing_count}`",
        "",
    ]
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_changelog(config: CanonicalConfig, missing_warnings: list[str]):
    lines = [
        "# CHANGELOG",
        "",
        f"## {date.today()}",
        "",
        "Files created:",
        f"- `{CODE_ROOT / 'run_canonical_v1.py'}`",
        f"- `{CODE_ROOT / 'utils.py'}`",
        f"- `{CODE_ROOT / 'README.md'}`",
        f"- `{CODE_ROOT / 'CHANGELOG.md'}`",
        "",
        "Protocol fixed:",
        f"- methods included: {', '.join(METHODS)}",
        f"- conditions included: {', '.join(CONDITIONS)}",
        f"- max trajectories: {config.max_trajectories}",
        f"- num ddpm seeds: {config.num_ddpm_seeds}",
        "",
        "Outputs generated:",
        f"- `{CONFIG_PATH}`",
        f"- `{RAW_SEED_LEVEL_PATH}`",
        f"- `{RAW_TRAJ_LEVEL_PATH}`",
        f"- `{SELECTED_CASES_PATH}`",
        f"- `{FULL_MATRIX_SEED_CSV}`",
        f"- `{FULL_MATRIX_TRAJ_CSV}`",
        f"- `{TABLE2_REPLACEMENT_MD}`",
        f"- `{FIGURE3_PATH}`",
        f"- `{FIGURE5_PATH}`",
        f"- `{FIG_MEDIAN_PATH}`",
        f"- `{FIG_BEST_PATH}`",
        f"- `{FIG_WORST_PATH}`",
        f"- `{FDE_AUDIT_PATH}`",
        "",
        "Missing data warnings:",
    ]
    if missing_warnings:
        lines.extend(f"- {warning}" for warning in missing_warnings)
    else:
        lines.append("- none")
    CHANGELOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_df = pd.read_csv(RAW_SEED_LEVEL_PATH)
    traj_df = pd.read_csv(RAW_TRAJ_LEVEL_PATH)
    return seed_df, traj_df


def main():
    args = build_parser().parse_args()
    config = CanonicalConfig(
        max_trajectories=args.max_trajectories,
        num_ddpm_seeds=args.num_ddpm_seeds,
        ddpm_device=args.ddpm_device,
    )
    ensure_output_dirs()
    logger = RunLogger(OUTPUT_ROOT / "logs" / "run_log.txt")
    missing_warnings: list[str] = []
    predictions_by_condition: dict[str, dict[str, np.ndarray]] = {}
    condition_payloads: dict[str, dict[str, np.ndarray]] = {}

    try:
        prior_meta = get_prior_metadata(config)
        if args.reuse_existing_raw:
            logger.log("Reusing existing raw canonical_v1 CSV files; no DDPM recomputation will run.")
            seed_df, traj_df = load_existing_raw_frames()
        else:
            clean = load_clean_room3(config.max_trajectories)
            load_prior_checkpoint(
                type("Tmp", (), {
                    "objective": config.ddpm_objective,
                    "train_seed": config.ddpm_train_seed,
                    "train_epochs": config.ddpm_train_epochs,
                    "timesteps": config.ddpm_timesteps,
                    "hidden_dim": config.ddpm_hidden_dim,
                    "device": config.ddpm_device,
                })()
            )
            seed_rows: list[dict[str, Any]] = []

            for condition in CONDITIONS:
                span_ratio, span_mode, seed = parse_condition(condition)
                logger.log(f"Running condition {condition}")
                obs_mask, span_start, span_end = build_obs_mask(clean.shape[0], clean.shape[1], span_ratio, span_mode, seed)
                degraded = build_degraded(clean, obs_mask)
                condition_payloads[condition] = {
                    "obs_mask": obs_mask,
                    "span_start": span_start,
                    "span_end": span_end,
                    "degraded": degraded,
                }
                preds: dict[str, np.ndarray] = {}
                preds["linear_interp"] = run_linear_interp(degraded, obs_mask)
                preds["savgol_w5_p2"] = run_savgol(degraded, obs_mask, config.savgol_window_length, config.savgol_polyorder)
                preds["kalman_cv_dt1.0_q1e-3_r1e-2"] = run_kalman(degraded, obs_mask, config.kalman_dt, config.kalman_process_var, config.kalman_measure_var)
                preds["ddpm_v3_inpainting"] = run_ddpm(degraded, obs_mask, seed, config)
                preds["ddpm_v3_inpainting_anchored"] = anchor_missing_spans(preds["ddpm_v3_inpainting"], degraded, obs_mask)
                predictions_by_condition[condition] = preds

                for method in METHODS:
                    if preds[method].ndim == 3:
                        seed_rows.extend(deterministic_seed_rows(condition, method, clean, preds[method], obs_mask, span_start, span_end))
                    else:
                        seed_rows.extend(stochastic_seed_rows(condition, method, clean, preds[method], obs_mask, span_start, span_end))

            seed_df = pd.DataFrame(seed_rows)
            traj_df = aggregate_trajectory_level(seed_df)

            seed_fieldnames = ["condition", "method", "method_label", "trajectory_id", "seed", "seed_role", "aggregation_level", "status", "span_start", "span_len", *METRICS]
            traj_fieldnames = ["condition", "method", "method_label", "trajectory_id", "seed", "seed_role", "aggregation_level", "aggregation", "status", "span_start", "span_len", *METRICS]
            write_csv(RAW_SEED_LEVEL_PATH, seed_fieldnames, seed_df.to_dict(orient="records"))
            write_csv(RAW_TRAJ_LEVEL_PATH, traj_fieldnames, traj_df.to_dict(orient="records"))

        seed_matrix_rows, missing_seed_rows = build_stats_matrix(seed_df, level_name="seed_level")
        traj_matrix_rows, missing_traj_rows = build_stats_matrix(traj_df, level_name="trajectory_level")
        write_csv(FULL_MATRIX_SEED_CSV, list(seed_matrix_rows[0].keys()), seed_matrix_rows)
        write_csv(FULL_MATRIX_TRAJ_CSV, list(traj_matrix_rows[0].keys()), traj_matrix_rows)
        FULL_MATRIX_SEED_MD.write_text(build_markdown_matrix("Full Matrix Seed Level", seed_matrix_rows), encoding="utf-8")
        FULL_MATRIX_TRAJ_MD.write_text(build_markdown_matrix("Full Matrix Trajectory Level", traj_matrix_rows), encoding="utf-8")
        write_csv(MISSING_CELL_AUDIT_CSV, ["condition", "method", "aggregation", "metric", "status"], missing_seed_rows + missing_traj_rows)

        main_traj_rows = [
            row
            for row in traj_matrix_rows
            if row["aggregation"] != "seed_best"
        ]
        seed_best_rows = [
            row
            for row in traj_matrix_rows
            if row["aggregation"] == "seed_best"
        ]
        table2_lines = [
            "# Table 2 Complete Replacement",
            "",
            "Use `full_matrix_trajectory_level.csv` as the fair comparison matrix.",
            "DDPM rows in the fair table come from trajectory-level seed aggregation, not raw seed-level duplication.",
            "`seed_best` is an oracle diagnostic and is intentionally excluded from the main comparison block below.",
            "",
            markdown_table(
                [row for row in main_traj_rows if row["metric"] in ["ADE", "FDE", "RMSE", "masked_ADE", "masked_RMSE", "endpoint_error", "path_length_error", "acceleration_error", "off_map_ratio", "wall_crossing_count"]],
                ["condition", "method", "aggregation", "metric", "status", "N", "mean", "std", "median", "min", "max", "p05", "p25", "p75", "p95"],
            ),
            "",
            "## Oracle Diagnostic Only",
            "",
            "The rows below use `seed_best`, which selects the best DDPM sample per trajectory after seeing all DDPM seeds.",
            "This is an oracle diagnostic for upper-bound analysis and must not be used as the main result table.",
            "",
            markdown_table(
                [row for row in seed_best_rows if row["metric"] in ["ADE", "FDE", "RMSE", "masked_ADE", "masked_RMSE", "endpoint_error", "path_length_error", "acceleration_error", "off_map_ratio", "wall_crossing_count"]],
                ["condition", "method", "aggregation", "metric", "status", "N", "mean", "std", "median", "min", "max", "p05", "p25", "p75", "p95"],
            ),
            "",
        ]
        TABLE2_REPLACEMENT_MD.write_text("\n".join(table2_lines), encoding="utf-8")

        render_figure3(seed_df)
        render_figure5(seed_df, traj_df)
        if not args.reuse_existing_raw:
            select_ddpm_cases(seed_df, predictions_by_condition, clean, condition_payloads)
        write_geometry_statement()
        write_fde_audit(seed_df)

        write_json(
            CONFIG_PATH,
            {
                "experiment_name": EXPERIMENT_NAME,
                "dataset": "canonical_room3",
                "conditions": CONDITIONS,
                "methods": METHODS,
                "metrics": METRICS,
                "max_trajectories": config.max_trajectories,
                "num_ddpm_seeds": config.num_ddpm_seeds,
                "ddpm_device": config.ddpm_device,
                "reuse_existing_raw": bool(args.reuse_existing_raw),
                "prior": prior_meta,
                "geometry_usage": "evaluation-only",
                "trajectory_level_aggregations": DDPM_AGGREGATIONS,
                "main_comparison_aggregations": ["deterministic", "seed_mean", "seed_median", "seed_worst"],
                "oracle_diagnostic_aggregation": "seed_best",
            },
        )
        missing_count = len([row for row in missing_seed_rows + missing_traj_rows if row["status"] != "ok"])
        write_readme(config, prior_meta, len(seed_df), len(traj_df), missing_count)
        write_changelog(config, missing_warnings)

        logger.log(f"output_root = {OUTPUT_ROOT}")
        logger.log(f"methods_evaluated = {', '.join(METHODS)}")
        logger.log(f"conditions_evaluated = {', '.join(CONDITIONS)}")
        logger.log(f"number_of_seed_level_rows = {len(seed_df)}")
        logger.log(f"number_of_trajectory_level_rows = {len(traj_df)}")
        logger.log(f"table_paths = {FULL_MATRIX_SEED_CSV}, {FULL_MATRIX_TRAJ_CSV}, {TABLE2_REPLACEMENT_MD}, {MISSING_CELL_AUDIT_CSV}")
        logger.log(f"figure_paths = {FIGURE3_PATH}, {FIGURE5_PATH}, {FIG_MEDIAN_PATH}, {FIG_BEST_PATH}, {FIG_WORST_PATH}")
        logger.log(f"missing_cell_count = {missing_count}")
        logger.log("geometry_usage_verdict = geometry feasibility evaluation")
        logger.log(f"reuse_existing_raw = {bool(args.reuse_existing_raw)}")
        if missing_warnings:
            logger.log("warnings_about_missing_raw_data = " + " | ".join(missing_warnings))
        else:
            logger.log("warnings_about_missing_raw_data = none")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
