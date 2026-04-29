from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUT_DIR = PROJECT_ROOT / "outputs" / "stage3" / "report_issue_checkout"
MAIN_MD = OUT_DIR / "STAGE3_REPORT_ISSUE_CHECKOUT.md"
TABLE2_CSV = OUT_DIR / "table2_full_matrix_checkout.csv"
FIG3_CSV = OUT_DIR / "figure3_spread_checkout.csv"
FIG5_CSV = OUT_DIR / "figure5_alpha_spread_checkout.csv"
DDPM_FIG_MD = OUT_DIR / "ddpm_trajectory_figure_checkout.md"
GEOM_MD = OUT_DIR / "geometry_usage_checkout.md"
MANIFEST_CSV = OUT_DIR / "raw_file_manifest.csv"

SUMMARY_REPORT_MD = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "eval" / "summary_report.md"
SUMMARY_METRICS_CSV = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "eval" / "summary_metrics.csv"
RANDOM_SPAN_BY_SEED_CSV = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "random_span_statistics" / "metrics_by_seed.csv"
RANDOM_SPAN_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "random_span_statistics" / "metrics_summary_mean_std.csv"
RANDOM_SPAN_FIG = PROJECT_ROOT / "outputs" / "stage3" / "phase1" / "canonical_room3" / "random_span_statistics" / "figures" / "random_span_masked_ADE_mean_std.png"
ALPHA_SWEEP_METRICS_CSV = PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "alpha_sweep" / "alpha_sweep_metrics.csv"
ALPHA_SWEEP_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "alpha_sweep" / "alpha_sweep_summary.csv"
ALPHA_SWEEP_REPORT_MD = PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "alpha_sweep" / "alpha_sweep_report.md"
ALPHA_SWEEP_FIG = PROJECT_ROOT / "outputs" / "stage3" / "refinement" / "alpha_sweep" / "figures" / "alpha_sweep_masked_ADE.png"
INPAINT_REPORT_MD = PROJECT_ROOT / "outputs" / "stage3" / "inpainting_experiment" / "REPORT.md"
INPAINT_FULL_CSV = PROJECT_ROOT / "outputs" / "stage3" / "inpainting_experiment" / "full_results.csv"
INPAINT_VAR_CSV = PROJECT_ROOT / "outputs" / "stage3" / "inpainting_experiment" / "variance_decomposition.csv"
CORRECTION_SELECTED_CASES = PROJECT_ROOT / "outputs" / "stage3" / "correction_exp01_ethucy_indomain_quick" / "raw" / "selected_cases.json"
CORRECTION_README = PROJECT_ROOT / "outputs" / "stage3" / "correction_exp01_ethucy_indomain_quick" / "README.md"
CORRECTION_FIG_DIR = PROJECT_ROOT / "outputs" / "stage3" / "correction_exp01_ethucy_indomain_quick" / "figures"
DIAGNOSIS_MD = PROJECT_ROOT / "outputs" / "stage3" / "diagnosis" / "DIAGNOSIS.md"
FIG_MANIFEST_MD = PROJECT_ROOT / "docs" / "assets" / "stage3" / "figure_manifest.md"
GEOM_PROFILE_MD = PROJECT_ROOT / "outputs" / "stage3" / "geometry_extension" / "geometry_profiles_summary.md"
GEOM_PROFILE_CSV = PROJECT_ROOT / "outputs" / "stage3" / "geometry_extension" / "geometry_profiles_summary.csv"


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_status(path: Path) -> str:
    return "exists" if path.exists() else "missing"


def build_raw_manifest() -> list[dict[str, str]]:
    paths = [
        SUMMARY_REPORT_MD,
        SUMMARY_METRICS_CSV,
        RANDOM_SPAN_BY_SEED_CSV,
        RANDOM_SPAN_SUMMARY_CSV,
        RANDOM_SPAN_FIG,
        ALPHA_SWEEP_METRICS_CSV,
        ALPHA_SWEEP_SUMMARY_CSV,
        ALPHA_SWEEP_REPORT_MD,
        ALPHA_SWEEP_FIG,
        INPAINT_REPORT_MD,
        INPAINT_FULL_CSV,
        INPAINT_VAR_CSV,
        CORRECTION_SELECTED_CASES,
        CORRECTION_README,
        DIAGNOSIS_MD,
        FIG_MANIFEST_MD,
        GEOM_PROFILE_MD,
        GEOM_PROFILE_CSV,
        PROJECT_ROOT / "tools" / "stage3" / "eval" / "run_phase1_random_span_statistics.py",
        PROJECT_ROOT / "tools" / "stage3" / "eval" / "plot_phase1_random_span_statistics.py",
        PROJECT_ROOT / "tools" / "stage3" / "refinement" / "run_alpha_sweep.py",
        PROJECT_ROOT / "tools" / "stage3" / "refinement" / "run_inpainting_experiment.py",
        PROJECT_ROOT / "tools" / "stage3" / "refinement" / "ddpm_refiner.py",
        PROJECT_ROOT / "tools" / "stage3" / "geometry_extension" / "run_geometry_extension.py",
        PROJECT_ROOT / "tools" / "stage3" / "eval" / "eval_geometry_metrics.py",
        PROJECT_ROOT / "tools" / "stage3" / "correction_exp01_ethucy_indomain_quick" / "run_exp01.py",
    ]
    figs = sorted(CORRECTION_FIG_DIR.glob("*.png"))
    paths.extend(figs)
    rows = []
    for path in paths:
        rows.append(
            {
                "path": rel(path),
                "exists": file_status(path),
                "type": "file",
            }
        )
    return rows


def section_table2() -> dict[str, Any]:
    summary_df = pd.read_csv(SUMMARY_METRICS_CSV)
    methods = sorted(summary_df["method_tag"].unique().tolist())
    conditions = sorted(summary_df["experiment_id"].unique().tolist())
    metrics = ["ADE", "FDE", "RMSE", "masked_ADE", "masked_RMSE", "off_map_ratio", "wall_crossing_count"]

    rows: list[dict[str, Any]] = []
    missing_cells: list[str] = []
    for method in methods:
        for condition in conditions:
            sub = summary_df[(summary_df["method_tag"] == method) & (summary_df["experiment_id"] == condition)]
            if sub.empty:
                for metric in metrics:
                    rows.append(
                        {
                            "method": method,
                            "condition": condition,
                            "metric": metric,
                            "N": "not_run",
                            "mean": "not_run",
                            "std": "not_run",
                            "median": "not_run",
                            "min": "not_run",
                            "max": "not_run",
                            "p05": "not_run",
                            "p25": "not_run",
                            "p75": "not_run",
                            "p95": "not_run",
                            "source_file": rel(SUMMARY_METRICS_CSV),
                        }
                    )
                    missing_cells.append(f"{method} × {condition} × {metric}: not_run")
                continue

            row = sub.iloc[0]
            for metric in metrics:
                mean_value = row[metric] if metric in row.index else "metric_unavailable"
                if metric not in row.index:
                    missing_cells.append(f"{method} × {condition} × {metric}: metric_unavailable")
                rows.append(
                    {
                        "method": method,
                        "condition": condition,
                        "metric": metric,
                        "N": int(row["num_samples"]) if "num_samples" in row.index else "missing_raw_data",
                        "mean": mean_value,
                        "std": "missing_raw_data",
                        "median": "missing_raw_data",
                        "min": "missing_raw_data",
                        "max": "missing_raw_data",
                        "p05": "missing_raw_data",
                        "p25": "missing_raw_data",
                        "p75": "missing_raw_data",
                        "p95": "missing_raw_data",
                        "source_file": rel(SUMMARY_METRICS_CSV),
                    }
                )
                if metric in row.index:
                    missing_cells.append(f"{method} × {condition} × {metric}: std/quantiles missing_raw_data")

    write_csv(
        TABLE2_CSV,
        rows,
        ["method", "condition", "metric", "N", "mean", "std", "median", "min", "max", "p05", "p25", "p75", "p95", "source_file"],
    )

    direct = "PARTIAL"
    verdict = "Table 2 hides information because it reports only winners / selected values"
    return {
        "direct": direct,
        "report_table_path": rel(SUMMARY_REPORT_MD),
        "source_script": "outputs/stage3/phase1/canonical_room3/eval/summary_report.md is a rendered report table; underlying aggregation file is outputs/stage3/phase1/canonical_room3/eval/summary_metrics.csv",
        "source_data_file": rel(SUMMARY_METRICS_CSV),
        "methods": methods,
        "conditions": conditions,
        "metrics": metrics,
        "missing_cells": missing_cells,
        "verdict": verdict,
    }


def section_figure3() -> dict[str, Any]:
    raw_df = pd.read_csv(RANDOM_SPAN_BY_SEED_CSV)
    summary_df = pd.read_csv(RANDOM_SPAN_SUMMARY_CSV)
    target = summary_df[summary_df["metric"] == "masked_ADE"].copy()
    out_rows = []
    for _, row in target.iterrows():
        method = row["method"]
        raw_sub = raw_df[raw_df["method"] == method]
        raw_std = float(raw_sub["masked_ADE"].std(ddof=0))
        out_rows.append(
            {
                "plot_group": method,
                "metric": "masked_ADE",
                "raw_N": int(len(raw_sub)),
                "plotted_mean": float(row["mean"]),
                "plotted_spread": float(row["std"]),
                "raw_std_if_available": raw_std,
                "seed_aggregated_std_if_available": float(row["std"]),
                "condition_aggregated_std_if_available": "",
                "spread_definition": "std across already-averaged per-seed experiment summaries",
                "source_file": rel(RANDOM_SPAN_SUMMARY_CSV),
            }
        )
    write_csv(
        FIG3_CSV,
        out_rows,
        [
            "plot_group",
            "metric",
            "raw_N",
            "plotted_mean",
            "plotted_spread",
            "raw_std_if_available",
            "seed_aggregated_std_if_available",
            "condition_aggregated_std_if_available",
            "spread_definition",
            "source_file",
        ],
    )
    return {
        "direct": "Figure 3 spread is misleading",
        "what_is_averaged": "For each method and metric, the plotted mean is the average across 20 random-span seeds of one summary number per seed.",
        "grouping_keys": ["method", "metric"],
        "N_per_point": "20 seed-level summary rows, not per-trajectory raw errors",
        "spread_type": "std",
        "spread_population": "already-averaged summaries over random span seeds",
        "raw_yes_no": "NO",
        "summary_yes_no": "YES",
        "small_spread_explainable": "YES",
        "figure_path": rel(RANDOM_SPAN_FIG),
        "source_script": "tools/stage3/eval/run_phase1_random_span_statistics.py + tools/stage3/eval/plot_phase1_random_span_statistics.py",
        "source_data": rel(RANDOM_SPAN_SUMMARY_CSV),
        "relevant_code": [
            "run_phase1_random_span_statistics.py:65-85 compute_summary_rows",
            "run_phase1_random_span_statistics.py:184-271 collect one row per seed and method",
            "plot_phase1_random_span_statistics.py:79-106 plot_metric_bar uses yerr=std from summary CSV",
        ],
    }


def section_figure5() -> dict[str, Any]:
    summary_df = pd.read_csv(ALPHA_SWEEP_SUMMARY_CSV)
    agg = summary_df.groupby(["alpha"], as_index=False).agg(
        mean=("masked_ADE", "mean"),
        min_val=("masked_ADE", "min"),
        max_val=("masked_ADE", "max"),
    )
    out_rows = []
    for _, row in agg.iterrows():
        alpha = float(row["alpha"])
        sub = summary_df[summary_df["alpha"] == alpha]
        total_std = float(sub["masked_ADE"].std(ddof=0))
        # trajectory and seed variability are not preserved in alpha_sweep_summary.csv
        out_rows.append(
            {
                "alpha": alpha,
                "metric": "masked_ADE",
                "N": int(len(sub)),
                "mean": float(row["mean"]),
                "total_std": total_std,
                "trajectory_std_if_available": "missing_raw_data",
                "seed_std_if_available": "not_applicable",
                "condition_std_if_available": "mixed_in_total_std",
                "spread_definition": "min/max envelope across degradation × coarse_method means at fixed alpha",
                "source_file": rel(ALPHA_SWEEP_SUMMARY_CSV),
            }
        )
    write_csv(
        FIG5_CSV,
        out_rows,
        [
            "alpha",
            "metric",
            "N",
            "mean",
            "total_std",
            "trajectory_std_if_available",
            "seed_std_if_available",
            "condition_std_if_available",
            "spread_definition",
            "source_file",
        ],
    )
    mean_by_alpha = agg["mean"].tolist()
    increasing = "UNKNOWN"
    if len(mean_by_alpha) >= 2:
        increasing = "YES" if mean_by_alpha[-1] > mean_by_alpha[0] else "NO"
    return {
        "direct": "Figure 5 spread is ambiguous",
        "what_is_averaged": "At each alpha, the plotted center is the mean masked_ADE across coarse methods and degradation settings after each cell has already been averaged.",
        "grouping_keys": ["coarse_method", "alpha", "degradation"],
        "N_per_alpha": "12 summary cells per alpha = 3 coarse methods × 4 degradation settings",
        "spread_type": "min/max envelope, not std/SEM/CI",
        "spread_population": "degradation-condition and coarse-method mixture after averaging",
        "variability_source": "coarse baseline + degradation condition + alpha mixture",
        "spread_increase": increasing,
        "ddpm_seed_main_source": "NO",
        "figure_path": rel(ALPHA_SWEEP_FIG),
        "source_script": "tools/stage3/refinement/run_alpha_sweep.py",
        "source_data": rel(ALPHA_SWEEP_SUMMARY_CSV),
        "relevant_code": [
            "run_alpha_sweep.py:151-155 aggregates mean/min/max by coarse_method and alpha",
            "run_alpha_sweep.py:166-180 plots yerr from min/max envelope",
            "run_alpha_sweep.py:349-350 report says mean masked_ADE by alpha across four degradation settings",
        ],
    }


def section_ddpm_figures() -> dict[str, Any]:
    figure_rows = []
    found_paths = []

    # correction exp01 figures
    if CORRECTION_SELECTED_CASES.exists():
        payload = json.loads(CORRECTION_SELECTED_CASES.read_text(encoding="utf-8"))
        for case_name, case in payload.get("cases", {}).items():
            figure_path = Path(case["figure_path"])
            found_paths.append(figure_path)
            figure_rows.append(
                {
                    "figure": rel(figure_path),
                    "source": rel(CORRECTION_SELECTED_CASES),
                    "clean target": "YES",
                    "degraded input": "YES",
                    "coarse reconstruction": "YES",
                    "DDPM candidate": "YES",
                    "final refined output": "YES",
                    "same trajectory_id/seed/span": "YES",
                    "missing segment highlighted": "YES",
                    "delta_masked_ADE stated": "NO",
                    "pass_five_column_traceability": "YES",
                    "notes": "Five columns exist, but coarse column is a reference linear baseline rather than a true internal upstream stage of v3.",
                }
            )

    # inpainting experiment figures
    for path in sorted((PROJECT_ROOT / "outputs" / "stage3" / "inpainting_experiment" / "trajectory_plots").glob("*.png")):
        found_paths.append(path)
        figure_rows.append(
            {
                "figure": rel(path),
                "source": rel(INPAINT_REPORT_MD),
                "clean target": "YES",
                "degraded input": "YES",
                "coarse reconstruction": "YES",
                "DDPM candidate": "YES",
                "final refined output": "NO",
                "same trajectory_id/seed/span": "YES",
                "missing segment highlighted": "YES",
                "delta_masked_ADE stated": "NO",
                "pass_five_column_traceability": "NO",
                "notes": "Column 5 is v3 mean over seeds, not final refined output.",
            }
        )

    ddpm_md = [
        "# DDPM Trajectory Figure Checkout",
        "",
        markdown_table(
            figure_rows,
            [
                "figure",
                "source",
                "clean target",
                "degraded input",
                "coarse reconstruction",
                "DDPM candidate",
                "final refined output",
                "same trajectory_id/seed/span",
                "missing segment highlighted",
                "delta_masked_ADE stated",
                "pass_five_column_traceability",
                "notes",
            ],
        ),
        "",
    ]
    write_text(DDPM_FIG_MD, "\n".join(ddpm_md))
    return {
        "direct": "DDPM trajectory figures are incomplete",
        "required_columns_present": {
            "clean target": "YES",
            "degraded input": "YES",
            "coarse reconstruction": "YES",
            "DDPM candidate": "YES",
            "final refined output": "NO",
        },
        "same_trace": "YES",
        "missing_highlighted": "YES",
        "delta_stated": "NO",
        "delta_def": "In inpainting_experiment figures only improvement_pct is shown; correction_exp01 shows delta_masked_ADE in title notes but the definition is not explicitly stated inside the figure.",
        "figure_paths": [rel(path) for path in found_paths],
        "selected_cases_file": rel(CORRECTION_SELECTED_CASES),
        "source_script": "tools/stage3/refinement/run_inpainting_experiment.py and tools/stage3/correction_exp01_ethucy_indomain_quick/run_exp01.py",
        "source_data": [rel(INPAINT_FULL_CSV), rel(CORRECTION_SELECTED_CASES)],
    }


def section_geometry() -> dict[str, Any]:
    lines = [
        "# Geometry Usage Checkout",
        "",
        "Direct answer: Geometry is evaluation-only",
        "",
        "- Used in training: NO",
        "- Used in conditioning: NO",
        "- Used in loss: NO",
        "- Used in sampling: NO",
        "- Used in rejection: NO",
        "- Used only in evaluation: YES",
        "- Can the report call this geometry-aware reconstruction? NO",
        "- Correct term: geometry feasibility evaluation",
        "",
        "Evidence:",
        f"- scripts: `{rel(PROJECT_ROOT / 'tools' / 'stage3' / 'eval' / 'eval_geometry_metrics.py')}`, `{rel(PROJECT_ROOT / 'tools' / 'stage3' / 'geometry_extension' / 'run_geometry_extension.py')}`, `{rel(PROJECT_ROOT / 'tools' / 'stage3' / 'refinement' / 'ddpm_refiner.py')}`",
        f"- result files: `{rel(GEOM_PROFILE_MD)}`, `{rel(GEOM_PROFILE_CSV)}`",
        "- `ddpm_refiner.py` implements refinement and inpainting without geometry inputs or geometry loss.",
        "- `eval_geometry_metrics.py` computes off-map and wall-crossing metrics after reconstruction.",
        "- `run_geometry_extension.py` repeatedly calls the outputs geometry feasibility extensions and synthetic feasibility stress tests.",
        "",
    ]
    write_text(GEOM_MD, "\n".join(lines))
    return {
        "direct": "Geometry is evaluation-only",
        "used_training": "NO",
        "used_conditioning": "NO",
        "used_loss": "NO",
        "used_sampling": "NO",
        "used_rejection": "NO",
        "used_eval_only": "YES",
        "can_call_geometry_aware": "NO",
        "correct_term": "geometry feasibility evaluation",
        "files": [
            rel(PROJECT_ROOT / "tools" / "stage3" / "eval" / "eval_geometry_metrics.py"),
            rel(PROJECT_ROOT / "tools" / "stage3" / "geometry_extension" / "run_geometry_extension.py"),
            rel(PROJECT_ROOT / "tools" / "stage3" / "refinement" / "ddpm_refiner.py"),
        ],
        "result_files": [rel(GEOM_PROFILE_MD), rel(GEOM_PROFILE_CSV)],
    }


def build_main_md(table2: dict[str, Any], fig3: dict[str, Any], fig5: dict[str, Any], ddpm: dict[str, Any], geom: dict[str, Any], missing_warnings: list[str]) -> str:
    lines = [
        "# STAGE3_REPORT_ISSUE_CHECKOUT",
        "",
        "============================================================",
        "1. Table 2 checkout",
        "============================================================",
        "",
        "Question:",
        "Does Table 2 show the complete method × condition × metric matrix, or does it only show winners / selected values?",
        "",
        f"- Direct answer: {table2['direct']}",
        "- Evidence:",
        f"  - report table path: {table2['report_table_path']}",
        f"  - source script: {table2['source_script']}",
        f"  - source data file: {table2['source_data_file']}",
        f"  - methods found: {', '.join(table2['methods'])}",
        f"  - conditions found: {', '.join(table2['conditions'])}",
        f"  - metrics found: {', '.join(table2['metrics'])}",
        "  - missing method × condition × metric cells:",
    ]
    lines.extend([f"    - {cell}" for cell in table2["missing_cells"][:25]])
    if len(table2["missing_cells"]) > 25:
        lines.append(f"    - ... ({len(table2['missing_cells']) - 25} more)")
    lines += [
        "- Verdict:",
        f"  - {table2['verdict']}",
        "",
        "============================================================",
        "2. Figure 3 spread checkout",
        "============================================================",
        "",
        "Question:",
        "Why is Figure 3 standard deviation extremely small? What exactly is averaged, and what does the spread represent?",
        "",
        f"- Direct answer: {fig3['direct']}",
        f"- What is averaged: {fig3['what_is_averaged']}",
        f"- Grouping keys: {', '.join(fig3['grouping_keys'])}",
        f"- N per plotted point: {fig3['N_per_point']}",
        f"- Spread type: {fig3['spread_type']}",
        f"- Spread population: {fig3['spread_population']}",
        f"- Does the spread use raw per-case values? {fig3['raw_yes_no']}",
        f"- Does the spread use already-averaged summaries? {fig3['summary_yes_no']}",
        f"- Is the small spread explainable? {fig3['small_spread_explainable']}",
        "- Evidence:",
        f"  - figure path: {fig3['figure_path']}",
        f"  - source script: {fig3['source_script']}",
        f"  - source data: {fig3['source_data']}",
        "  - relevant code lines or function names:",
    ]
    lines.extend([f"    - {item}" for item in fig3["relevant_code"]])
    lines += [
        "",
        "============================================================",
        "3. Figure 5 alpha sweep checkout",
        "============================================================",
        "",
        "Question:",
        "What does Figure 5 spread represent? Does variability mainly come from coarse filters, DDPM sampling, degradation settings, alpha settings, or a mixture?",
        "",
        f"- Direct answer: {fig5['direct']}",
        f"- What is averaged: {fig5['what_is_averaged']}",
        f"- Grouping keys: {', '.join(fig5['grouping_keys'])}",
        f"- N per alpha: {fig5['N_per_alpha']}",
        f"- Spread type: {fig5['spread_type']}",
        f"- Spread population: {fig5['spread_population']}",
        f"- Variability source: {fig5['variability_source']}",
        f"- Does spread increase with DDPM contribution? {fig5['spread_increase']}",
        f"- If yes, is DDPM sampling likely the main source? {fig5['ddpm_seed_main_source']}",
        "- Evidence:",
        f"  - figure path: {fig5['figure_path']}",
        f"  - source script: {fig5['source_script']}",
        f"  - source data: {fig5['source_data']}",
        "  - relevant code lines or function names:",
    ]
    lines.extend([f"    - {item}" for item in fig5["relevant_code"]])
    lines += [
        "",
        "============================================================",
        "4. DDPM trajectory figure checkout",
        "============================================================",
        "",
        "Question:",
        "Do existing DDPM trajectory figures show the actual effect of DDPM refinement using clean target, degraded input, coarse reconstruction, DDPM candidate, and final refined output?",
        "",
        f"- Direct answer: {ddpm['direct']}",
        "- Required columns present:",
        f"  - clean target: {ddpm['required_columns_present']['clean target']}",
        f"  - degraded input: {ddpm['required_columns_present']['degraded input']}",
        f"  - coarse reconstruction: {ddpm['required_columns_present']['coarse reconstruction']}",
        f"  - DDPM candidate: {ddpm['required_columns_present']['DDPM candidate']}",
        f"  - final refined output: {ddpm['required_columns_present']['final refined output']}",
        f"- Are all columns from the same trajectory_id / seed / missing span? {ddpm['same_trace']}",
        f"- Is missing segment highlighted? {ddpm['missing_highlighted']}",
        f"- Is delta_masked_ADE definition stated? {ddpm['delta_stated']}",
        f"- delta definition if available: {ddpm['delta_def']}",
        "- Evidence:",
        f"  - figure paths: {', '.join(ddpm['figure_paths'])}",
        f"  - selected_cases file: {ddpm['selected_cases_file']}",
        f"  - source script: {ddpm['source_script']}",
        f"  - source data: {', '.join(ddpm['source_data'])}",
        "",
        "============================================================",
        "5. Geometry usage checkout",
        "============================================================",
        "",
        "Question:",
        "Is geometry integrated into the reconstruction method, or only used afterward as evaluation?",
        "",
        f"- Direct answer: {geom['direct']}",
        f"- Used in training: {geom['used_training']}",
        f"- Used in conditioning: {geom['used_conditioning']}",
        f"- Used in loss: {geom['used_loss']}",
        f"- Used in sampling: {geom['used_sampling']}",
        f"- Used in rejection: {geom['used_rejection']}",
        f"- Used only in evaluation: {geom['used_eval_only']}",
        f"- Can the report call this geometry-aware reconstruction? {geom['can_call_geometry_aware']}",
        f"- Correct term: {geom['correct_term']}",
        "- Evidence:",
        f"  - files/functions inspected: {', '.join(geom['files'])}",
        f"  - geometry-related scripts: {', '.join(geom['files'])}",
        f"  - geometry-related result files: {', '.join(geom['result_files'])}",
        "",
        "============================================================",
        "6. Final direct verdict",
        "============================================================",
        "",
        "| Issue | Direct verdict | Needs correction? | Minimal correction |",
        "|---|---|---|---|",
        "| Table 2 completeness | Table 2 hides information because it reports only winners / selected values | YES | Replace with a full method × condition × metric matrix and expose raw-population stats or state that they are unavailable |",
        "| Figure 3 spread | Figure 3 spread is misleading | YES | State that the bars use std across seed-level summary rows, not per-trajectory raw dispersion |",
        "| Figure 5 spread | Figure 5 spread is ambiguous | YES | State that the envelope is min/max across degradation × coarse-method means, not DDPM seed variance |",
        "| DDPM trajectory figures | DDPM trajectory figures are incomplete | YES | Add a figure with clean, degraded, coarse, DDPM candidate, and final refined output from the same trace |",
        "| Geometry wording | Geometry is evaluation-only | YES | Rename as geometry feasibility evaluation, not geometry-aware reconstruction |",
    ]
    if missing_warnings:
        lines += ["", "Missing raw data warnings:"]
        lines.extend([f"- {warning}" for warning in missing_warnings])
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()
    manifest_rows = build_raw_manifest()
    write_csv(MANIFEST_CSV, manifest_rows, ["path", "exists", "type"])

    table2 = section_table2()
    fig3 = section_figure3()
    fig5 = section_figure5()
    ddpm = section_ddpm_figures()
    geom = section_geometry()

    missing_warnings = []
    if not SUMMARY_METRICS_CSV.exists():
        missing_warnings.append(f"Missing raw data: {rel(SUMMARY_METRICS_CSV)}")
    if not RANDOM_SPAN_BY_SEED_CSV.exists():
        missing_warnings.append(f"Missing raw data: {rel(RANDOM_SPAN_BY_SEED_CSV)}")
    if not ALPHA_SWEEP_METRICS_CSV.exists():
        missing_warnings.append(f"Missing raw data: {rel(ALPHA_SWEEP_METRICS_CSV)}")
    if not INPAINT_FULL_CSV.exists():
        missing_warnings.append(f"Missing raw data: {rel(INPAINT_FULL_CSV)}")

    write_text(MAIN_MD, build_main_md(table2, fig3, fig5, ddpm, geom, missing_warnings))

    print(str(MAIN_MD))
    print(f"Table 2 complete: {table2['direct']}")
    print(f"Figure 3 spread definition: {fig3['spread_type']} across {fig3['spread_population']}")
    print(f"Figure 5 spread definition: {fig5['spread_type']} across {fig5['spread_population']}")
    print(f"DDPM five-column figures exist: {'YES' if ddpm['required_columns_present']['final refined output'] == 'YES' else 'PARTIAL'}")
    print(f"Geometry usage: {geom['direct']}")
    if missing_warnings:
        print("Missing raw data warnings:")
        for warning in missing_warnings:
            print(warning)
    else:
        print("Missing raw data warnings: none")


if __name__ == "__main__":
    main()
