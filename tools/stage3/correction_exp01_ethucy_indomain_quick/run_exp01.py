from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3.correction_exp01_ethucy_indomain_quick.utils import (
    BEST_FIG_PATH,
    COARSE_REFERENCE_METHOD,
    CONFIG_JSON_PATH,
    DEFAULT_MAX_TRAJECTORIES,
    DEFAULT_NUM_DDPM_SEEDS,
    DEFAULT_SEED,
    DEFAULT_SPAN_MODE,
    DEFAULT_SPAN_RATIO,
    ExperimentConfig,
    FULL_STATS_CSV_PATH,
    FULL_STATS_MD_PATH,
    INPUT_ABS_PATH,
    MEDIAN_FIG_PATH,
    METHOD_LABELS,
    METHOD_ORDER,
    METRIC_ORDER,
    MISSING_CONDITION,
    OUTPUT_CHANGELOG_PATH,
    OUTPUT_README_PATH,
    PER_CASE_CSV_PATH,
    RAW_DIR,
    RUN_LOG_PATH,
    RunLogger,
    SELECTED_CASES_JSON_PATH,
    SUMMARY_MD_PATH,
    WORST_FIG_PATH,
    anchor_missing_spans,
    build_missing_only_dataset,
    build_output_changelog,
    build_output_readme,
    build_summary_markdown,
    compute_case_metrics,
    ensure_output_dirs,
    get_prior_metadata,
    load_absolute_trajectories,
    make_five_column_figure,
    run_ddpm_v3,
    run_kalman,
    run_linear_interp,
    run_savgol,
    summarize_metric,
    write_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exp01_ethucy_indomain_quick Stage 3 correction experiment."
    )
    parser.add_argument("--input_abs_path", type=str, default=str(INPUT_ABS_PATH))
    parser.add_argument("--max_trajectories", type=int, default=DEFAULT_MAX_TRAJECTORIES)
    parser.add_argument("--span_ratio", type=float, default=DEFAULT_SPAN_RATIO)
    parser.add_argument("--span_mode", type=str, default=DEFAULT_SPAN_MODE, choices=["fixed", "random"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_ddpm_seeds", type=int, default=DEFAULT_NUM_DDPM_SEEDS)
    parser.add_argument("--ddpm_device", type=str, default="auto")
    return parser


def deterministic_case_row(
    method: str,
    traj_idx: int,
    clean: np.ndarray,
    pred: np.ndarray,
    obs_mask: np.ndarray,
    span_start: int,
    span_end: int,
) -> dict[str, Any]:
    metrics = compute_case_metrics(clean, pred, obs_mask, span_start, span_end)
    return {
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "missing_condition": MISSING_CONDITION,
        "trajectory_idx": int(traj_idx),
        "seed_idx": "",
        "case_id": f"traj{traj_idx}",
        "population_unit": "trajectory",
        "span_start": int(span_start),
        "span_end": int(span_end),
        **metrics,
    }


def stochastic_case_row(
    method: str,
    traj_idx: int,
    seed_idx: int,
    clean: np.ndarray,
    pred: np.ndarray,
    obs_mask: np.ndarray,
    span_start: int,
    span_end: int,
) -> dict[str, Any]:
    metrics = compute_case_metrics(clean, pred, obs_mask, span_start, span_end)
    return {
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "missing_condition": MISSING_CONDITION,
        "trajectory_idx": int(traj_idx),
        "seed_idx": int(seed_idx),
        "case_id": f"traj{traj_idx}_seed{seed_idx}",
        "population_unit": "trajectory_seed",
        "span_start": int(span_start),
        "span_end": int(span_end),
        **metrics,
    }


def build_stats_rows(per_case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = [row for row in per_case_rows if row["method"] == method]
        if not method_rows:
            continue
        for metric in METRIC_ORDER:
            stats = summarize_metric([float(row[metric]) for row in method_rows])
            out.append(
                {
                    "method": method,
                    "missing_condition": MISSING_CONDITION,
                    "metric": metric,
                    **stats,
                }
            )
    return out


def build_selected_cases_payload(
    dataset_payload: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    per_case_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ddpm_rows = [row for row in per_case_rows if row["method"] == "ddpm_v3_inpainting_anchored"]
    if not ddpm_rows:
        raise RuntimeError("No ddpm_v3_inpainting_anchored rows available for representative figure selection.")

    coarse_rows = {
        int(row["trajectory_idx"]): row
        for row in per_case_rows
        if row["method"] == COARSE_REFERENCE_METHOD
    }

    enriched = []
    for row in ddpm_rows:
        traj_idx = int(row["trajectory_idx"])
        coarse_row = coarse_rows[traj_idx]
        improvement = float(coarse_row["masked_ADE"] - row["masked_ADE"])
        enriched.append({**row, "coarse_masked_ADE": float(coarse_row["masked_ADE"]), "ddpm_improvement_vs_coarse": improvement})

    improvements = np.asarray([row["ddpm_improvement_vs_coarse"] for row in enriched], dtype=np.float64)
    median_value = float(np.median(improvements))
    median_case = min(enriched, key=lambda row: abs(row["ddpm_improvement_vs_coarse"] - median_value))
    best_case = max(enriched, key=lambda row: row["ddpm_improvement_vs_coarse"])
    worst_case = min(enriched, key=lambda row: row["ddpm_improvement_vs_coarse"])

    note = (
        "The current v3 interface reconstructs directly from degraded observed input. "
        "The five-column figure therefore uses linear interpolation as a reference coarse column, "
        "not as an internal upstream dependency of ddpm_v3_inpainting."
    )

    def pack(case: dict[str, Any], figure_name: str) -> dict[str, Any]:
        traj_idx = int(case["trajectory_idx"])
        seed_idx = int(case["seed_idx"])
        return {
            "trajectory_idx": traj_idx,
            "seed_idx": seed_idx,
            "case_id": case["case_id"],
            "span_start": int(case["span_start"]),
            "span_end": int(case["span_end"]),
            "coarse_reference_method": COARSE_REFERENCE_METHOD,
            "coarse_masked_ADE": float(case["coarse_masked_ADE"]),
            "final_masked_ADE": float(case["masked_ADE"]),
            "ddpm_improvement_vs_coarse": float(case["ddpm_improvement_vs_coarse"]),
            "missing_intermediate_note": note,
            "available_columns": [
                "clean target",
                "degraded input",
                "coarse reconstruction",
                "DDPM candidate",
                "final refined output",
            ],
            "figure_path": str(figure_name),
        }

    return {
        "experiment_name": dataset_payload.get("experiment_name", ""),
        "coarse_reference_method": COARSE_REFERENCE_METHOD,
        "selection_metric": "masked_ADE improvement of ddpm_v3_inpainting_anchored relative to linear_interp",
        "selection_population": "trajectory_seed cases",
        "missing_intermediate_note": note,
        "cases": {
            "median_case": pack(median_case, MEDIAN_FIG_PATH),
            "best_ddpm_improvement_case": pack(best_case, BEST_FIG_PATH),
            "worst_ddpm_degradation_case": pack(worst_case, WORST_FIG_PATH),
        },
    }


def render_selected_case_figures(
    dataset_payload: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    selected_cases_payload: dict[str, Any],
):
    clean_all = dataset_payload["traj_abs"]
    degraded_all = dataset_payload["traj_obs"]
    obs_mask_all = dataset_payload["obs_mask"]
    coarse_all = predictions[COARSE_REFERENCE_METHOD]
    candidate_all = predictions["ddpm_v3_inpainting"]
    final_all = predictions["ddpm_v3_inpainting_anchored"]

    case_map = [
        ("median_case", MEDIAN_FIG_PATH, "Median DDPM improvement case"),
        ("best_ddpm_improvement_case", BEST_FIG_PATH, "Best DDPM improvement case"),
        ("worst_ddpm_degradation_case", WORST_FIG_PATH, "Worst DDPM degradation case"),
    ]
    for key, figure_path, title_prefix in case_map:
        case = selected_cases_payload["cases"][key]
        traj_idx = int(case["trajectory_idx"])
        seed_idx = int(case["seed_idx"])
        span_start = int(case["span_start"])
        span_end = int(case["span_end"])
        notes = [
            f"traj={traj_idx}",
            f"seed={seed_idx}",
            f"coarse={COARSE_REFERENCE_METHOD}",
            f"delta_masked_ADE={case['ddpm_improvement_vs_coarse']:+.6f}",
        ]
        make_five_column_figure(
            figure_path=figure_path,
            title=title_prefix,
            clean=clean_all[traj_idx],
            degraded=degraded_all[traj_idx],
            coarse=coarse_all[traj_idx],
            ddpm_candidate=candidate_all[traj_idx, seed_idx],
            final_refined=final_all[traj_idx, seed_idx],
            obs_mask=obs_mask_all[traj_idx],
            span_start=span_start,
            span_end=span_end,
            notes=notes,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ensure_output_dirs()
    logger = RunLogger(RUN_LOG_PATH)
    warnings: list[str] = []
    skipped_methods: list[dict[str, str]] = []
    try:
        config = ExperimentConfig(
            span_ratio=args.span_ratio,
            span_mode=args.span_mode,
            seed=args.seed,
            num_ddpm_seeds=args.num_ddpm_seeds,
            max_trajectories=args.max_trajectories,
            ddpm_device=args.ddpm_device,
        )
        logger.log("=" * 72)
        logger.log(f"Running {config.experiment_name}")
        logger.log(f"input_abs_path       = {args.input_abs_path}")
        logger.log(f"max_trajectories     = {config.max_trajectories}")
        logger.log(f"missing_condition    = {config.missing_condition}")
        logger.log(f"span_ratio           = {config.span_ratio}")
        logger.log(f"span_mode            = {config.span_mode}")
        logger.log(f"seed                 = {config.seed}")
        logger.log(f"num_ddpm_seeds       = {config.num_ddpm_seeds}")
        logger.log("=" * 72)

        traj_abs = load_absolute_trajectories(Path(args.input_abs_path), max_trajectories=config.max_trajectories)
        dataset_payload = build_missing_only_dataset(
            traj_abs=traj_abs,
            span_ratio=config.span_ratio,
            span_mode=config.span_mode,
            seed=config.seed,
        )
        dataset_payload["experiment_name"] = config.experiment_name

        prior_meta = get_prior_metadata(config)
        predictions: dict[str, np.ndarray] = {}

        logger.log("Running linear interpolation baseline")
        predictions["linear_interp"] = run_linear_interp(dataset_payload["traj_obs"], dataset_payload["obs_mask"])

        try:
            logger.log("Running Savitzky-Golay baseline")
            predictions["savgol_w5_p2"] = run_savgol(
                dataset_payload["traj_obs"],
                dataset_payload["obs_mask"],
                window_length=config.savgol_window_length,
                polyorder=config.savgol_polyorder,
            )
        except Exception as exc:
            reason = f"runtime unavailable: {exc}"
            warnings.append(f"savgol_w5_p2 skipped: {reason}")
            skipped_methods.append({"method": "savgol_w5_p2", "reason": reason})
            logger.log(f"[warning] savgol_w5_p2 skipped: {reason}")

        logger.log("Running Kalman baseline")
        predictions["kalman_cv_dt1.0_q1e-3_r1e-2"] = run_kalman(
            dataset_payload["traj_obs"],
            dataset_payload["obs_mask"],
            dt=config.kalman_dt,
            process_var=config.kalman_process_var,
            measure_var=config.kalman_measure_var,
        )

        logger.log("Running ddpm_v3_inpainting")
        predictions["ddpm_v3_inpainting"] = run_ddpm_v3(
            dataset_payload["traj_obs"],
            dataset_payload["obs_mask"],
            config,
        )
        logger.log("Running ddpm_v3_inpainting_anchored")
        predictions["ddpm_v3_inpainting_anchored"] = anchor_missing_spans(
            predictions["ddpm_v3_inpainting"],
            dataset_payload["traj_obs"],
            dataset_payload["obs_mask"],
        )

        per_case_rows: list[dict[str, Any]] = []
        for method, pred in predictions.items():
            if pred.ndim == 3:
                for traj_idx in range(pred.shape[0]):
                    per_case_rows.append(
                        deterministic_case_row(
                            method=method,
                            traj_idx=traj_idx,
                            clean=dataset_payload["traj_abs"][traj_idx],
                            pred=pred[traj_idx],
                            obs_mask=dataset_payload["obs_mask"][traj_idx],
                            span_start=int(dataset_payload["span_start"][traj_idx]),
                            span_end=int(dataset_payload["span_end"][traj_idx]),
                        )
                    )
            elif pred.ndim == 4:
                for traj_idx in range(pred.shape[0]):
                    for seed_idx in range(pred.shape[1]):
                        per_case_rows.append(
                            stochastic_case_row(
                                method=method,
                                traj_idx=traj_idx,
                                seed_idx=seed_idx,
                                clean=dataset_payload["traj_abs"][traj_idx],
                                pred=pred[traj_idx, seed_idx],
                                obs_mask=dataset_payload["obs_mask"][traj_idx],
                                span_start=int(dataset_payload["span_start"][traj_idx]),
                                span_end=int(dataset_payload["span_end"][traj_idx]),
                            )
                        )
            else:
                raise ValueError(f"Unsupported prediction array rank for {method}: {pred.shape}")

        stats_rows = build_stats_rows(per_case_rows)
        selected_cases_payload = build_selected_cases_payload(dataset_payload, predictions, per_case_rows)
        render_selected_case_figures(dataset_payload, predictions, selected_cases_payload)

        included_methods = [method for method in METHOD_ORDER if method in predictions]
        generated_paths = [
            CONFIG_JSON_PATH,
            PER_CASE_CSV_PATH,
            SELECTED_CASES_JSON_PATH,
            FULL_STATS_CSV_PATH,
            FULL_STATS_MD_PATH,
            SUMMARY_MD_PATH,
            MEDIAN_FIG_PATH,
            BEST_FIG_PATH,
            WORST_FIG_PATH,
            RUN_LOG_PATH,
            OUTPUT_README_PATH,
            OUTPUT_CHANGELOG_PATH,
        ]

        config_payload = {
            "experiment_name": config.experiment_name,
            "purpose": "Quick in-domain sanity check of Stage 3 missing reconstruction on ETH+UCY public trajectories.",
            "missing_condition": config.missing_condition,
            "dataset": {
                "absolute_path": str(Path(args.input_abs_path)),
                "relative_reference_path": str(PROJECT_ROOT / "datasets" / "processed" / "data_eth_ucy_20_rel.npy"),
                "max_trajectories": int(config.max_trajectories),
                "num_trajectories_used": int(dataset_payload["traj_abs"].shape[0]),
                "sequence_length": int(dataset_payload["traj_abs"].shape[1]),
                "natural_scale": True,
                "room3_used": False,
                "global_scaling_applied": False,
            },
            "prior": prior_meta,
            "ddpm": {
                "num_seeds_per_trajectory": int(config.num_ddpm_seeds),
                "device": config.ddpm_device,
            },
            "methods_requested": METHOD_ORDER,
            "methods_included": included_methods,
            "methods_skipped": skipped_methods,
            "coarse_reference_method_for_figures": COARSE_REFERENCE_METHOD,
            "warnings": warnings,
        }

        write_json(CONFIG_JSON_PATH, config_payload)
        write_csv(
            PER_CASE_CSV_PATH,
            fieldnames=[
                "method",
                "method_label",
                "missing_condition",
                "trajectory_idx",
                "seed_idx",
                "case_id",
                "population_unit",
                "span_start",
                "span_end",
                "masked_ADE",
                "masked_RMSE",
                "endpoint_error",
                "path_length_error",
                "acceleration_error",
            ],
            rows=per_case_rows,
        )
        write_json(SELECTED_CASES_JSON_PATH, selected_cases_payload)
        write_csv(
            FULL_STATS_CSV_PATH,
            fieldnames=[
                "method",
                "missing_condition",
                "metric",
                "N",
                "mean",
                "std",
                "median",
                "min",
                "max",
                "p05",
                "p25",
                "p75",
                "p95",
            ],
            rows=stats_rows,
        )

        stats_md_lines = [
            f"# {config.experiment_name} Full Stats Matrix",
            "",
            "Population note:",
            f"- deterministic methods: `N={dataset_payload['traj_abs'].shape[0]}` trajectories",
            f"- DDPM methods: `N={dataset_payload['traj_abs'].shape[0]} × {config.num_ddpm_seeds} = {dataset_payload['traj_abs'].shape[0] * config.num_ddpm_seeds}` trajectory-seed cases",
            "",
            "| method | missing_condition | metric | N | mean | std | median | min | max | p05 | p25 | p75 | p95 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in stats_rows:
            stats_md_lines.append(
                f"| {row['method']} | {row['missing_condition']} | {row['metric']} | {row['N']} | "
                f"{row['mean']:.6f} | {row['std']:.6f} | {row['median']:.6f} | {row['min']:.6f} | "
                f"{row['max']:.6f} | {row['p05']:.6f} | {row['p25']:.6f} | {row['p75']:.6f} | {row['p95']:.6f} |"
            )
        FULL_STATS_MD_PATH.write_text("\n".join(stats_md_lines) + "\n", encoding="utf-8")
        SUMMARY_MD_PATH.write_text(
            build_summary_markdown(config, included_methods, skipped_methods, stats_rows, warnings),
            encoding="utf-8",
        )
        OUTPUT_README_PATH.write_text(
            build_output_readme(
                config,
                dataset_payload,
                prior_meta,
                included_methods,
                skipped_methods,
                stats_rows,
                selected_cases_payload,
                warnings,
            ),
            encoding="utf-8",
        )
        OUTPUT_CHANGELOG_PATH.write_text(
            build_output_changelog(
                run_date=str(date.today()),
                included_methods=included_methods,
                skipped_methods=skipped_methods,
                generated_paths=generated_paths,
            ),
            encoding="utf-8",
        )

        logger.log("=" * 72)
        logger.log(f"output_root_path     = {RAW_DIR.parent}")
        logger.log(f"number_of_cases      = {len(per_case_rows)}")
        logger.log(f"methods_evaluated    = {', '.join(included_methods)}")
        logger.log(f"generated_tables     = {FULL_STATS_CSV_PATH}, {FULL_STATS_MD_PATH}, {SUMMARY_MD_PATH}")
        logger.log(f"generated_figures    = {MEDIAN_FIG_PATH}, {BEST_FIG_PATH}, {WORST_FIG_PATH}")
        if warnings:
            logger.log("warnings:")
            for warning in warnings:
                logger.log(f"- {warning}")
        else:
            logger.log("warnings: none")
        logger.log("=" * 72)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
