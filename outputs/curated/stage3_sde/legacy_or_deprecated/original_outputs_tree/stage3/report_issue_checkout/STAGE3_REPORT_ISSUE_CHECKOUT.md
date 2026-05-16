# STAGE3_REPORT_ISSUE_CHECKOUT

============================================================
1. Table 2 checkout
============================================================

Question:
Does Table 2 show the complete method × condition × metric matrix, or does it only show winners / selected values?

- Direct answer: PARTIAL
- Evidence:
  - report table path: outputs/stage3/phase1/canonical_room3/eval/summary_report.md
  - source script: outputs/stage3/phase1/canonical_room3/eval/summary_report.md is a rendered report table; underlying aggregation file is outputs/stage3/phase1/canonical_room3/eval/summary_metrics.csv
  - source data file: outputs/stage3/phase1/canonical_room3/eval/summary_metrics.csv
  - methods found: kalman_cv_dt1.0_q1e-3_r1e-2, linear_interp, savgol_w5_p2
  - conditions found: span10_fixed_seed42, span20_fixed_seed42, span20_random_seed42, span20_random_seed43, span20_random_seed44, span30_fixed_seed42
  - metrics found: ADE, FDE, RMSE, masked_ADE, masked_RMSE, off_map_ratio, wall_crossing_count
  - missing method × condition × metric cells:
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span10_fixed_seed42 × ADE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span10_fixed_seed42 × FDE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span10_fixed_seed42 × RMSE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span10_fixed_seed42 × masked_ADE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span10_fixed_seed42 × masked_RMSE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span10_fixed_seed42 × off_map_ratio: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span10_fixed_seed42 × wall_crossing_count: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_fixed_seed42 × ADE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_fixed_seed42 × FDE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_fixed_seed42 × RMSE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_fixed_seed42 × masked_ADE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_fixed_seed42 × masked_RMSE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_fixed_seed42 × off_map_ratio: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_fixed_seed42 × wall_crossing_count: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed42 × ADE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed42 × FDE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed42 × RMSE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed42 × masked_ADE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed42 × masked_RMSE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed42 × off_map_ratio: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed42 × wall_crossing_count: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed43 × ADE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed43 × FDE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed43 × RMSE: std/quantiles missing_raw_data
    - kalman_cv_dt1.0_q1e-3_r1e-2 × span20_random_seed43 × masked_ADE: std/quantiles missing_raw_data
    - ... (101 more)
- Verdict:
  - Table 2 hides information because it reports only winners / selected values

============================================================
2. Figure 3 spread checkout
============================================================

Question:
Why is Figure 3 standard deviation extremely small? What exactly is averaged, and what does the spread represent?

- Direct answer: Figure 3 spread is misleading
- What is averaged: For each method and metric, the plotted mean is the average across 20 random-span seeds of one summary number per seed.
- Grouping keys: method, metric
- N per plotted point: 20 seed-level summary rows, not per-trajectory raw errors
- Spread type: std
- Spread population: already-averaged summaries over random span seeds
- Does the spread use raw per-case values? NO
- Does the spread use already-averaged summaries? YES
- Is the small spread explainable? YES
- Evidence:
  - figure path: outputs/stage3/phase1/canonical_room3/random_span_statistics/figures/random_span_masked_ADE_mean_std.png
  - source script: tools/stage3/eval/run_phase1_random_span_statistics.py + tools/stage3/eval/plot_phase1_random_span_statistics.py
  - source data: outputs/stage3/phase1/canonical_room3/random_span_statistics/metrics_summary_mean_std.csv
  - relevant code lines or function names:
    - run_phase1_random_span_statistics.py:65-85 compute_summary_rows
    - run_phase1_random_span_statistics.py:184-271 collect one row per seed and method
    - plot_phase1_random_span_statistics.py:79-106 plot_metric_bar uses yerr=std from summary CSV

============================================================
3. Figure 5 alpha sweep checkout
============================================================

Question:
What does Figure 5 spread represent? Does variability mainly come from coarse filters, DDPM sampling, degradation settings, alpha settings, or a mixture?

- Direct answer: Figure 5 spread is ambiguous
- What is averaged: At each alpha, the plotted center is the mean masked_ADE across coarse methods and degradation settings after each cell has already been averaged.
- Grouping keys: coarse_method, alpha, degradation
- N per alpha: 12 summary cells per alpha = 3 coarse methods × 4 degradation settings
- Spread type: min/max envelope, not std/SEM/CI
- Spread population: degradation-condition and coarse-method mixture after averaging
- Variability source: coarse baseline + degradation condition + alpha mixture
- Does spread increase with DDPM contribution? YES
- If yes, is DDPM sampling likely the main source? NO
- Evidence:
  - figure path: outputs/stage3/refinement/alpha_sweep/figures/alpha_sweep_masked_ADE.png
  - source script: tools/stage3/refinement/run_alpha_sweep.py
  - source data: outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv
  - relevant code lines or function names:
    - run_alpha_sweep.py:151-155 aggregates mean/min/max by coarse_method and alpha
    - run_alpha_sweep.py:166-180 plots yerr from min/max envelope
    - run_alpha_sweep.py:349-350 report says mean masked_ADE by alpha across four degradation settings

============================================================
4. DDPM trajectory figure checkout
============================================================

Question:
Do existing DDPM trajectory figures show the actual effect of DDPM refinement using clean target, degraded input, coarse reconstruction, DDPM candidate, and final refined output?

- Direct answer: DDPM trajectory figures are incomplete
- Required columns present:
  - clean target: YES
  - degraded input: YES
  - coarse reconstruction: YES
  - DDPM candidate: YES
  - final refined output: NO
- Are all columns from the same trajectory_id / seed / missing span? YES
- Is missing segment highlighted? YES
- Is delta_masked_ADE definition stated? NO
- delta definition if available: In inpainting_experiment figures only improvement_pct is shown; correction_exp01 shows delta_masked_ADE in title notes but the definition is not explicitly stated inside the figure.
- Evidence:
  - figure paths: outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/median_case_five_column.png, outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/best_ddpm_improvement_five_column.png, outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/worst_ddpm_degradation_five_column.png, outputs/stage3/inpainting_experiment/trajectory_plots/case_p10_sample930.png, outputs/stage3/inpainting_experiment/trajectory_plots/case_p25_sample63.png, outputs/stage3/inpainting_experiment/trajectory_plots/case_p50_sample334.png, outputs/stage3/inpainting_experiment/trajectory_plots/case_p75_sample62.png, outputs/stage3/inpainting_experiment/trajectory_plots/case_p90_sample968.png
  - selected_cases file: outputs/stage3/correction_exp01_ethucy_indomain_quick/raw/selected_cases.json
  - source script: tools/stage3/refinement/run_inpainting_experiment.py and tools/stage3/correction_exp01_ethucy_indomain_quick/run_exp01.py
  - source data: outputs/stage3/inpainting_experiment/full_results.csv, outputs/stage3/correction_exp01_ethucy_indomain_quick/raw/selected_cases.json

============================================================
5. Geometry usage checkout
============================================================

Question:
Is geometry integrated into the reconstruction method, or only used afterward as evaluation?

- Direct answer: Geometry is evaluation-only
- Used in training: NO
- Used in conditioning: NO
- Used in loss: NO
- Used in sampling: NO
- Used in rejection: NO
- Used only in evaluation: YES
- Can the report call this geometry-aware reconstruction? NO
- Correct term: geometry feasibility evaluation
- Evidence:
  - files/functions inspected: tools/stage3/eval/eval_geometry_metrics.py, tools/stage3/geometry_extension/run_geometry_extension.py, tools/stage3/refinement/ddpm_refiner.py
  - geometry-related scripts: tools/stage3/eval/eval_geometry_metrics.py, tools/stage3/geometry_extension/run_geometry_extension.py, tools/stage3/refinement/ddpm_refiner.py
  - geometry-related result files: outputs/stage3/geometry_extension/geometry_profiles_summary.md, outputs/stage3/geometry_extension/geometry_profiles_summary.csv

============================================================
6. Final direct verdict
============================================================

| Issue | Direct verdict | Needs correction? | Minimal correction |
|---|---|---|---|
| Table 2 completeness | Table 2 hides information because it reports only winners / selected values | YES | Replace with a full method × condition × metric matrix and expose raw-population stats or state that they are unavailable |
| Figure 3 spread | Figure 3 spread is misleading | YES | State that the bars use std across seed-level summary rows, not per-trajectory raw dispersion |
| Figure 5 spread | Figure 5 spread is ambiguous | YES | State that the envelope is min/max across degradation × coarse-method means, not DDPM seed variance |
| DDPM trajectory figures | DDPM trajectory figures are incomplete | YES | Add a figure with clean, degraded, coarse, DDPM candidate, and final refined output from the same trace |
| Geometry wording | Geometry is evaluation-only | YES | Rename as geometry feasibility evaluation, not geometry-aware reconstruction |
