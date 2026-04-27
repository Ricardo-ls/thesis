# Stage 3 Geometry Extension Report

## Scope

The fixed `canonical_room3` benchmark remains unchanged.
This report keeps `wall_door_v1` as a separate geometry feasibility extension rather than a replacement benchmark.
It does not introduce a new reconstruction method and does not change the existing reconstruction metrics.

## Geometry Profile

- Room boundary: `[0, 3] x [0, 3]`
- Internal wall: vertical wall at `x = 1.5`
- Door opening: `y in [1.2, 1.8]`
- A side-to-side crossing of `x = 1.5` is feasible only if the crossing point lies within the door interval.

## Feasible Clean-Target Filtering

Before evaluating reconstructed or refined trajectories, clean target windows are filtered under the synthetic wall-door layout.
A clean window is retained only if:

- `clean_off_map_ratio == 0`
- `clean_infeasible_transition_count == 0`

Infeasible clean windows are excluded from the main geometry evaluation but are not deleted from the repository.
All main geometry metrics below are interpreted only on the feasible subset.

## Filter Summary

| total_windows | feasible_windows | discarded_windows | retention_rate | same_side_windows | door_transition_windows |
| --- | --- | --- | --- | --- | --- |
| 36073 | 30014 | 6059 | 0.832035 | 25053 | 4961 |

## Main Reported Metrics

Large raw counts are secondary in this revised experiment.
The primary geometry diagnostics are:

- `window_violation_rate`
- `infeasible_transition_rate`
- `mean_infeasible_transitions_per_window`
- `masked_infeasible_transition_rate` when a missing mask is available

## Summary By Source Family

| source_family | evaluated_windows | window_violation_rate | infeasible_transition_rate | mean_infeasible_transitions_per_window | masked_infeasible_transition_rate | off_map_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| controlled_benchmark | 480224 | 0.021592 | 0.002047 | 0.038884 | 0.001085 | 0.000150 |
| phase1_baseline | 2341092 | 0.002381 | 0.000195 | 0.003707 | 0.000614 | 0.000023 |
| refinement | 1800840 | 0.021012 | 0.001889 | 0.035888 | 0.001456 | 0.000215 |
| refinement_alpha_sweep | 2521176 | 0.021748 | 0.001997 | 0.037939 | 0.001445 | 0.000165 |

## Highest-Violation Rows By Normalized Rate

| source_family | experiment_id | degradation | coarse_method | method_name | alpha | window_violation_rate | infeasible_transition_rate | mean_infeasible_transitions_per_window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| refinement | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_replace_v1 |  | 0.040914 | 0.005010 | 0.095189 |
| refinement_alpha_sweep | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha1.00 | 1.000000 | 0.040914 | 0.005010 | 0.095189 |
| refinement_alpha_sweep | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.75 | 0.750000 | 0.040181 | 0.004892 | 0.092957 |
| refinement_alpha_sweep | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.50 | 0.500000 | 0.039781 | 0.004805 | 0.091291 |
| refinement | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.25 | 0.250000 | 0.039548 | 0.004745 | 0.090158 |
| refinement_alpha_sweep | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.25 | 0.250000 | 0.039548 | 0.004745 | 0.090158 |
| refinement_alpha_sweep | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.10 | 0.100000 | 0.039348 | 0.004726 | 0.089791 |
| refinement_alpha_sweep | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.05 | 0.050000 | 0.039348 | 0.004724 | 0.089758 |
| controlled_benchmark | span20_fixed_seed42 | missing_noise_drift |  | linear_interp |  | 0.039315 | 0.004717 | 0.089625 |
| refinement | span20_fixed_seed42 | missing_noise_drift | linear_interp | identity_refiner |  | 0.039315 | 0.004717 | 0.089625 |
| refinement_alpha_sweep | span20_fixed_seed42 | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.00 | 0.000000 | 0.039315 | 0.004717 | 0.089625 |
| controlled_benchmark | span20_fixed_seed42 | missing_noise_drift |  | input_degraded |  | 0.037782 | 0.004415 | 0.083894 |

## Interpretation

The revised `wall_door_v1` experiment is a feasibility diagnostic on a feasible clean-target subset, not a new reconstruction benchmark and not a new trajectory generator.
Its purpose is to test whether existing Stage 3 outputs remain physically plausible under a simple wall-and-door indoor layout.

## Limitation

`wall_door_v1` is a synthetic feasibility stress test applied after scaling ETH+UCY trajectories into `canonical_room3`.
Its violation counts should be interpreted as feasibility diagnostics under this artificial layout, not as direct evidence of real-room navigation failure.

## Output Files

- `feasible_indices.npy`
- `geometry_filter_summary.csv`
- `geometry_metrics.csv`
- `geometry_summary.csv`
- `geometry_extension_report.md`
- `figures/geometry_violation_summary.png`
