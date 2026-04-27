# Stage 3 Geometry Extension Report

## Scope

These profiles are geometry feasibility extensions.
They do not replace `canonical_room3`.
They do not define new reconstruction methods.
They are synthetic feasibility stress tests.
Clean target windows are first filtered for each geometry profile.
Geometry metrics are interpreted only on the clean-target feasible subset.
Normalized rates are emphasized over raw counts.

## Geometry Profile

- `profile_name`: `obstacle_v1`
- Main constraint: central rectangular obstacle [1.2, 1.8] x [1.2, 1.8]
- Room boundary: `[0, 3] x [0, 3]`

## Filter Summary

| profile_name | total_windows | feasible_windows | discarded_windows | retention_rate | obstacle_free_same_region_windows | obstacle_bypass_windows |
| --- | --- | --- | --- | --- | --- | --- |
| obstacle_v1 | 36073 | 25430 | 10643 | 0.704959 | 23617 | 1813 |

## Summary By Source Family

| source_family | evaluated_windows | window_violation_rate | infeasible_transition_rate | mean_infeasible_transitions_per_window | masked_infeasible_transition_rate | off_map_ratio | obstacle_point_violation_rate | obstacle_segment_crossing_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled_benchmark | 406880 | 0.280771 | 0.075752 | 1.439284 | 0.256688 | 0.000188 | 0.007124 | 0.075501 |
| phase1_baseline | 1983540 | 0.002366 | 0.000321 | 0.006107 | 0.001067 | 0.000024 | 0.000168 | 0.000283 |
| refinement | 1525800 | 0.043657 | 0.012223 | 0.232237 | 0.013295 | 0.000262 | 0.009228 | 0.011894 |
| refinement_alpha_sweep | 2136120 | 0.043702 | 0.011027 | 0.209506 | 0.012203 | 0.000205 | 0.007921 | 0.010756 |


## Output Files

- `feasible_indices.npy`
- `geometry_filter_summary.csv`
- `geometry_metrics.csv`
- `geometry_summary.csv`
- `geometry_extension_report.md`
- `figures/obstacle_v1_layout.png`
- `figures/geometry_violation_summary.png`
