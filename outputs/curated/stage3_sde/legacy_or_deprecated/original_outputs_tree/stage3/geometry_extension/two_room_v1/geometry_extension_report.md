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

- `profile_name`: `two_room_v1`
- Main constraint: internal wall with narrow opening y in [1.35, 1.65]
- Room boundary: `[0, 3] x [0, 3]`

## Filter Summary

| profile_name | total_windows | feasible_windows | discarded_windows | retention_rate | same_room_windows | room_transition_windows | valid_crossing_windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_room_v1 | 36073 | 27544 | 8529 | 0.763563 | 25053 | 2491 | 2491 |

## Summary By Source Family

| source_family | evaluated_windows | window_violation_rate | infeasible_transition_rate | mean_infeasible_transitions_per_window | masked_infeasible_transition_rate | off_map_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| controlled_benchmark | 440704 | 0.028096 | 0.002695 | 0.051209 | 0.001438 | 0.000159 |
| phase1_baseline | 2148432 | 0.003925 | 0.000290 | 0.005507 | 0.000841 | 0.000023 |
| refinement | 1652640 | 0.027671 | 0.002474 | 0.047004 | 0.001934 | 0.000231 |
| refinement_alpha_sweep | 2313696 | 0.028727 | 0.002643 | 0.050216 | 0.001946 | 0.000175 |


## Output Files

- `feasible_indices.npy`
- `geometry_filter_summary.csv`
- `geometry_metrics.csv`
- `geometry_summary.csv`
- `geometry_extension_report.md`
- `figures/two_room_v1_layout.png`
- `figures/geometry_violation_summary.png`
