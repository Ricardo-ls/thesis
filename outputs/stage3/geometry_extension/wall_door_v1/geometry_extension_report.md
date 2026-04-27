# Stage 3 Geometry Extension Report

## Scope

The fixed `canonical_room3` benchmark remains unchanged.
This report adds a separate geometry extension profile, `canonical_room3_wall_door_v1`, to test indoor feasibility constraints on already-generated trajectories.

## Geometry Profile

- Room boundary: `[0, 3] x [0, 3]`
- Internal wall: vertical wall at `x = 1.5`
- Door opening: `y in [1.2, 1.8]`
- Side-to-side crossings of the internal wall are feasible only if the crossing point lies inside the door interval.

## Metric Interpretation

- `off_map_ratio`: fraction of trajectory points outside the room boundary.
- `boundary_violation_count`: count of points outside the room boundary.
- `internal_wall_crossing_count`: count of side-to-side crossings of the internal wall.
- `door_valid_crossing_count`: subset of wall crossings that pass through the door opening.
- `infeasible_transition_count`: count of transitions that either use an invalid wall crossing or have an off-boundary endpoint.

## Source Coverage

- Evaluated rows: `238`
- Source families covered: `controlled_benchmark, phase1_baseline, refinement, refinement_alpha_sweep`

## Summary By Source Family

| source_family | off_map_ratio | boundary_violation_count | internal_wall_crossing_count | door_valid_crossing_count | door_crossing_valid_ratio | infeasible_transition_count |
| --- | --- | --- | --- | --- | --- | --- |
| controlled_benchmark | 0.000144 | 1659 | 926774 | 83316 | 0.089899 | 845526 |
| phase1_baseline | 0.000020 | 1112 | 885876 | 393836 | 0.444572 | 493674 |
| refinement | 0.000194 | 8390 | 757377 | 316527 | 0.417925 | 450796 |
| refinement_alpha_sweep | 0.000155 | 9419 | 1145075 | 474093 | 0.414028 | 682654 |

## Highest-Violation Rows

| source_family | experiment_id | degradation | coarse_method | method_name | alpha | infeasible_transition_count | internal_wall_crossing_count | door_valid_crossing_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled_benchmark | span20_fixed_seed42 | missing_noise_drift |  | input_degraded |  | 189929 | 194682 | 5023 |
| controlled_benchmark | span20_fixed_seed42 | missing_noise |  | input_degraded |  | 189898 | 194694 | 4936 |
| controlled_benchmark | span20_fixed_seed42 | missing_drift |  | input_degraded |  | 185025 | 188656 | 3697 |
| controlled_benchmark | span20_fixed_seed42 | missing_only |  | input_degraded |  | 184964 | 188589 | 3625 |
| refinement |  | missing_noise_drift | linear_interp | ddpm_prior_masked_replace_v1 |  | 12291 | 19166 | 7223 |
| refinement_alpha_sweep |  | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha1.00 | 1.000000 | 12291 | 19166 | 7223 |
| refinement |  | missing_noise | linear_interp | ddpm_prior_masked_replace_v1 |  | 12243 | 19154 | 7120 |
| refinement_alpha_sweep |  | missing_noise | linear_interp | ddpm_prior_masked_blend_v2_alpha1.00 | 1.000000 | 12243 | 19154 | 7120 |
| refinement_alpha_sweep |  | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.75 | 0.750000 | 11884 | 18464 | 6915 |
| refinement_alpha_sweep |  | missing_noise | linear_interp | ddpm_prior_masked_blend_v2_alpha0.75 | 0.750000 | 11873 | 18482 | 6800 |
| refinement_alpha_sweep |  | missing_noise_drift | linear_interp | ddpm_prior_masked_blend_v2_alpha0.50 | 0.500000 | 11572 | 17848 | 6595 |
| refinement_alpha_sweep |  | missing_noise | linear_interp | ddpm_prior_masked_blend_v2_alpha0.50 | 0.500000 | 11542 | 17860 | 6486 |

## Interpretation

This extension does not redefine the Stage 3 reconstruction task and does not replace the canonical empty-room protocol.
It adds a stricter indoor feasibility lens so that future reconstruction or refinement methods can be checked for physically valid transitions under a simple wall-and-door layout.

## Output Files

- `geometry_metrics.csv`
- `geometry_summary.csv`
- `geometry_extension_report.md`
- `figures/geometry_violation_summary.png`
