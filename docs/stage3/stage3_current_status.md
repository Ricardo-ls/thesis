# Stage 3 Current Status

## Mainline

The active Stage 3 mainline now has five layers:

1. Phase 1 `canonical_room3` benchmark
2. Random-span statistical evaluation
3. Controlled coarse reconstruction benchmark
4. DDPM refinement interface and alpha sweep
5. Geometry feasibility extension: `wall_door_v1`, `obstacle_v1`, and `two_room_v1`

Stage 2 prior and DDPM work is complete background context and is not changed by Stage 3 repository reorganization.

## Phase 1 Canonical Room3

Current benchmark root:

- `outputs/stage3/phase1/canonical_room3/`

Current scope:

- one contiguous missing span
- baselines: Linear, Savitzky-Golay, Kalman
- metric views:
  - full trajectory: `ADE`, `FDE`, `RMSE`
  - missing segment: `masked_ADE`, `masked_RMSE`
  - geometry checks: `off_map_ratio`, `wall_crossing_count`

Current documentation:

- [`phase1_canonical_room3/stage3_phase1_room3_protocol.md`](phase1_canonical_room3/stage3_phase1_room3_protocol.md)
- [`phase1_canonical_room3/stage3_phase1_runbook.md`](phase1_canonical_room3/stage3_phase1_runbook.md)
- [`phase1_canonical_room3/stage3_phase1_experiment_checklist.md`](phase1_canonical_room3/stage3_phase1_experiment_checklist.md)

## Random-Span Statistics

Current random-span statistics root:

- `outputs/stage3/phase1/canonical_room3/random_span_statistics/`

Standard outputs:

- `metrics_by_seed.csv`
- `metrics_summary_mean_std.csv`
- `random_span_statistics_report.md`
- `figures/ADE_mean_std_bar.png`
- `figures/RMSE_mean_std_bar.png`
- `figures/masked_ADE_mean_std_bar.png`
- `figures/full_vs_masked_comparison.png`

Current role:

- test whether Phase 1 rankings remain stable across many span positions and seeds
- report both full-trajectory consistency and missing-segment reconstruction quality
- make clear that masked metrics are not auxiliary, because they directly measure the missing segment

## Controlled Coarse Reconstruction Benchmark

Current controlled benchmark root:

- `outputs/stage3/controlled_benchmark/`

Current subfolders:

- `degradation/`
- `reconstruction/`
- `eval/`
- `figures/`

Current script entry points:

- `python -m tools.stage3.controlled.build_controlled_degradation`
- `python -m tools.stage3.controlled.run_coarse_reconstruction_baselines`
- `python -m tools.stage3.controlled.evaluate_coarse_reconstruction`

Current role:

- keep the fixed `canonical_room3` benchmark unchanged
- add a separate controlled stress-test layer with `missing_only`, `missing_noise`, `missing_drift`, and `missing_noise_drift`
- evaluate how the standard coarse baselines behave before any learned-prior refinement is applied

Compatibility wrappers are kept under `tools/stage3/` so older direct paths do not break immediately.

## DDPM Refinement Interface And Alpha Sweep

Current refinement roots:

- `outputs/stage3/refinement/`
- `outputs/stage3/refinement/alpha_sweep/`

Current refinement scope:

- naive refiners: `identity_refiner`, `light_savgol_refiner`
- DDPM prior interfaces:
  - `ddpm_prior_interface_v0`
  - `ddpm_prior_masked_replace_v1`
  - `ddpm_prior_masked_blend_v2_alpha0.25`
- alpha sweep for `ddpm_prior_masked_blend_v2`:
  - `alpha = 0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00`

Current interpretation:

- `v0` verifies that the Stage 2 prior checkpoint can be connected to Stage 3 refinement
- `v1` preserves observed points but still relies directly on the same DDPM candidate over the missing span
- `v2` adds soft blending on the missing span
- the current alpha sweep shows that large alpha usually hurts, so the unconditional DDPM prior is not yet a reliable direct missing-segment refiner

## Geometry Feasibility Extension

Current geometry extension root:

- `outputs/stage3/geometry_extension/`

Current profiles:

- `wall_door_v1`
- `obstacle_v1`
- `two_room_v1`

Current geometry extension scope:

- all profiles are separate geometry feasibility extensions
- they do not replace `canonical_room3`
- they are synthetic stress tests
- clean target trajectories are filtered first
- normalized violation rates are emphasized over raw counts
- existing Phase 1, controlled benchmark, refinement, and alpha-sweep outputs are re-evaluated on the feasible clean-target subset only

Profile summary:

| Profile | Main constraint | Retention rate |
| --- | --- | ---: |
| `wall_door_v1` | internal wall with door opening | 0.832035 |
| `obstacle_v1` | central blocked obstacle | 0.704959 |
| `two_room_v1` | internal wall with narrow transition opening | 0.763563 |

Current geometry metrics:

- `off_map_ratio`
- `infeasible_transition_count`
- `window_violation_rate`
- `infeasible_transition_rate`
- `mean_infeasible_transitions_per_window`

Current interpretation:

- `obstacle_v1` is currently the strongest geometry stress test by `window_violation_rate`
- `two_room_v1` is stricter than `wall_door_v1` because it narrows the valid transition opening
- existing refinement outputs do not show a clear geometry-feasibility advantage over each other
- this layer is evaluation-only, not a new reconstruction method

Important limitation:

- these geometry profiles are synthetic feasibility stress tests applied after scaling ETH+UCY trajectories into `canonical_room3`
- their violation counts should be interpreted as feasibility diagnostics under artificial layouts, not as direct evidence of real-room navigation failure
