# Stage 3 Current Status

## Mainline

The active Stage 3 mainline has two layers:

1. Phase 1 `canonical_room3` benchmark
2. Controlled benchmark for coarse reconstruction stress testing

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

## Controlled Benchmark

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

Compatibility wrappers are kept under `tools/stage3/` so older direct paths do not break immediately.
