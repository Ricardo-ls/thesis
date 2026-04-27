# Controlled Benchmark

This directory documents the controlled Stage 3 benchmark layer that sits after
the fixed Phase 1 `canonical_room3` benchmark.

Primary code entry points:

- `python -m tools.stage3.controlled.build_controlled_degradation`
- `python -m tools.stage3.controlled.run_coarse_reconstruction_baselines`
- `python -m tools.stage3.controlled.evaluate_coarse_reconstruction`

Primary output root:

- `outputs/stage3/controlled_benchmark/`

Subdirectories:

- `degradation/`
- `reconstruction/`
- `eval/`
- `figures/`
