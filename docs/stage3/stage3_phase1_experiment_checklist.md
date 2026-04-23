# Stage 3 Phase 1 Experiment Checklist

## Common Setup

Canonical root:

- `outputs/stage3/phase1/canonical_room3/`

Common input:

- `outputs/stage3/phase1/canonical_room3/data/clean_windows_room3.npz`

Common occupancy map:

- `outputs/stage3/phase1/canonical_room3/data/occupancy_map_room3_empty.npz`

Common baselines:

- `linear_interp`
- `savgol_w5_p2`
- `kalman_cv_dt1.0_q1e-3_r1e-2`

Common metrics:

- ADE
- FDE
- RMSE
- masked_ADE
- masked_RMSE
- off-map ratio
- wall-crossing count

## Experiment 0: Main Table

Experiment id:

- `span20_fixed_seed42`

Input file:

- `outputs/stage3/phase1/canonical_room3/data/experiments/span20_fixed_seed42/missing_span_windows.npz`

Baseline output directory:

- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/`

Evaluation output directory:

- `outputs/stage3/phase1/canonical_room3/eval/span20_fixed_seed42/`

## Experiment 1: Missing-Length Sweep

Experiment ids:

- `span10_fixed_seed42`
- `span20_fixed_seed42`
- `span30_fixed_seed42`

Input files:

- `outputs/stage3/phase1/canonical_room3/data/experiments/<experiment_id>/missing_span_windows.npz`

Baseline output directories:

- `outputs/stage3/phase1/canonical_room3/baselines/<experiment_id>/<method_tag>/`

Evaluation output directories:

- `outputs/stage3/phase1/canonical_room3/eval/<experiment_id>/<method_tag>/`

## Experiment 2: Missing-Position Control

Experiment ids:

- `span20_fixed_seed42`
- `span20_random_seed42`
- `span20_random_seed43`
- `span20_random_seed44`

Input files:

- `outputs/stage3/phase1/canonical_room3/data/experiments/<experiment_id>/missing_span_windows.npz`

Baseline output directories:

- `outputs/stage3/phase1/canonical_room3/baselines/<experiment_id>/<method_tag>/`

Evaluation output directories:

- `outputs/stage3/phase1/canonical_room3/eval/<experiment_id>/<method_tag>/`
