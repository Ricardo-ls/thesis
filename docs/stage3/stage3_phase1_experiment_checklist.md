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

Interpretation rule:

- `ADE`, `FDE`, and `RMSE` are the full-trajectory view.
- `masked_ADE` and `masked_RMSE` are the missing-segment view.
- Since the task is missing-segment reconstruction, the masked view should be
  emphasized when discussing gap-filling quality.
- If the full and masked views rank methods differently, both rankings should
  be reported explicitly.

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

- `outputs/stage3/phase1/canonical_room3/data/experiments/span10_fixed_seed42/missing_span_windows.npz`
- `outputs/stage3/phase1/canonical_room3/data/experiments/span20_fixed_seed42/missing_span_windows.npz`
- `outputs/stage3/phase1/canonical_room3/data/experiments/span30_fixed_seed42/missing_span_windows.npz`

Baseline output directories:

- `outputs/stage3/phase1/canonical_room3/baselines/span10_fixed_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/baselines/span10_fixed_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span10_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span30_fixed_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/baselines/span30_fixed_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span30_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`

Evaluation output directories:

- `outputs/stage3/phase1/canonical_room3/eval/span10_fixed_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/eval/span10_fixed_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/eval/span10_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_fixed_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_fixed_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/eval/span30_fixed_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/eval/span30_fixed_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/eval/span30_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`

## Experiment 2: Missing-Position Control

Experiment ids:

- `span20_fixed_seed42`
- `span20_random_seed42`
- `span20_random_seed43`
- `span20_random_seed44`

Input files:

- `outputs/stage3/phase1/canonical_room3/data/experiments/span20_fixed_seed42/missing_span_windows.npz`
- `outputs/stage3/phase1/canonical_room3/data/experiments/span20_random_seed42/missing_span_windows.npz`
- `outputs/stage3/phase1/canonical_room3/data/experiments/span20_random_seed43/missing_span_windows.npz`
- `outputs/stage3/phase1/canonical_room3/data/experiments/span20_random_seed44/missing_span_windows.npz`

Baseline output directories:

- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed43/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed43/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed43/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed44/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed44/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_random_seed44/kalman_cv_dt1.0_q1e-3_r1e-2/`

Evaluation output directories:

- `outputs/stage3/phase1/canonical_room3/eval/span20_fixed_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_fixed_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed42/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed42/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed43/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed43/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed43/kalman_cv_dt1.0_q1e-3_r1e-2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed44/linear_interp/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed44/savgol_w5_p2/`
- `outputs/stage3/phase1/canonical_room3/eval/span20_random_seed44/kalman_cv_dt1.0_q1e-3_r1e-2/`
