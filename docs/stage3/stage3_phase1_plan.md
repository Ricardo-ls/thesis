# Stage 3 Phase 1 Plan

## Scope

Stage 2 remains a completed prior-learning study and is not expanded in this phase.

Stage 3 Phase 1 uses `datasets/processed/data_eth_ucy_20.npy` as the clean window input and establishes a minimal indoor trajectory imputation benchmark with one contiguous missing span.

- input: degraded coarse absolute trajectory with one contiguous missing span
- output: completed absolute trajectory
- target: clean trajectory

## Baselines

The first comparison layer is intentionally simple:

- linear interpolation
- Savitzky-Golay smoothing
- constant-velocity Kalman filter

## Metrics

Reconstruction metrics:

- ADE
- FDE
- RMSE

Geometry feasibility metrics:

- wall-crossing count
- off-map ratio

The minimal execution order is recorded in `docs/stage3/stage3_run_checklist.md`.

## Not Included

This phase does not include:

- raw-data reconstruction
- prior integration
- q20 comparison
- multi-dataset comparison
- learning backbone design
- complex geometry conditioning
