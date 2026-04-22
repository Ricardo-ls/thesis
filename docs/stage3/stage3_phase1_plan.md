# Stage 3 Phase 1 Plan

## Scope

Stage 2 remains a completed prior-learning study and is not expanded in this phase.

Stage 3 Phase 1 is defined as a minimal indoor trajectory imputation benchmark:

- input: degraded coarse absolute trajectory with one contiguous missing span
- output: completed absolute trajectory
- target: clean trajectory reconstruction

## Current Benchmark

The first comparison layer is intentionally simple:

- linear interpolation
- Savitzky-Golay smoothing
- constant-velocity Kalman filter

The purpose of this phase is to establish a stable benchmark before any learning-based or prior-based extension is introduced.

## Current Evaluation

Reconstruction metrics:

- ADE
- FDE
- RMSE

Geometry feasibility metrics:

- wall-crossing count
- off-map ratio

## Current Data Interface

The minimal data path is:

1. clean indoor trajectory windows
2. one contiguous missing span per window
3. baseline reconstruction
4. metric evaluation

The geometry interface is limited to an occupancy map / free-space mask.

## Explicit Non-Goals

This phase does not include:

- prior integration
- q20 comparison
- multi-dataset comparison
- complex geometry conditioning
- complex backbone design
