# Stage 3 Geometry Extension Protocol

## Purpose

This document defines Stage 3 geometry feasibility extensions for evaluation.

These profiles do not replace `canonical_room3`.
They do not define new reconstruction methods.
They are synthetic feasibility stress tests layered on top of already-generated trajectories.

## Relationship To Existing Stage 3 Assets

- `canonical_room3` remains the fixed Stage 3 Phase 1 reference benchmark.
- Stage 2 prior checkpoints remain unchanged.
- Controlled benchmark degradation definitions remain unchanged.
- The DDPM sampling formula remains unchanged.
- Existing reconstruction metrics remain unchanged.
- This extension adds geometry-feasibility evaluation layers only.

## Shared Evaluation Principle

For each geometry profile:

1. Clean target windows are first filtered for feasibility under that profile.
2. Existing reconstructed and refined outputs are evaluated only on the clean-target feasible subset.
3. Normalized rates are emphasized over raw counts.

Primary normalized metrics:

- `window_violation_rate`
- `infeasible_transition_rate`
- `mean_infeasible_transitions_per_window`
- `masked_infeasible_transition_rate` when a missing mask is available

## Geometry Profiles

### 1. `obstacle_v1`

- Room boundary: `[0, 3] x [0, 3]`
- Central blocked obstacle: `[1.2, 1.8] x [1.2, 1.8]`

### 2. `two_room_v1`

- Room boundary: `[0, 3] x [0, 3]`
- Internal wall: vertical wall at `x = 1.5`
- Narrow transition opening: `y in [1.35, 1.65]`

## Interpretation

- These profiles are geometry feasibility extensions.
- They do not replace `canonical_room3`.
- They do not define new reconstruction methods.
- They are synthetic feasibility stress tests.
- Clean target windows are first filtered for each geometry profile.
- Geometry metrics are interpreted only on the clean-target feasible subset.
- Normalized rates are emphasized over raw counts.

## Limitation

If a profile has `retention_rate < 0.10`, do not make strong conclusions.

Use this warning:

"The scaled ETH+UCY trajectories are poorly matched to this synthetic geometry profile. This profile should be treated only as a preliminary feasibility stress test. A future geometry-controlled synthetic trajectory set may be needed."
