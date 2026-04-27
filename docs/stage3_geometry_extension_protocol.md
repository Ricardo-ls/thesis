# Stage 3 Geometry Extension Protocol

## Purpose

This document defines a minimal indoor geometry extension for Stage 3 evaluation.

The fixed `canonical_room3` benchmark remains the reference protocol and is not replaced by this extension.

The new profile is:

- `canonical_room3_wall_door_v1`

Its role is to add a simple physical-feasibility check on top of already-generated trajectories.

## Relationship To The Existing Benchmark

- `canonical_room3` remains the fixed Stage 3 Phase 1 reference benchmark.
- The controlled benchmark remains unchanged.
- Existing DDPM refinement outputs and alpha sweep outputs remain unchanged.
- This extension is an evaluation layer only.
- It does not introduce a new reconstruction method.
- It does not change full-trajectory or masked reconstruction metrics such as `ADE`, `FDE`, `RMSE`, `masked_ADE`, or `masked_RMSE`.

## Geometry Profile: `canonical_room3_wall_door_v1`

- Room boundary: `[0, 3] x [0, 3]`
- Internal wall: vertical wall at `x = 1.5`
- Door opening on that wall: `y in [1.2, 1.8]`

Interpretation:

- A trajectory point outside the room boundary is a boundary violation.
- A trajectory segment that crosses from one side of `x = 1.5` to the other is an internal wall crossing.
- That wall crossing is feasible only when the crossing point lies within the door interval.
- Otherwise the crossing is an infeasible wall transition.

## Geometry Metrics

The extension adds the following geometry checks:

- `off_map_ratio`
  Fraction of trajectory points that lie outside the room boundary.

- `boundary_violation_count`
  Count of trajectory points outside the room boundary.

- `internal_wall_crossing_count`
  Count of side-to-side crossings of the internal wall.

- `door_valid_crossing_count`
  Count of internal wall crossings that pass through the door interval.

- `infeasible_transition_count`
  Count of transitions that are physically infeasible under this profile.
  In the current implementation this includes:
  - transitions with an off-boundary endpoint
  - internal wall crossings that do not pass through the door

## Scope Of Application

This extension is applied to existing Stage 3 outputs when available:

- Phase 1 baseline reconstructions
- Controlled benchmark reconstructions
- Refinement outputs
- Refinement alpha sweep outputs

It is intentionally a secondary feasibility layer, not a replacement protocol.

## Output Location

All extension outputs are written under:

- `outputs/stage3/geometry_extension/wall_door_v1/`

Expected files:

- `geometry_metrics.csv`
- `geometry_summary.csv`
- `geometry_extension_report.md`
- `figures/geometry_violation_summary.png`

## Intended Use

This extension addresses the methodological point that the empty `3m x 3m` room only tests boundary feasibility.

The wall-door profile provides a minimal indoor-structure check so future reconstruction or refinement methods can be assessed not only for reconstruction error, but also for whether their trajectories remain physically plausible under simple indoor constraints.
