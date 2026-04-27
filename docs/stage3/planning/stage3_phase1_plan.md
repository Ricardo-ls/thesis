# Stage 3 Phase 1 Plan

## Scope

Stage 2 is treated as complete and is not expanded in this phase.

Stage 3 Phase 1 is benchmark-first. The current benchmark uses the canonical
room3 coordinate system and fixes the working room to:

- `x in [0, 3]`
- `y in [0, 3]`

The source input remains:

- `datasets/processed/data_eth_ucy_20.npy`

## Included

This phase includes only:

- one contiguous missing span per trajectory window
- linear interpolation
- Savitzky-Golay smoothing
- constant-velocity Kalman filtering
- ADE
- FDE
- RMSE
- masked_ADE
- masked_RMSE
- off-map ratio
- wall-crossing count

## Not Included

This phase does not include:

- prior integration
- q20 comparison
- adapter design
- multi-dataset comparison
- learning backbones
- complex geometry conditioning
- complex obstacle maps

The canonical room3 protocol is documented in
`docs/stage3/phase1_canonical_room3/stage3_phase1_room3_protocol.md`.

Metric interpretation uses two complementary views:

- full-trajectory consistency: `ADE`, `FDE`, `RMSE`
- missing-segment reconstruction quality: `masked_ADE`, `masked_RMSE`

Since the task is missing-segment reconstruction, the masked view should be
emphasized when discussing gap recovery quality.
