# Stage 3 Phase 1 Room3 Protocol

## Input

The current Stage 3 Phase 1 proxy input is fixed to:

- `datasets/processed/data_eth_ucy_20.npy`

The input is expected to be pre-windowed absolute trajectories with shape
`[N, T, 2]`.

## Source Coordinate Range

The current source coordinate bounds are fixed as:

- `x_min = -7.69`
- `x_max = 15.613144`
- `y_min = -1.81`
- `y_max = 13.89191`

These values are written into
`outputs/stage3/phase1/canonical_room3/data/clean_windows_room3_meta.json`
when the clean room3 dataset is built.

## Target Room

The canonical benchmark room is:

- `[0, 3] x [0, 3]`

The corresponding empty-room occupancy map is saved as:

- `outputs/stage3/phase1/canonical_room3/data/occupancy_map_room3_empty.npz`

## Normalization

The current normalization mode is separate linear scaling:

- x is mapped from `[-7.69, 15.613144]` to `[0, 3]`
- y is mapped from `[-1.81, 13.89191]` to `[0, 3]`

This phase does not use isotropic scaling and does not attempt complex
geometry-preserving reconstruction.

## Interpretation Boundary

The canonical room3 setup exists to make Stage 3 Phase 1 benchmark interfaces,
result directories, and geometry evaluation comparable and reproducible.

It is a canonical benchmark room for this proxy experiment. It does not directly
support conclusions about a future real experimental room.
