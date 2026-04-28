# Stage 3 Output Layout

This directory now keeps the active Stage 3 outputs at the top level and
pushes older flat-layout leftovers into `archive/`.

## Active directories

- `phase1/`
  Canonical `canonical_room3` Phase 1 benchmark outputs, including data,
  baselines, evaluation, and random-span statistics.
- `controlled_benchmark/`
  Controlled coarse reconstruction degradation, reconstruction, evaluation,
  and figure outputs.
- `refinement/`
  Refinement outputs, metrics, figures, and alpha-sweep results.
- `geometry_extension/`
  Geometry feasibility extension outputs for `obstacle_v1` and `two_room_v1`.

## Archive

- `archive/legacy_flat_outputs/`
  Older flat Stage 3 output paths that were previously placed directly under
  `outputs/stage3/`. They are retained only for historical reference and are
  not the active output layout anymore.

## Presentation figures

Curated presentation copies now live only under:

- `docs/assets/stage3/`

## Cleanup rule

If a new Stage 3 result belongs to an existing layer, place it inside that
layer's directory instead of creating a new flat directory directly under
`outputs/stage3/`.
