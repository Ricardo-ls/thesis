# Stage 3 Docs

Stage 3 is organized around five active benchmark and evaluation layers:

1. Phase 1 `canonical_room3` benchmark
2. Random-span statistical evaluation
3. Controlled coarse reconstruction benchmark
4. DDPM refinement interface and alpha sweep
5. Geometry feasibility extension: `wall_door_v1`, `obstacle_v1`, and `two_room_v1`

The documentation tree is grouped into four folders:

- [`phase1_canonical_room3/`](phase1_canonical_room3/): active Phase 1 benchmark protocol, runbook, figures, and experiment checklist
- [`controlled_benchmark/`](controlled_benchmark/): controlled coarse-to-refined benchmark notes and navigation
- [`planning/`](planning/): formal scope statements and planning documents
- [`archive/`](archive/): older checklists and superseded navigation files kept for traceability

Recommended reading order:

1. [`stage3_current_status.md`](stage3_current_status.md)
2. [`planning/stage3_phase1_formal_spec.md`](planning/stage3_phase1_formal_spec.md)
3. [`phase1_canonical_room3/stage3_phase1_room3_protocol.md`](phase1_canonical_room3/stage3_phase1_room3_protocol.md)
4. [`phase1_canonical_room3/stage3_phase1_runbook.md`](phase1_canonical_room3/stage3_phase1_runbook.md)
5. [`phase1_canonical_room3/stage3_phase1_experiment_checklist.md`](phase1_canonical_room3/stage3_phase1_experiment_checklist.md)
6. [`../stage3_geometry_extension_protocol.md`](../stage3_geometry_extension_protocol.md)

Script navigation:

- Phase 1 data: `tools/stage3/data/`
- Phase 1 baselines: `tools/stage3/baselines/`
- Phase 1 evaluation: `tools/stage3/eval/`
- Controlled benchmark: `tools/stage3/controlled/`
- Refinement and alpha sweep: `tools/stage3/refinement/`
- Geometry feasibility extension: `tools/stage3/geometry_extension/`

Output navigation:

- `outputs/stage3/phase1/canonical_room3/`
- `outputs/stage3/phase1/canonical_room3/random_span_statistics/`
- `outputs/stage3/controlled_benchmark/`
- `outputs/stage3/refinement/`
- `outputs/stage3/refinement/alpha_sweep/`
- `outputs/stage3/geometry_extension/`
- `outputs/stage3/geometry_extension/wall_door_v1/`
- `outputs/stage3/geometry_extension/obstacle_v1/`
- `outputs/stage3/geometry_extension/two_room_v1/`
- `outputs/stage3/geometry_extension/geometry_profiles_summary.csv`
- `outputs/stage3/geometry_extension/geometry_profiles_summary.md`
- `docs/assets/stage3/geometry_profiles_comparison.png`

Geometry-feasibility note:

- `wall_door_v1`, `obstacle_v1`, and `two_room_v1` are synthetic feasibility stress tests
- they do not replace `canonical_room3`
- clean target windows are filtered first, and normalized violation rates are emphasized in the final reporting
