# Geometry Usage Checkout

Direct answer: Geometry is evaluation-only

- Used in training: NO
- Used in conditioning: NO
- Used in loss: NO
- Used in sampling: NO
- Used in rejection: NO
- Used only in evaluation: YES
- Can the report call this geometry-aware reconstruction? NO
- Correct term: geometry feasibility evaluation

Evidence:
- scripts: `tools/stage3/eval/eval_geometry_metrics.py`, `tools/stage3/geometry_extension/run_geometry_extension.py`, `tools/stage3/refinement/ddpm_refiner.py`
- result files: `outputs/stage3/geometry_extension/geometry_profiles_summary.md`, `outputs/stage3/geometry_extension/geometry_profiles_summary.csv`
- `ddpm_refiner.py` implements refinement and inpainting without geometry inputs or geometry loss.
- `eval_geometry_metrics.py` computes off-map and wall-crossing metrics after reconstruction.
- `run_geometry_extension.py` repeatedly calls the outputs geometry feasibility extensions and synthetic feasibility stress tests.
