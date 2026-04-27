# Stage 3 Phase 1 Representative Visualization

## Purpose

This figure provides one representative qualitative reconstruction example for
the current Stage 3 Phase 1 canonical_room3 benchmark.

It is intended for report and presentation use. The figure shows:

- the full ground-truth trajectory
- the degraded input with the missing span
- the reconstructed result from Linear
- the reconstructed result from SG
- the reconstructed result from Kalman

This qualitative figure should be read together with the two metric views:

- full-trajectory metrics (`ADE`, `FDE`, `RMSE`)
- missing-segment metrics (`masked_ADE`, `masked_RMSE`)

Since the task is missing-segment reconstruction, the masked view should be
emphasized when discussing the reconstructed gap itself.

## Input Files

The default visualization uses:

- `outputs/stage3/phase1/canonical_room3/data/clean_windows_room3.npz`
- `outputs/stage3/phase1/canonical_room3/data/experiments/span20_fixed_seed42/missing_span_windows.npz`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/linear_interp/results.npz`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/savgol_w5_p2/results.npz`
- `outputs/stage3/phase1/canonical_room3/baselines/span20_fixed_seed42/kalman_cv_dt1.0_q1e-3_r1e-2/results.npz`

## Sample Selection

The default sample is:

- `experiment_id = span20_fixed_seed42`
- `sample_index = 0`

The plotting script supports changing both values from the command line.

## Command

Run from the `SingularTrajectory/` repository root:

```bash
python tools/stage3/eval/plot_representative_reconstruction.py
```

Example with explicit arguments:

```bash
python tools/stage3/eval/plot_representative_reconstruction.py \
  --experiment_id span20_fixed_seed42 \
  --sample_index 0
```

## Output

The output directory is:

- `outputs/stage3/phase1/canonical_room3/figures/`

Generated files:

- `outputs/stage3/phase1/canonical_room3/figures/exp0_representative_reconstruction.png`
- `outputs/stage3/phase1/canonical_room3/figures/exp0_representative_reconstruction.pdf`
