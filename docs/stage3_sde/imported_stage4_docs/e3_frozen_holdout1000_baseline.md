# E3 Frozen Holdout-1000 Confidence-Aware SDEdit Baseline

Date frozen: 2026-05-15

This document freezes the current E3 holdout-1000 confidence-aware SDEdit baseline as the reference validation result. The seed 13000-13999 hold-out set is now an analysis / validation set and must not be used for further tuning.

## Data Source

- Clean hold-out path: `data/stage4/e3_holdout_1000/clean_trajs.npy`
- Clean shape: `(1000, 20, 2)`
- Seed range: `13000-13999`
- N trajectories: `1000`
- Degradation metadata: `data/stage4/e3_holdout_1000/degradation_metadata.json`
- Evaluation output directory: `outputs/stage4/e3_holdout1000_confidence_aware_sdedit/`

The hold-out seed range is non-overlapping with the existing Stage 3 train / validation ranges:

- `train_trajs.npy`: seeds `1000-10999`
- `val_trajs.npy`: seeds `11000-12999`
- E3 holdout-1000: seeds `13000-13999`

## Frozen Method

- Method family: observation-initialized SDEdit with confidence-aware hard-threshold fusion.
- Prior: indoor-v2 unconditional DDPM prior.
- Prior checkpoint: `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`
- Normalization: `data/stage3_indoor/rel_norm_params_v2.npz`
- SDEdit interface:
  `degraded absolute y -> relative displacement -> normalize -> q_sample(y_rel, t_start) -> unconditional reverse denoise -> denormalize -> reconstruct absolute trajectory using degraded y[0] anchor`
- Main fusion setting: `tau_high = 0.7`, `gamma = 2`
- Main protocol: global fixed `t_start = 1`
- Diagnostic upper-bound protocol: per-condition previously selected best t_start
  - gaussian_medium: `2`
  - drift_medium: `1`
  - burst_medium: `1`
  - bias_medium: `1`
  - jump_medium: `3`
  - combined_medium: `1`

## Frozen Results

Global fixed `t_start = 1`:

- fused better than noisy: `6/6` conditions
- high-confidence no-harm recovered: `5/6` conditions
- low-confidence preservation: `6/6` eligible conditions
- mean motion_usage_ratio: `0.434335`

Per-condition best diagnostic upper-bound:

- fused better than noisy: `6/6` conditions
- high-confidence no-harm recovered: `5/6` conditions
- low-confidence preservation: `6/6` eligible conditions
- mean motion_usage_ratio: `0.435138`

## Frozen Rule

Seed `13000-13999` is now treated as an analysis / validation set.

Do not further tune `tau_high`, `gamma`, or `t_start` on this set.

Any new adaptive rule must be confirmed on a new independent hold-out set, for example seed `14000-14999`.

Oracle diagnostics computed from this set are analysis-only. They must not be reported as method performance.
