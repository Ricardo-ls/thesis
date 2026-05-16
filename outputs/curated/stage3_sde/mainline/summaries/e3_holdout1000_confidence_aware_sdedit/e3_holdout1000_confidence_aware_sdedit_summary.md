# E3 Holdout-1000 Confidence-Aware SDEdit Expanded Validation

This validation uses a new independent clean hold-out set with seeds 13000-13999. It does not use old `clean_trajs[200:]`, old degraded arrays, retraining, checkpoint modification, or a changed SDEdit reverse process.

## Data

- clean hold-out: `data/stage4/e3_holdout_1000/clean_trajs.npy`
- clean metadata: `data/stage4/e3_holdout_1000/metadata.json`
- degradation metadata: `data/stage4/e3_holdout_1000/degradation_metadata.json`
- output arrays: `outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays`
- shape: `(1000, 20, 2)`
- seed range: `13000-13999`

## Method

- checkpoint: `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`
- normalization: `data/stage3_indoor/rel_norm_params_v2.npz`
- SDEdit seeds averaged: `[42, 43, 44, 45, 46]`
- t_start sweep: `[1, 2, 3]`
- global fixed t_start: `1`
- fusion: `tau_high=0.7`, `gamma=2`
- confidence: `c_t = exp(- ||y_t - x*_t|| / delta_0)`, with per-condition `delta_0` equal to median degraded-frame error on this hold-out set.

## Fixed Previous Best t_start Diagnostic Upper Bound

      condition  previous_best_t_start   curve_type_on_holdout  noisy_ADE  selected_sdedit_ADE  selected_delta_ADE
gaussian_medium                      2          small_U-shaped   0.062373             0.060458           -0.001915
   drift_medium                      1          small_U-shaped   0.035624             0.035278           -0.000346
   burst_medium                      1         monotonic_worse   0.072314             0.076426            0.004112
    bias_medium                      1 tiny-t_only_improvement   0.187123             0.187056           -0.000067
    jump_medium                      3        monotonic_better   0.260431             0.249893           -0.010538
combined_medium                      1        monotonic_better   0.206374             0.205783           -0.000591

## Trend Stability Checks

          protocol  fused_better_than_noisy_conditions  fused_better_than_noisy_ge4of6  high_conf_no_harm_conditions  high_conf_no_harm_recovered  low_conf_preserved_conditions  low_conf_eligible_conditions  low_conf_preservation_pass  mean_motion_usage_ratio  not_noisy_reversion
         global_t1                                   6                            True                             5                         True                              6                             6                        True                 0.434335                 True
per_condition_best                                   6                            True                             5                         True                              6                             6                        True                 0.435138                 True

## Answers

1. New hold-out clean trajectories generated: `True`, shape `(1000, 20, 2)`, seeds `13000-13999`.
2. Six degradations generated: `True`.
3. SDEdit t_start {1,2,3} completed for all six conditions: `True`.
4. Global t=1 fused >=4/6 overall improvement: `True` (6/6).
5. Global t=1 high-conf no-harm recovered: `True` (5/6).
6. Global t=1 low-conf preservation: `True` (6/6 eligible).
7. Per-condition best fused >=4/6 overall improvement: `True` (6/6).
8. Per-condition best high-conf no-harm recovered: `True` (5/6).
9. Mean motion usage ratio, global t=1: `0.434335`; per-condition best: `0.435138`.
10. Drift trend stable under global t=1: `True`.
11. Burst trend stable under global t=1: `True`.
12. Bias remains structurally limited by relative-displacement generation anchored at degraded y[0].

## Recommendation

The expanded hold-out result supports moving to formal E3 pre-registration / final report framing for confidence-aware SDEdit.

## Output Files

- `e3_holdout1000_vanilla_tstart_metrics.csv`
- `e3_holdout1000_global_t1_metrics.csv`
- `e3_holdout1000_per_condition_best_metrics.csv`
- `e3_holdout1000_confidence_bin_metrics.csv`
- `e3_holdout1000_paired_statistics.csv`
- `e3_holdout1000_trend_stability_summary.csv`
- `arrays/`
- `figures/`
