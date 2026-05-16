# Stage 3 SDEdit t_start 1/2/3 + Confidence Hard-Fusion Diagnosis

This is a mechanism diagnosis, not a formal E3 method. The indoor-v2 prior checkpoint and old vanilla SDEdit reverse process were left unchanged.

## Setup

- checkpoint: `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`
- normalization: `data/stage3_indoor/rel_norm_params_v2.npz`
- SDEdit seeds: `[42, 43, 44, 45, 46]`
- t_start sweep: `[1, 2, 3]`
- confidence source: validated reconstructed Formal E1 cache under `outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache`
- 6 condition x 3 t_start cache generated successfully: `True`
- gaussian t2 regenerated-vs-legacy max abs delta: `0.0`

## Best t_start By Condition

      condition  best_t_start              curve_type  noisy_ADE  best_sdedit_ADE  delta_ADE  relative_improvement_percent  high_conf_delta  low_conf_delta  destructive_all_tstarts
gaussian_medium             2          small_U-shaped   0.062567         0.060812  -0.001755                      2.804624         0.006115       -0.004159                    False
   drift_medium             1         monotonic_worse   0.017648         0.018343   0.000695                     -3.940363         0.001684        0.000160                     True
   burst_medium             1         monotonic_worse   0.074755         0.078045   0.003290                     -4.400355         0.010933       -0.002402                     True
    bias_medium             1         monotonic_worse   0.188996         0.189057   0.000061                     -0.032515        -0.000101        0.000140                     True
    jump_medium             3        monotonic_better   0.263368         0.254214  -0.009154                      3.475883         0.008864       -0.017291                    False
combined_medium             1 tiny-t_only_improvement   0.199064         0.198917  -0.000147                      0.073666         0.001898       -0.001205                    False

## Mechanism Answers

1. Cache complete: `True`.
2. Best t_start values are listed above; the selection rule was min overall ADE, tie-broken by high-conf ADE.
3. Curve types are condition-specific, not universally gaussian-like.
4. Vanilla high-conf harm count: `5/6` conditions.
5. Vanilla low-conf improvement count: `4/6` conditions.
6. Best fusion setting: `tau07_gamma2`.
7. Best fusion improves over noisy in `5/6` conditions.
8. High-conf no-harm count under best fusion: `5/6` conditions.
9. Low-conf preservation count under best fusion: `4/6` eligible conditions.
10. Mean motion usage ratio under best fusion: `0.438461`.
11. Drift best-fusion ADE delta vs noisy: `-0.000040`; high no-harm ratio `1.000000`.
12. Burst best-fusion ADE delta vs noisy: `-0.001163`; high no-harm ratio `1.000000`.
13. Bias remains a structural limitation because the relative-displacement SDEdit reconstruction is anchored at degraded y[0].
14. Final case classification: `Case 1: confidence-aware fusion succeeds`.

## Interpretation

Confidence-aware hard-threshold fusion supports the hypothesis that vanilla SDEdit's main weakness is missing confidence control. A formal confidence-aware SDEdit / E3 pre-registration is justified.

## Output Files

- `stage3_sdedit_vanilla_condition_metrics.csv`
- `stage3_sdedit_vanilla_confidence_bin_metrics.csv`
- `stage3_sdedit_tstart_sweep_summary.csv`
- `stage3_sdedit_best_tstart_by_condition.csv`
- `stage3_sdedit_hardfusion_condition_metrics.csv`
- `stage3_sdedit_hardfusion_confidence_bin_metrics.csv`
- `stage3_sdedit_hardfusion_passfail_diagnostic.csv`
- `arrays/`
- `figures/`
