# E3-Initialized DPS Aligned Audit

This is an aligned audit, not formal E4 and not hyperparameter selection. It reads frozen E3 `fused_t1_tau07_gamma2` arrays, initializes from them, and runs short t=5 norm-based DPS polish for all zeta values.

## Setup Answers

1. Initialized from E3 fused_t1_tau07_gamma2: `yes`.
2. Pure Gaussian initialization avoided: `yes`.
3. 0 NaN / Inf: `True`.

## Zeta Summary

| zeta | six_condition_mean_ADE | six_condition_mean_E3_fused_ADE | ADE_improvement_percent_vs_E3 | high_conf_ratio_vs_E3 | acceleration_ratio_vs_E3 | likelihood_decreased | nan_inf_cases | audit_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.300000 | 0.959216 | 0.136494 | -602.753585 | 43.606144 | 1.029128 | False | 0 | FAIL |
| 1.000000 | 3.023827 | 0.136494 | -2115.354939 | 139.681749 | 1.312010 | False | 0 | FAIL |
| 3.000000 | 9.430406 | 0.136494 | -6809.025821 | 438.081423 | 2.106067 | False | 0 | FAIL |

## Decision Questions

- zeta=0.3: mean ADE `0.959216`, improvement vs E3 `-602.753585%`, high-conf ratio `43.606144`, acceleration ratio `1.029128`, likelihood decreased `False`, decision `FAIL`.
- zeta=1.0: mean ADE `3.023827`, improvement vs E3 `-2115.354939%`, high-conf ratio `139.681749`, acceleration ratio `1.312010`, likelihood decreased `False`, decision `FAIL`.
- zeta=3.0: mean ADE `9.430406`, improvement vs E3 `-6809.025821%`, high-conf ratio `438.081423`, acceleration ratio `2.106067`, likelihood decreased `False`, decision `FAIL`.

Overall audit decision: `FAIL`.
Recommendation: Stop DPS for now and keep E3 confidence-aware fusion as the final stable method.

## Output Files

- `outputs/stage4/e3_initialized_dps_aligned_audit/e3_initialized_dps_aligned_metrics.csv`
- `outputs/stage4/e3_initialized_dps_aligned_audit/e3_initialized_dps_aligned_condition_summary.csv`
- `outputs/stage4/e3_initialized_dps_aligned_audit/e3_initialized_dps_aligned_seed_zeta_summary.csv`
- `outputs/stage4/e3_initialized_dps_aligned_audit/e3_initialized_dps_aligned_step_trace.csv`
- `outputs/stage4/e3_initialized_dps_aligned_audit/figures`
