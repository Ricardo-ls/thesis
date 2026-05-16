# E3-DPS Phase A Extension

This extension only evaluates zeta=0 no-guidance re-diffusion and zeta=3e-3 on the existing Phase A calibration set. It does not generate new data, train, change likelihood, change t_start, or enter Phase B.

## Key Questions

1. zeta=0 ADE: `0.147443730`.
2. zeta=0 impact vs E3 fused: delta `0.010876317`, improvement `-7.964064%`.
3. zeta=1e-3 ADE from Phase A: `0.137241931`.
4. zeta=3e-3 ADE: `0.134885949`.
5. zeta=3e-3 better than zeta=1e-3: `True`.
6. zeta=3e-3 beats E3 fused baseline: `True`.
7. zeta=3e-3 likelihood decreased: `False`.
8. zeta=3e-3 high-conf ratio vs E3: `1.443696614`.
9. zeta=3e-3 acceleration ratio vs E3: `0.950675161`.
10. Recommend Phase B: `False`.

## Extension Zeta Summary

| zeta | mean_ADE_E3_fused | mean_ADE_DPS | ADE_delta | ADE_improvement_percent | high_conf_ratio_vs_E3 | acceleration_ratio_vs_E3 | likelihood_before | likelihood_after | likelihood_decreased | max_update_to_x_ratio | finite_ok | nan_inf_cases | audit_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000000 | 0.136567413 | 0.147443730 | 0.010876317 | -7.964064284 | 2.603895597 | 0.950323132 | 0.350189090 | 4.320113063 | False | 0.000000000 | True | 0 | FAIL |
| 0.003000000 | 0.136567413 | 0.134885949 | -0.001681464 | 1.231233538 | 1.443696614 | 0.950675161 | 0.350189090 | 1.757373989 | False | 0.074670888 | True | 0 | FAIL |

## Decision

No Phase B is recommended. Keep E3 confidence-aware fusion as the final stable method.

## Output Files

- `outputs/stage4/e3_dps_calibrated_phaseA_extension/e3_dps_phaseA_extension_metrics.csv`
- `outputs/stage4/e3_dps_calibrated_phaseA_extension/e3_dps_phaseA_extension_zeta_summary.csv`
