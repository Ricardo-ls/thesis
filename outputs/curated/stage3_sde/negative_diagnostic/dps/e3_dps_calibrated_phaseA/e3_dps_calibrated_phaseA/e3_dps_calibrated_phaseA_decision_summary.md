# Calibrated E3-Initialized DPS Phase A Audit

This is a calibration audit, not training, not formal E4, and not final method selection.

## Setup Answers

1. Calibration hold-out generated: `yes`, seed range `14000-14499`.
2. Six degradations generated: `yes`.
3. E3 fused baseline reproduced: `yes`.
4. Zeta values evaluated: `[1e-05, 3e-05, 0.0001, 0.0003, 0.001]`.

## Zeta Summary

| zeta | mean_ADE_E3_fused | mean_ADE_DPS | ADE_improvement_percent | high_conf_ratio_vs_E3 | acceleration_ratio_vs_E3 | likelihood_decreased | max_update_to_x_ratio | finite_ok | audit_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000010000 | 0.136567413 | 0.147270037 | -7.836879550 | 2.589086483 | 0.950323332 | False | 0.000278546 | True | FAIL |
| 0.000030000 | 0.136567413 | 0.146927559 | -7.586103507 | 2.559829699 | 0.950323816 | False | 0.000832557 | True | FAIL |
| 0.000100000 | 0.136567413 | 0.145780052 | -6.745854643 | 2.461223075 | 0.950325718 | False | 0.002754287 | True | FAIL |
| 0.000300000 | 0.136567413 | 0.142932289 | -4.660610928 | 2.211664840 | 0.950333316 | False | 0.008407416 | True | FAIL |
| 0.001000000 | 0.136567413 | 0.137241931 | -0.493908327 | 1.686777728 | 0.950384055 | False | 0.027279060 | True | FAIL |

## Decision

- Overall Phase A decision: `FAIL`
- Locked zeta for Phase B, if any: `nan`
- Recommendation: Stop DPS and keep E3 fusion as final stable method.

## Output Files

- `outputs/stage4/e3_dps_calibrated_phaseA/e3_dps_calibrated_phaseA_metrics.csv`
- `outputs/stage4/e3_dps_calibrated_phaseA/e3_dps_calibrated_phaseA_condition_summary.csv`
- `outputs/stage4/e3_dps_calibrated_phaseA/e3_dps_calibrated_phaseA_zeta_summary.csv`
- `outputs/stage4/e3_dps_calibrated_phaseA/e3_dps_calibrated_phaseA_step_diagnostics.csv`
- `outputs/stage4/e3_dps_calibrated_phaseA/figures`
