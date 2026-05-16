# E3-DPS Extended Sign Check

This is a single-step numerical diagnostic only. It does not run full DPS, calibrated audit, training, new hold-out generation, or method tuning.

## Summary Answers

- Sign correct across all 6 conditions and t in {1,3,5}: `True`
- Any sign-reversed case: `False`
- Any interface-suspect case: `False`
- Global largest stable eps: `0.001`
- Most conservative stable eps: `0.001`
- eps=1e-4 stable everywhere: `True`
- eps=1e-3 stable everywhere: `True`
- eps=1e-2 overshoots broadly: `True`
- Stable eps strongly condition-dependent: `False`
- Stable eps strongly step-dependent: `False`
- Decision case: `Case 1`
- Recommended Phase A zeta range: `{1e-5, 3e-5, 1e-4, 3e-4, 1e-3}`
- Recommended scaling mode: `global constant zeta`

## Per-Case Stable Range

| condition | t | classification | largest_stable_eps | largest_stable_update_to_x_ratio | best_eps_by_likelihood_drop | eps_at_overshoot_if_any |
| --- | --- | --- | --- | --- | --- | --- |
| bias_medium | 1 | SIGN_CORRECT_STABLE | 0.001 | 0.0040303 | 0.001 | 0.01 |
| bias_medium | 3 | SIGN_CORRECT_STABLE | 0.001 | 0.00389646 | 0.001 | 0.01 |
| bias_medium | 5 | SIGN_CORRECT_STABLE | 0.001 | 0.00383096 | 0.001 | 0.01 |
| burst_medium | 1 | SIGN_CORRECT_STABLE | 0.001 | 0.00378365 | 0.001 | 0.01 |
| burst_medium | 3 | SIGN_CORRECT_STABLE | 0.001 | 0.00366498 | 0.001 | 0.01 |
| burst_medium | 5 | SIGN_CORRECT_STABLE | 0.001 | 0.00355043 | 0.001 | 0.01 |
| combined_medium | 1 | SIGN_CORRECT_STABLE | 0.001 | 0.00357683 | 0.001 | 0.01 |
| combined_medium | 3 | SIGN_CORRECT_STABLE | 0.001 | 0.00347508 | 0.001 | 0.01 |
| combined_medium | 5 | SIGN_CORRECT_STABLE | 0.001 | 0.00339075 | 0.001 | 0.01 |
| drift_medium | 1 | SIGN_CORRECT_STABLE | 0.001 | 0.00374968 | 0.001 | 0.01 |
| drift_medium | 3 | SIGN_CORRECT_STABLE | 0.001 | 0.00365783 | 0.001 | 0.01 |
| drift_medium | 5 | SIGN_CORRECT_STABLE | 0.001 | 0.00359087 | 0.001 | 0.01 |
| gaussian_medium | 1 | SIGN_CORRECT_STABLE | 0.001 | 0.00371213 | 0.001 | 0.01 |
| gaussian_medium | 3 | SIGN_CORRECT_STABLE | 0.001 | 0.00363132 | 0.001 | 0.01 |
| gaussian_medium | 5 | SIGN_CORRECT_STABLE | 0.001 | 0.00354892 | 0.001 | 0.01 |
| jump_medium | 1 | SIGN_CORRECT_STABLE | 0.001 | 0.0033226 | 0.001 | 0.01 |
| jump_medium | 3 | SIGN_CORRECT_STABLE | 0.001 | 0.00323944 | 0.001 | 0.01 |
| jump_medium | 5 | SIGN_CORRECT_STABLE | 0.001 | 0.00307137 | 0.001 | 0.01 |

## Condition Dependence

| condition | min | max |
| --- | --- | --- |
| bias_medium | 0.001 | 0.001 |
| burst_medium | 0.001 | 0.001 |
| combined_medium | 0.001 | 0.001 |
| drift_medium | 0.001 | 0.001 |
| gaussian_medium | 0.001 | 0.001 |
| jump_medium | 0.001 | 0.001 |

## Step Dependence

| t | min | max |
| --- | --- | --- |
| 1 | 0.001 | 0.001 |
| 3 | 0.001 | 0.001 |
| 5 | 0.001 | 0.001 |

## Output Files

- `outputs/stage4/e3_dps_extended_sign_check/e3_dps_extended_sign_check.csv`
- `outputs/stage4/e3_dps_extended_sign_check/e3_dps_extended_sign_check_case_summary.csv`
