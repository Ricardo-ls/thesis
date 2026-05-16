# E3 DPS Single-Step Sign Check

This diagnostic uses one case only: `drift_medium`, trajectory `0`, initialized from frozen E3 `fused_t1_tau07_gamma2`, q_sampled to `t=5` with seed `42`.

- L_zero: `2.296465874`
- grad_norm: `25.091220856`
- x_norm: `6.987510681`
- zeta=0.3 equivalent update_to_x_ratio at this x_t: `1.077260072`

## Direction Trend

| eps | L_minus | delta_L_minus | L_plus | delta_L_plus | update_to_x_ratio |
| --- | --- | --- | --- | --- | --- |
| 1e-07 | 2.296399355 | -0.000066519 | 2.296527863 | 0.000061989 | 0.000000359 |
| 1e-06 | 2.295833111 | -0.000632763 | 2.297092199 | 0.000626326 | 0.000003591 |
| 1e-05 | 2.290171862 | -0.006294012 | 2.302757502 | 0.006291628 | 0.000035909 |
| 1e-04 | 2.233644962 | -0.062820911 | 2.359551430 | 0.063085556 | 0.000359087 |
| 1e-03 | 1.685276628 | -0.611189246 | 2.936358690 | 0.639892817 | 0.003590867 |
| 1e-02 | 4.711350441 | 2.414884567 | 8.942473412 | 6.646007538 | 0.035908669 |

## Diagnostic Answers

- Which direction lowers likelihood: `minus` eps values `[1e-07, 1e-06, 1e-05, 0.0001, 0.001]`, `plus` eps values `[]`.
- At minimum eps=1e-07: minus delta `-0.000066519`, plus delta `0.000061989`.
- Diagnostic case: `Case 1: sign correct, step too large`.
- Note: Minus-gradient direction lowers likelihood at small eps, but large eps overshoots.
- Stable minus eps range max: `0.001`.
- Is zeta=0.3 clearly too large for this single-step scale: `True`.

## Next Step Recommendation

B. Do not treat the previous DPS failure as final method failure. The sign is locally correct, but zeta=0.3 is many orders larger than the stable single-step eps range; any future audit would need a pre-registered log-scale zeta sweep on a new hold-out.

## Output

- `outputs/stage4/e3_dps_single_step_sign_check/e3_dps_single_step_sign_check.csv`
