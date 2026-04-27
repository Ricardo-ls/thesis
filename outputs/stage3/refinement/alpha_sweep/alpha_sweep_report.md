# Stage 3 DDPM Masked Blend v2 Alpha Sweep

## Purpose

This sweep tests whether the fixed alpha=0.25 used by `ddpm_prior_masked_blend_v2` is appropriate.
Observed points are preserved, while missing points are blended with the cached `ddpm_prior_interface_v0` candidate.

## Questions

1. Which alpha gives the best masked_ADE for each coarse method?
2. Which alpha gives the best masked_ADE for each degradation/coarse_method pair?
3. Does any alpha improve over identity/coarse?
4. Does any alpha improve over light_savgol_refiner?
5. Does larger alpha consistently degrade performance?
6. What does this imply about the unconditional DDPM prior?

## Best Alpha By Coarse Method

| coarse_method | alpha | masked_ADE |
| --- | --- | --- |
| kalman_cv_dt1.0_q1e-3_r1e-2 | 0.100000 | 0.049230 |
| linear_interp | 0.000000 | 0.027261 |
| savgol_w5_p2 | 0.000000 | 0.027930 |

## Best Alpha By Degradation And Coarse Method

| degradation | coarse_method | alpha | masked_ADE | improvement_masked_ADE |
| --- | --- | --- | --- | --- |
| missing_drift | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.050000 | 0.043867 | 0.009616 |
| missing_drift | linear_interp | 0.000000 | 0.028530 | 0.000000 |
| missing_drift | savgol_w5_p2 | 0.000000 | 0.028412 | 0.000000 |
| missing_noise | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.100000 | 0.057296 | 0.028361 |
| missing_noise | linear_interp | 0.000000 | 0.031154 | 0.000000 |
| missing_noise | savgol_w5_p2 | 0.000000 | 0.032901 | 0.000000 |
| missing_noise_drift | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.100000 | 0.063352 | 0.023611 |
| missing_noise_drift | linear_interp | 0.000000 | 0.040778 | 0.000000 |
| missing_noise_drift | savgol_w5_p2 | 0.000000 | 0.042147 | 0.000000 |
| missing_only | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.050000 | 0.031466 | 0.000638 |
| missing_only | linear_interp | 0.000000 | 0.008584 | 0.000000 |
| missing_only | savgol_w5_p2 | 0.000000 | 0.008262 | 0.000000 |

## Answers

### 1. Best alpha by coarse method

- Kalman: best alpha = `0.10` with mean masked_ADE = `0.049230`
- Linear: best alpha = `0.00` with mean masked_ADE = `0.027261`
- SG: best alpha = `0.00` with mean masked_ADE = `0.027930`

### 2. Best alpha by degradation/coarse_method pair

- missing_drift / Kalman: best alpha = `0.05` with masked_ADE = `0.043867`
- missing_drift / Linear: best alpha = `0.00` with masked_ADE = `0.028530`
- missing_drift / SG: best alpha = `0.00` with masked_ADE = `0.028412`
- missing_noise / Kalman: best alpha = `0.10` with masked_ADE = `0.057296`
- missing_noise / Linear: best alpha = `0.00` with masked_ADE = `0.031154`
- missing_noise / SG: best alpha = `0.00` with masked_ADE = `0.032901`
- missing_noise_drift / Kalman: best alpha = `0.10` with masked_ADE = `0.063352`
- missing_noise_drift / Linear: best alpha = `0.00` with masked_ADE = `0.040778`
- missing_noise_drift / SG: best alpha = `0.00` with masked_ADE = `0.042147`
- missing_only / Kalman: best alpha = `0.05` with masked_ADE = `0.031466`
- missing_only / Linear: best alpha = `0.00` with masked_ADE = `0.008584`
- missing_only / SG: best alpha = `0.00` with masked_ADE = `0.008262`

### 3. Does any alpha improve over identity/coarse?

- Yes

### 4. Does any alpha improve over light_savgol_refiner?

- Yes

### 5. Does larger alpha consistently degrade performance?

- Linear: larger alpha consistently degrades masked_ADE = `True`
- SG: larger alpha consistently degrades masked_ADE = `True`
- Kalman: larger alpha consistently degrades masked_ADE = `False`

### 6. Implication for the unconditional DDPM prior

The unconditional DDPM prior is not acting like a reliable direct missing-segment refiner here. If the best alpha stays near 0 and larger alpha worsens masked_ADE, the prior candidate is adding harmful trajectory content unless it is conditioned more tightly on the observed context and missing-mask structure.

## Output Files

- `alpha_sweep_metrics.csv`
- `alpha_sweep_metrics.json`
- `alpha_sweep_summary.csv`
- `alpha_sweep_report.md`
- `figures/alpha_sweep_ADE.png`
- `figures/alpha_sweep_masked_ADE.png`
- `figures/alpha_sweep_improvement_masked_ADE.png`
