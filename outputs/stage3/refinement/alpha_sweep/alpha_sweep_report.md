# Stage 3 DDPM Masked Blend v2 Alpha Sweep

## Purpose

This sweep tests whether the fixed alpha=0.25 used by `ddpm_prior_masked_blend_v2` is appropriate.
Observed points are preserved, while missing points are blended with the cached `ddpm_prior_interface_v0` candidate.
The main summary figure now reports mean `masked_ADE` by alpha across the four degradation settings for each coarse method.
`Lower is better` for every point in that summary figure.

## Best Alpha By Coarse Method

| coarse_method | alpha | masked_ADE |
| --- | --- | --- |
| kalman_cv_dt1.0_q1e-3_r1e-2 | 0.10 | 0.049230 |
| linear_interp | 0.00 | 0.027261 |
| savgol_w5_p2 | 0.00 | 0.027930 |

## Best Alpha By Degradation And Coarse Method

| degradation | coarse_method | alpha | masked_ADE |
| --- | --- | --- | --- |
| missing_drift | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.05 | 0.043867 |
| missing_drift | linear_interp | 0.00 | 0.028530 |
| missing_drift | savgol_w5_p2 | 0.00 | 0.028412 |
| missing_noise | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.10 | 0.057296 |
| missing_noise | linear_interp | 0.00 | 0.031154 |
| missing_noise | savgol_w5_p2 | 0.00 | 0.032901 |
| missing_noise_drift | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.10 | 0.063352 |
| missing_noise_drift | linear_interp | 0.00 | 0.040778 |
| missing_noise_drift | savgol_w5_p2 | 0.00 | 0.042147 |
| missing_only | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.05 | 0.031466 |
| missing_only | linear_interp | 0.00 | 0.008584 |
| missing_only | savgol_w5_p2 | 0.00 | 0.008262 |

## Main Reading

- Linear: best alpha = `0.00`.
- Savitzky-Golay: best alpha = `0.00`.
- Kalman: best alpha is near `0.10`.
- Larger alpha values generally hurt missing-segment reconstruction quality.

## Output Files

- `alpha_sweep_metrics.csv`
- `alpha_sweep_metrics.json`
- `alpha_sweep_summary.csv`
- `alpha_sweep_report.md`
- `figures/alpha_sweep_ADE.png`
- `figures/alpha_sweep_masked_ADE.png`
- `figures/alpha_sweep_improvement_masked_ADE.png`
