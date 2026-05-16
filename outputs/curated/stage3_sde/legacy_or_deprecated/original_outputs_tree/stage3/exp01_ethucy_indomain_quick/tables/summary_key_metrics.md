# exp01_ethucy_indomain_quick Summary

- missing condition: `missing_only`
- methods evaluated: `linear_interp`, `savgol_w5_p2`, `kalman_cv_dt1.0_q1e-3_r1e-2`, `ddpm_v3_inpainting`, `ddpm_v3_inpainting_anchored`
- DDPM seeds per trajectory: `5`
- max trajectories used in this quick run: `1024`

| method | metric | N | mean | std | median | p05 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear_interp | masked_ADE | 1024 | 0.084269 | 0.085497 | 0.063745 | 0.000086 | 0.244934 |
| linear_interp | masked_RMSE | 1024 | 0.063204 | 0.063442 | 0.048571 | 0.000066 | 0.183702 |
| linear_interp | endpoint_error | 1024 | 0.069458 | 0.078720 | 0.048801 | 0.000064 | 0.219464 |
| savgol_w5_p2 | masked_ADE | 1024 | 0.081871 | 0.083248 | 0.061651 | 0.000101 | 0.237618 |
| savgol_w5_p2 | masked_RMSE | 1024 | 0.061897 | 0.062250 | 0.046990 | 0.000077 | 0.179523 |
| savgol_w5_p2 | endpoint_error | 1024 | 0.064755 | 0.074673 | 0.044343 | 0.000089 | 0.206792 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | masked_ADE | 1024 | 0.277678 | 0.253865 | 0.224637 | 0.000139 | 0.706803 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | masked_RMSE | 1024 | 0.216318 | 0.197787 | 0.173482 | 0.000107 | 0.556008 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | endpoint_error | 1024 | 0.442780 | 0.417500 | 0.350486 | 0.000198 | 1.171946 |
| ddpm_v3_inpainting | masked_ADE | 5120 | 0.304263 | 0.230202 | 0.251038 | 0.038645 | 0.755161 |
| ddpm_v3_inpainting | masked_RMSE | 5120 | 0.239656 | 0.179509 | 0.198310 | 0.030521 | 0.589194 |
| ddpm_v3_inpainting | endpoint_error | 5120 | 0.488641 | 0.373420 | 0.408063 | 0.057209 | 1.210126 |
| ddpm_v3_inpainting_anchored | masked_ADE | 5120 | 0.089707 | 0.068702 | 0.073027 | 0.012279 | 0.226266 |
| ddpm_v3_inpainting_anchored | masked_RMSE | 5120 | 0.068152 | 0.051644 | 0.055786 | 0.009366 | 0.171439 |
| ddpm_v3_inpainting_anchored | endpoint_error | 5120 | 0.075014 | 0.070176 | 0.055867 | 0.007020 | 0.211658 |
