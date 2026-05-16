# exp01_ethucy_indomain_quick Summary

- missing condition: `missing_only`
- methods evaluated: `linear_interp`, `savgol_w5_p2`, `kalman_cv_dt1.0_q1e-3_r1e-2`, `ddpm_v3_inpainting`, `ddpm_v3_inpainting_anchored`
- DDPM seeds per trajectory: `5`
- max trajectories used in this quick run: `128`

| method | metric | N | mean | std | median | p05 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear_interp | masked_ADE | 128 | 0.117882 | 0.096267 | 0.090939 | 0.000000 | 0.292653 |
| linear_interp | masked_RMSE | 128 | 0.088840 | 0.071852 | 0.065301 | 0.000000 | 0.217141 |
| linear_interp | endpoint_error | 128 | 0.098784 | 0.097943 | 0.070128 | 0.000000 | 0.314179 |
| savgol_w5_p2 | masked_ADE | 128 | 0.116141 | 0.094873 | 0.089454 | 0.000000 | 0.290804 |
| savgol_w5_p2 | masked_RMSE | 128 | 0.088048 | 0.071053 | 0.064654 | 0.000000 | 0.217833 |
| savgol_w5_p2 | endpoint_error | 128 | 0.096247 | 0.096406 | 0.068057 | 0.000000 | 0.307707 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | masked_ADE | 128 | 0.320332 | 0.257278 | 0.259567 | 0.000000 | 0.771476 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | masked_RMSE | 128 | 0.248345 | 0.197583 | 0.201005 | 0.000000 | 0.592234 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | endpoint_error | 128 | 0.496470 | 0.413126 | 0.384893 | 0.000000 | 1.246613 |
| ddpm_v3_inpainting | masked_ADE | 640 | 0.307332 | 0.230151 | 0.254322 | 0.032709 | 0.739384 |
| ddpm_v3_inpainting | masked_RMSE | 640 | 0.239805 | 0.176179 | 0.206190 | 0.025164 | 0.564306 |
| ddpm_v3_inpainting | endpoint_error | 640 | 0.469053 | 0.355806 | 0.394472 | 0.043709 | 1.138165 |
| ddpm_v3_inpainting_anchored | masked_ADE | 640 | 0.126066 | 0.085908 | 0.115917 | 0.013374 | 0.283362 |
| ddpm_v3_inpainting_anchored | masked_RMSE | 640 | 0.095714 | 0.064546 | 0.090295 | 0.010123 | 0.209067 |
| ddpm_v3_inpainting_anchored | endpoint_error | 640 | 0.105415 | 0.093445 | 0.081451 | 0.007848 | 0.303870 |
