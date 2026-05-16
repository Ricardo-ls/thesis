# exp01_ethucy_indomain_quick Full Stats Matrix

Population note:
- deterministic methods: `N=1024` trajectories
- DDPM methods: `N=1024 × 5 = 5120` trajectory-seed cases

| method | missing_condition | metric | N | mean | std | median | min | max | p05 | p25 | p75 | p95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_interp | missing_only | masked_ADE | 1024 | 0.084269 | 0.085497 | 0.063745 | 0.000000 | 0.764854 | 0.000086 | 0.027149 | 0.110830 | 0.244934 |
| linear_interp | missing_only | masked_RMSE | 1024 | 0.063204 | 0.063442 | 0.048571 | 0.000000 | 0.549852 | 0.000066 | 0.020577 | 0.084440 | 0.183702 |
| linear_interp | missing_only | endpoint_error | 1024 | 0.069458 | 0.078720 | 0.048801 | 0.000000 | 0.636254 | 0.000064 | 0.017085 | 0.090332 | 0.219464 |
| linear_interp | missing_only | path_length_error | 1024 | 0.034984 | 0.093710 | 0.006441 | 0.000000 | 0.960357 | 0.000000 | 0.000558 | 0.023558 | 0.185530 |
| linear_interp | missing_only | acceleration_error | 1024 | 0.031694 | 0.033172 | 0.022152 | 0.000000 | 0.200640 | 0.000069 | 0.008698 | 0.041709 | 0.106615 |
| savgol_w5_p2 | missing_only | masked_ADE | 1024 | 0.081871 | 0.083248 | 0.061651 | 0.000000 | 0.742674 | 0.000101 | 0.026192 | 0.107683 | 0.237618 |
| savgol_w5_p2 | missing_only | masked_RMSE | 1024 | 0.061897 | 0.062250 | 0.046990 | 0.000000 | 0.536781 | 0.000077 | 0.019740 | 0.081946 | 0.179523 |
| savgol_w5_p2 | missing_only | endpoint_error | 1024 | 0.064755 | 0.074673 | 0.044343 | 0.000000 | 0.582950 | 0.000089 | 0.017091 | 0.083059 | 0.206792 |
| savgol_w5_p2 | missing_only | path_length_error | 1024 | 0.042513 | 0.089986 | 0.014159 | 0.000000 | 0.898628 | 0.000037 | 0.003775 | 0.037299 | 0.194712 |
| savgol_w5_p2 | missing_only | acceleration_error | 1024 | 0.054704 | 0.052474 | 0.031066 | 0.000001 | 0.263280 | 0.006527 | 0.018309 | 0.084991 | 0.162532 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | masked_ADE | 1024 | 0.277678 | 0.253865 | 0.224637 | 0.000000 | 2.456559 | 0.000139 | 0.103496 | 0.376745 | 0.706803 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | masked_RMSE | 1024 | 0.216318 | 0.197787 | 0.173482 | 0.000000 | 1.911021 | 0.000107 | 0.082727 | 0.289861 | 0.556008 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | endpoint_error | 1024 | 0.442780 | 0.417500 | 0.350486 | 0.000000 | 3.950963 | 0.000198 | 0.149969 | 0.605453 | 1.171946 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | path_length_error | 1024 | 0.321851 | 0.588189 | 0.090238 | 0.000000 | 6.105679 | 0.000054 | 0.021161 | 0.349937 | 1.450157 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | acceleration_error | 1024 | 0.120690 | 0.103476 | 0.097670 | 0.000000 | 0.946725 | 0.011372 | 0.050388 | 0.161984 | 0.299258 |
| ddpm_v3_inpainting | missing_only | masked_ADE | 5120 | 0.304263 | 0.230202 | 0.251038 | 0.001811 | 1.800232 | 0.038645 | 0.128246 | 0.429333 | 0.755161 |
| ddpm_v3_inpainting | missing_only | masked_RMSE | 5120 | 0.239656 | 0.179509 | 0.198310 | 0.001522 | 1.402279 | 0.030521 | 0.101651 | 0.340630 | 0.589194 |
| ddpm_v3_inpainting | missing_only | endpoint_error | 5120 | 0.488641 | 0.373420 | 0.408063 | 0.003734 | 2.807095 | 0.057209 | 0.200453 | 0.690761 | 1.210126 |
| ddpm_v3_inpainting | missing_only | path_length_error | 5120 | 0.402438 | 0.383047 | 0.282567 | 0.000037 | 2.860419 | 0.016277 | 0.106098 | 0.593470 | 1.186653 |
| ddpm_v3_inpainting | missing_only | acceleration_error | 5120 | 0.045478 | 0.033117 | 0.037074 | 0.000827 | 0.200183 | 0.006446 | 0.023004 | 0.058874 | 0.116840 |
| ddpm_v3_inpainting_anchored | missing_only | masked_ADE | 5120 | 0.089707 | 0.068702 | 0.073027 | 0.001390 | 0.556781 | 0.012279 | 0.041708 | 0.117839 | 0.226266 |
| ddpm_v3_inpainting_anchored | missing_only | masked_RMSE | 5120 | 0.068152 | 0.051644 | 0.055786 | 0.001075 | 0.395377 | 0.009366 | 0.032026 | 0.089227 | 0.171439 |
| ddpm_v3_inpainting_anchored | missing_only | endpoint_error | 5120 | 0.075014 | 0.070176 | 0.055867 | 0.000531 | 0.575366 | 0.007020 | 0.028002 | 0.098790 | 0.211658 |
| ddpm_v3_inpainting_anchored | missing_only | path_length_error | 5120 | 0.046285 | 0.091735 | 0.012255 | 0.000005 | 0.911161 | 0.000548 | 0.003710 | 0.039792 | 0.226224 |
| ddpm_v3_inpainting_anchored | missing_only | acceleration_error | 5120 | 0.037931 | 0.031254 | 0.029226 | 0.000563 | 0.196546 | 0.004869 | 0.017388 | 0.047541 | 0.107836 |
