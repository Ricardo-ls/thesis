# exp01_ethucy_indomain_quick Full Stats Matrix

Population note:
- deterministic methods: `N=128` trajectories
- DDPM methods: `N=128 × 5 = 640` trajectory-seed cases

| method | missing_condition | metric | N | mean | std | median | min | max | p05 | p25 | p75 | p95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_interp | missing_only | masked_ADE | 128 | 0.117882 | 0.096267 | 0.090939 | 0.000000 | 0.379674 | 0.000000 | 0.038711 | 0.193341 | 0.292653 |
| linear_interp | missing_only | masked_RMSE | 128 | 0.088840 | 0.071852 | 0.065301 | 0.000000 | 0.273148 | 0.000000 | 0.029549 | 0.142321 | 0.217141 |
| linear_interp | missing_only | endpoint_error | 128 | 0.098784 | 0.097943 | 0.070128 | 0.000000 | 0.428896 | 0.000000 | 0.025865 | 0.146326 | 0.314179 |
| linear_interp | missing_only | path_length_error | 128 | 0.069619 | 0.128912 | 0.021415 | 0.000000 | 0.796797 | 0.000000 | 0.000000 | 0.079657 | 0.303330 |
| linear_interp | missing_only | acceleration_error | 128 | 0.054365 | 0.043241 | 0.048411 | 0.000000 | 0.184166 | 0.000000 | 0.018702 | 0.085764 | 0.128978 |
| savgol_w5_p2 | missing_only | masked_ADE | 128 | 0.116141 | 0.094873 | 0.089454 | 0.000000 | 0.378728 | 0.000000 | 0.036871 | 0.187464 | 0.290804 |
| savgol_w5_p2 | missing_only | masked_RMSE | 128 | 0.088048 | 0.071053 | 0.064654 | 0.000000 | 0.272721 | 0.000000 | 0.028697 | 0.139962 | 0.217833 |
| savgol_w5_p2 | missing_only | endpoint_error | 128 | 0.096247 | 0.096406 | 0.068057 | 0.000000 | 0.446800 | 0.000000 | 0.031690 | 0.139311 | 0.307707 |
| savgol_w5_p2 | missing_only | path_length_error | 128 | 0.086816 | 0.122552 | 0.042447 | 0.000000 | 0.757102 | 0.000000 | 0.014269 | 0.125361 | 0.281320 |
| savgol_w5_p2 | missing_only | acceleration_error | 128 | 0.102549 | 0.053862 | 0.125324 | 0.000001 | 0.263280 | 0.005679 | 0.061742 | 0.136140 | 0.160534 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | masked_ADE | 128 | 0.320332 | 0.257278 | 0.259567 | 0.000000 | 0.969671 | 0.000000 | 0.115428 | 0.524409 | 0.771476 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | masked_RMSE | 128 | 0.248345 | 0.197583 | 0.201005 | 0.000000 | 0.746836 | 0.000000 | 0.087067 | 0.426697 | 0.592234 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | endpoint_error | 128 | 0.496470 | 0.413126 | 0.384893 | 0.000000 | 1.555618 | 0.000000 | 0.175911 | 0.885380 | 1.246613 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | path_length_error | 128 | 0.405179 | 0.564207 | 0.120787 | 0.000000 | 2.216175 | 0.000000 | 0.020742 | 0.545693 | 1.747014 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | acceleration_error | 128 | 0.146430 | 0.102128 | 0.124050 | 0.000000 | 0.420937 | 0.004904 | 0.073877 | 0.221347 | 0.323809 |
| ddpm_v3_inpainting | missing_only | masked_ADE | 640 | 0.307332 | 0.230151 | 0.254322 | 0.007918 | 1.733692 | 0.032709 | 0.121792 | 0.457011 | 0.739384 |
| ddpm_v3_inpainting | missing_only | masked_RMSE | 640 | 0.239805 | 0.176179 | 0.206190 | 0.005748 | 1.339861 | 0.025164 | 0.099791 | 0.355524 | 0.564306 |
| ddpm_v3_inpainting | missing_only | endpoint_error | 640 | 0.469053 | 0.355806 | 0.394472 | 0.010313 | 2.734710 | 0.043709 | 0.180067 | 0.673784 | 1.138165 |
| ddpm_v3_inpainting | missing_only | path_length_error | 640 | 0.389062 | 0.363509 | 0.260191 | 0.000094 | 2.781198 | 0.024294 | 0.112688 | 0.586613 | 1.134697 |
| ddpm_v3_inpainting | missing_only | acceleration_error | 640 | 0.063810 | 0.041917 | 0.058420 | 0.002026 | 0.197629 | 0.006038 | 0.031239 | 0.092120 | 0.137324 |
| ddpm_v3_inpainting_anchored | missing_only | masked_ADE | 640 | 0.126066 | 0.085908 | 0.115917 | 0.001695 | 0.373314 | 0.013374 | 0.052599 | 0.185448 | 0.283362 |
| ddpm_v3_inpainting_anchored | missing_only | masked_RMSE | 640 | 0.095714 | 0.064546 | 0.090295 | 0.001379 | 0.273321 | 0.010123 | 0.040173 | 0.140949 | 0.209067 |
| ddpm_v3_inpainting_anchored | missing_only | endpoint_error | 640 | 0.105415 | 0.093445 | 0.081451 | 0.001739 | 0.550483 | 0.007848 | 0.035026 | 0.143482 | 0.303870 |
| ddpm_v3_inpainting_anchored | missing_only | path_length_error | 640 | 0.100297 | 0.135726 | 0.041851 | 0.000024 | 0.793598 | 0.002551 | 0.017872 | 0.129729 | 0.386261 |
| ddpm_v3_inpainting_anchored | missing_only | acceleration_error | 640 | 0.058667 | 0.040990 | 0.053288 | 0.001037 | 0.192752 | 0.004440 | 0.025536 | 0.088695 | 0.131970 |
