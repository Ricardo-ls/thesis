# E2-DPS Stage 8A Pilot Summary

This is Stage 8A only. Stage 8B was not started.

Guidance: Amendment 002 canonical norm-based DPS guidance. Overshoot caution remains active because the small re-audit was Case B.

## Six-Condition Mean ADE
| zeta | six_condition_mean_ADE |
| --- | --- |
| 0.300000 | 2.274388 |
| 1.000000 | 5.587513 |
| 3.000000 | 16.042216 |

## H1/H2 Decision Gate
| zeta | H1_drift_high_conf_no_harm_ratio | H1_valid_finite | H1_pass | H2_burst_high_conf_no_harm_ratio | H2_valid_finite | H2_pass | bias_ADE_E2DPS | bias_ADE_E1 | bias_red_flag | any_core_signal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.300000 | 1.084750 | True | False | 1.485396 | True | False | 2.272162 | 1.602825 | False | False |
| 1.000000 | 1.773175 | True | False | 3.787598 | True | False | 5.524973 | 1.602825 | False | False |
| 3.000000 | 3.776743 | True | False | 10.313890 | True | False | 16.237520 | 1.602825 | False | False |

## Numerical Stability
DPS rows with non-finite ADE:
_No rows._

DPS rows with non-finite sampling count:
_No rows._

## Diagnostics
| degradation | zeta | method | ADE_mean | ADE_high_mean | ADE_high_noisy_mean | high_conf_no_harm_ratio | ADE_low_mean | low_conf_wilcoxon_p_vs_noisy | motion_usage_ratio | noisy_reversion_gap | acceleration_RMS_mean | max_update_to_x_ratio | late_update_to_x_ratio_max | nonfinite_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 0.300000 | E2_DPS_zeta_0.3 | 2.272738 | 2.314767 | 1.529894 | 1.513024 | 2.250110 | 1.000000 | 4.075996 | 0.669627 | 0.266508 | 2.277177 | 2.142521 | 0 |
| gaussian_medium | 1.000000 | E2_DPS_zeta_1.0 | 5.681340 | 5.821750 | 1.529894 | 3.805328 | 5.660454 | 1.000000 | 13.302707 | 4.078229 | 0.759035 | 3.210831 | 2.676907 | 0 |
| gaussian_medium | 3.000000 | E2_DPS_zeta_3.0 | 16.650658 | 17.569127 | 1.529894 | 11.483883 | 16.543125 | 1.000000 | 41.250271 | 15.047546 | 2.261605 | 12.084697 | 2.770679 | 0 |
| drift_medium | 0.300000 | E2_DPS_zeta_0.3 | 2.183804 | 1.539581 | 1.419295 | 1.084750 | 2.583611 | 1.000000 | 15.171403 | 0.581041 | 0.226142 | 3.222626 | 2.825272 | 0 |
| drift_medium | 1.000000 | E2_DPS_zeta_1.0 | 5.575216 | 2.516658 | 1.419295 | 1.773175 | 7.162293 | 1.000000 | 55.378286 | 3.972453 | 0.724065 | 3.342185 | 2.833653 | 0 |
| drift_medium | 3.000000 | E2_DPS_zeta_3.0 | 15.443533 | 5.360312 | 1.419295 | 3.776743 | 20.686704 | 1.000000 | 165.595893 | 13.840770 | 2.191676 | 11.671755 | 2.425818 | 0 |
| burst_medium | 0.300000 | E2_DPS_zeta_0.3 | 2.301177 | 2.265502 | 1.525184 | 1.485396 | 2.281816 | 1.000000 | 3.451920 | 0.692500 | 0.363800 | 2.259989 | 2.249347 | 0 |
| burst_medium | 1.000000 | E2_DPS_zeta_1.0 | 5.606744 | 5.776782 | 1.525184 | 3.787598 | 5.613348 | 1.000000 | 10.508266 | 3.998067 | 0.773422 | 3.443358 | 2.615494 | 0 |
| burst_medium | 3.000000 | E2_DPS_zeta_3.0 | 15.942527 | 15.730578 | 1.525184 | 10.313890 | 16.411904 | 1.000000 | 31.654185 | 14.333850 | 2.287838 | 10.891079 | 2.751457 | 0 |
| bias_medium | 0.300000 | E2_DPS_zeta_0.3 | 2.272162 | 2.482180 | 1.629856 | 1.522945 | 2.213646 | 0.999995 | 17.766279 | 0.668403 | 0.238339 | 3.500387 | 2.732264 | 0 |
| bias_medium | 1.000000 | E2_DPS_zeta_1.0 | 5.524973 | 5.062886 | 1.629856 | 3.106340 | 5.065534 | 1.000000 | 63.227245 | 3.921213 | 0.740943 | 3.474750 | 2.777520 | 0 |
| bias_medium | 3.000000 | E2_DPS_zeta_3.0 | 16.237520 | 16.247375 | 1.629856 | 9.968596 | 15.541853 | 1.000000 | 198.815815 | 14.633760 | 2.269731 | 11.030971 | 2.435310 | 0 |
| jump_medium | 0.300000 | E2_DPS_zeta_0.3 | 2.271287 | 1.694070 | 1.519952 | 1.114554 | 2.461916 | 1.000000 | 6.433413 | 0.665138 | 0.301953 | 2.759782 | 2.471922 | 0 |
| jump_medium | 1.000000 | E2_DPS_zeta_1.0 | 5.536825 | 2.725170 | 1.519952 | 1.792931 | 6.246252 | 1.000000 | 20.293784 | 3.930676 | 0.740714 | 3.873293 | 2.646518 | 0 |
| jump_medium | 3.000000 | E2_DPS_zeta_3.0 | 16.419656 | 6.403952 | 1.519952 | 4.213258 | 18.821557 | 1.000000 | 63.045122 | 14.813507 | 2.214904 | 9.960217 | 3.026435 | 0 |
| combined_medium | 0.300000 | E2_DPS_zeta_0.3 | 2.345160 | 2.274446 | 1.415321 | 1.607018 | 2.488160 | 1.000000 | 4.704286 | 0.720747 | 0.267493 | 2.346731 | 2.320891 | 0 |
| combined_medium | 1.000000 | E2_DPS_zeta_1.0 | 5.599982 | 6.319439 | 1.415321 | 4.465022 | 5.802454 | 1.000000 | 15.085312 | 3.975569 | 0.734916 | 3.332572 | 2.636926 | 0 |
| combined_medium | 3.000000 | E2_DPS_zeta_3.0 | 15.559403 | 18.447672 | 1.415321 | 13.034267 | 16.582130 | 1.000000 | 44.867521 | 13.934990 | 2.241729 | 12.913029 | 2.550487 | 0 |

## Figures
- drift: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_pilot/figures/drift_drift_medium_traj9.png
- burst: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_pilot/figures/burst_burst_medium_traj9.png
- bias_negative_control: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_pilot/figures/bias_negative_control_bias_medium_traj11.png
- jump_diagnostic: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_pilot/figures/jump_diagnostic_jump_medium_traj14.png
