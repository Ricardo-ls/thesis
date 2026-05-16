# Stage 4 E1 Full Experiment: Oracle Confidence Residual Gating

## Objective and hypothesis
E1 tests whether oracle per-frame confidence can control the Stage 3 conditional residual correction and reduce over-refinement. It does not redesign the model, retrain, use SDEdit, or sample a new conditional output.

Unified data terminology: Stage 3 protocol-validated per-frame conditional outputs. This includes archived_original outputs and matched_protocol_recomputed outputs that reproduce Stage 3 ADE under the same protocol.

## Data sources
- clean trajectories: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/data/stage3_indoor/clean_trajs.npy`
- Stage 3 gaussian summary: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/cond_residual_gaussian_summary.csv`
- Stage 3 generalization summary: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_summary.csv`

Loaded array inventory:
- gaussian_medium: degraded_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/eval_degraded_gaussian.npy`; cond_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/gaussian_cond_residual_t20_refined.npy`; source=archived_original
- drift_medium: degraded_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_drift.npy`; cond_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/drift_cond_residual_t20_refined.npy`; source=archived_original
- jump_medium: degraded_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_jump.npy`; cond_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/reconstructed_stage3_cond_outputs/jump_cond_residual_t20_refined_reconstructed.npy`; source=matched_protocol_recomputed
- burst_medium: degraded_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_burst.npy`; cond_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/burst_cond_residual_t20_refined.npy`; source=archived_original
- bias_medium: degraded_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_bias.npy`; cond_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/bias_cond_residual_t20_refined.npy`; source=archived_original
- combined_medium: degraded_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_combined.npy`; cond_exists=True `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/reconstructed_stage3_cond_outputs/combined_cond_residual_t20_refined_reconstructed.npy`; source=matched_protocol_recomputed

Loaded shapes:
- gaussian_medium: clean=(200, 20, 2), degraded=(200, 20, 2), cond=(200, 20, 2)
- drift_medium: clean=(200, 20, 2), degraded=(200, 20, 2), cond=(200, 20, 2)
- jump_medium: clean=(200, 20, 2), degraded=(200, 20, 2), cond=(200, 20, 2)
- burst_medium: clean=(200, 20, 2), degraded=(200, 20, 2), cond=(200, 20, 2)
- bias_medium: clean=(200, 20, 2), degraded=(200, 20, 2), cond=(200, 20, 2)
- combined_medium: clean=(200, 20, 2), degraded=(200, 20, 2), cond=(200, 20, 2)

## Stage 3 ADE reproduction check
- recomputed gaussian cond_residual_t20 ADE mean: 0.055695
- official Stage 3 gaussian cond_residual_t20 ADE mean: 0.055695
- recomputed gaussian cond_residual_t20 ADE std: 0.016248
- official Stage 3 gaussian cond_residual_t20 ADE std: 0.016248
- reproduction passed: TRUE

## Source / provenance table
| degradation | conditional_source | computed_cond_ADE_mean | stage3_summary_cond_ADE_mean | cond_ADE_mean_abs_diff | protocol_validation_pass |
| --- | --- | --- | --- | --- | --- |
| gaussian_medium | archived_original | 0.055695 | 0.055695 | 0.000000 | True |
| drift_medium | archived_original | 0.031591 | 0.031591 | 0.000000 | True |
| jump_medium | matched_protocol_recomputed | 0.246523 | 0.246523 | 0.000000 | True |
| burst_medium | archived_original | 0.091885 | 0.091885 | 0.000000 | True |
| bias_medium | archived_original | 0.188868 | 0.188868 | 0.000000 | True |
| combined_medium | matched_protocol_recomputed | 0.197429 | 0.197429 | 0.000000 | True |

## Method formula
- r_hat = x_cond - y
- c_t = exp(- ||y_t - x*_t|| / delta_0)
- lambda_t = 1 - c_t
- x_E1,t = y_t + lambda_t * r_hat_t

## Delta_0 setting
Main E1 metrics use an automatic per-degradation delta_0 equal to the median degraded-input frame error over the full evaluation set. Fixed values [0.02, 0.05, 0.10, 0.20] are reported separately in the sensitivity sweep.

## Full method x degradation metric table
| degradation | e1_oracle_gated_residual | noisy_input | stage3_cond_residual_t20 |
| --- | --- | --- | --- |
| bias_medium | 0.187644 | 0.188996 | 0.188868 |
| burst_medium | 0.073922 | 0.074755 | 0.091885 |
| combined_medium | 0.194204 | 0.199064 | 0.197429 |
| drift_medium | 0.023272 | 0.017648 | 0.031591 |
| gaussian_medium | 0.048260 | 0.062567 | 0.055695 |
| jump_medium | 0.246264 | 0.263368 | 0.246523 |

## Confidence-bin analysis
| degradation | method | confidence_bin | N_frames | ADE_mean | ADE_std | ADE_median |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | noisy_input | high | 334 | 0.014083 | 0.004760 | 0.014848 |
| gaussian_medium | stage3_cond_residual_t20 | high | 334 | 0.044177 | 0.027213 | 0.040041 |
| gaussian_medium | e1_oracle_gated_residual | high | 334 | 0.015238 | 0.008354 | 0.014625 |
| gaussian_medium | noisy_input | mid | 2177 | 0.046816 | 0.013634 | 0.047015 |
| gaussian_medium | stage3_cond_residual_t20 | mid | 2177 | 0.048853 | 0.027127 | 0.044759 |
| gaussian_medium | e1_oracle_gated_residual | mid | 2177 | 0.038979 | 0.019785 | 0.036694 |
| gaussian_medium | noisy_input | low | 1489 | 0.096470 | 0.020663 | 0.092009 |
| gaussian_medium | stage3_cond_residual_t20 | low | 1489 | 0.068283 | 0.034671 | 0.065513 |
| gaussian_medium | e1_oracle_gated_residual | low | 1489 | 0.069237 | 0.031262 | 0.067295 |
| drift_medium | noisy_input | high | 581 | 0.002408 | 0.002033 | 0.002763 |
| drift_medium | stage3_cond_residual_t20 | high | 581 | 0.014949 | 0.015574 | 0.012717 |
| drift_medium | e1_oracle_gated_residual | high | 581 | 0.003635 | 0.004026 | 0.002766 |
| drift_medium | noisy_input | mid | 1815 | 0.012141 | 0.003730 | 0.012010 |
| drift_medium | stage3_cond_residual_t20 | mid | 1815 | 0.029475 | 0.018139 | 0.026012 |
| drift_medium | e1_oracle_gated_residual | mid | 1815 | 0.018201 | 0.011887 | 0.015413 |
| drift_medium | noisy_input | low | 1604 | 0.029400 | 0.008862 | 0.027104 |
| drift_medium | stage3_cond_residual_t20 | low | 1604 | 0.040013 | 0.022484 | 0.036489 |
| drift_medium | e1_oracle_gated_residual | low | 1604 | 0.036123 | 0.020386 | 0.032846 |
| jump_medium | noisy_input | high | 986 | 0.000000 | 0.000000 | 0.000000 |
| jump_medium | stage3_cond_residual_t20 | high | 986 | 0.026686 | 0.026608 | 0.020582 |
| jump_medium | e1_oracle_gated_residual | high | 986 | 0.000000 | 0.000000 | 0.000000 |
| jump_medium | noisy_input | mid | 1675 | 0.285296 | 0.044582 | 0.289903 |
| jump_medium | stage3_cond_residual_t20 | mid | 1675 | 0.260577 | 0.068040 | 0.261242 |
| jump_medium | e1_oracle_gated_residual | mid | 1675 | 0.268952 | 0.053551 | 0.267064 |
| jump_medium | noisy_input | low | 1339 | 0.429875 | 0.040297 | 0.431185 |
| jump_medium | stage3_cond_residual_t20 | low | 1339 | 0.390824 | 0.059442 | 0.393337 |
| jump_medium | e1_oracle_gated_residual | low | 1339 | 0.399224 | 0.049602 | 0.398840 |
| burst_medium | noisy_input | high | 351 | 0.003384 | 0.001181 | 0.003594 |
| burst_medium | stage3_cond_residual_t20 | high | 351 | 0.051637 | 0.041084 | 0.039875 |
| burst_medium | e1_oracle_gated_residual | high | 351 | 0.011275 | 0.009347 | 0.008348 |
| burst_medium | noisy_input | mid | 2073 | 0.010783 | 0.003299 | 0.010690 |
| burst_medium | stage3_cond_residual_t20 | mid | 2073 | 0.051580 | 0.043346 | 0.036415 |
| burst_medium | e1_oracle_gated_residual | mid | 2073 | 0.027593 | 0.023396 | 0.019222 |
| burst_medium | noisy_input | low | 1576 | 0.174797 | 0.191952 | 0.069795 |
| burst_medium | stage3_cond_residual_t20 | low | 1576 | 0.153864 | 0.143019 | 0.106378 |
| burst_medium | e1_oracle_gated_residual | low | 1576 | 0.148815 | 0.145618 | 0.094722 |
| bias_medium | noisy_input | high | 220 | 0.042336 | 0.013585 | 0.048183 |
| bias_medium | stage3_cond_residual_t20 | high | 220 | 0.047505 | 0.022163 | 0.049633 |
| bias_medium | e1_oracle_gated_residual | high | 220 | 0.041958 | 0.014743 | 0.042579 |
| bias_medium | noisy_input | mid | 2220 | 0.133953 | 0.037313 | 0.134148 |
| bias_medium | stage3_cond_residual_t20 | mid | 2220 | 0.135918 | 0.039186 | 0.134370 |
| bias_medium | e1_oracle_gated_residual | mid | 2220 | 0.133933 | 0.037303 | 0.131986 |
| bias_medium | noisy_input | low | 1560 | 0.288008 | 0.065810 | 0.275063 |
| bias_medium | stage3_cond_residual_t20 | low | 1560 | 0.284157 | 0.067435 | 0.267522 |
| bias_medium | e1_oracle_gated_residual | low | 1560 | 0.284625 | 0.066293 | 0.266604 |
| combined_medium | noisy_input | high | 290 | 0.044971 | 0.015676 | 0.046756 |
| combined_medium | stage3_cond_residual_t20 | high | 290 | 0.074983 | 0.035018 | 0.073079 |
| combined_medium | e1_oracle_gated_residual | high | 290 | 0.046620 | 0.018140 | 0.045893 |
| combined_medium | noisy_input | mid | 2257 | 0.150219 | 0.043471 | 0.150413 |
| combined_medium | stage3_cond_residual_t20 | mid | 2257 | 0.150917 | 0.052231 | 0.150537 |
| combined_medium | e1_oracle_gated_residual | mid | 2257 | 0.147806 | 0.045582 | 0.146692 |
| combined_medium | noisy_input | low | 1453 | 0.305691 | 0.062798 | 0.289861 |
| combined_medium | stage3_cond_residual_t20 | low | 1453 | 0.294118 | 0.072603 | 0.281438 |
| combined_medium | e1_oracle_gated_residual | low | 1453 | 0.295732 | 0.068061 | 0.281106 |

## High-confidence no-harm analysis
| degradation | N_high_frames | ADE_noisy_high | ADE_cond_high | ADE_e1_high | C2_high_no_harm_pass | no_harm_frame_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 334 | 0.014083 | 0.044177 | 0.015238 | True | 0.562874 |
| drift_medium | 581 | 0.002408 | 0.014949 | 0.003635 | False | 0.583477 |
| jump_medium | 986 | 0.000000 | 0.026686 | 0.000000 | True | 1.000000 |
| burst_medium | 351 | 0.003384 | 0.051637 | 0.011275 | False | 0.185185 |
| bias_medium | 220 | 0.042336 | 0.047505 | 0.041958 | True | 0.890909 |
| combined_medium | 290 | 0.044971 | 0.074983 | 0.046620 | True | 0.627586 |

## Low-confidence correction-preservation analysis
| degradation | N_low_frames | ADE_noisy_low | ADE_cond_low | ADE_e1_low | conditional_useful_low | C3_low_preservation_pass | low_confidence_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 1489 | 0.096470 | 0.068283 | 0.069237 | True | True | conditional useful; E1 close-to-or-better-than conditional |
| drift_medium | 1604 | 0.029400 | 0.040013 | 0.036123 | False | True | no useful conditional correction existed to preserve |
| jump_medium | 1339 | 0.429875 | 0.390824 | 0.399224 | True | True | conditional useful; E1 close-to-or-better-than conditional |
| burst_medium | 1576 | 0.174797 | 0.153864 | 0.148815 | True | True | conditional useful; E1 close-to-or-better-than conditional |
| bias_medium | 1560 | 0.288008 | 0.284157 | 0.284625 | True | True | conditional useful; E1 close-to-or-better-than conditional |
| combined_medium | 1453 | 0.305691 | 0.294118 | 0.295732 | True | True | conditional useful; E1 close-to-or-better-than conditional |

## Delta_0 sensitivity
| degradation | delta0_label | delta0 | ADE_noisy | ADE_cond | ADE_e1 | ADE_e1_high | ADE_e1_low | win_rate_vs_conditional | N_high_frames | N_mid_frames | N_low_frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | fixed_0.02 | 0.020000 | 0.062567 | 0.055695 | 0.051854 | 0.009418 | 0.055064 | 0.980000 | 37 | 400 | 3563 |
| gaussian_medium | fixed_0.05 | 0.050000 | 0.062567 | 0.055695 | 0.048304 | 0.014069 | 0.064037 | 0.880000 | 242 | 1821 | 1937 |
| gaussian_medium | fixed_0.10 | 0.100000 | 0.062567 | 0.055695 | 0.049931 | 0.021634 | 0.101530 | 0.635000 | 876 | 2931 | 193 |
| gaussian_medium | fixed_0.20 | 0.200000 | 0.062567 | 0.055695 | 0.054086 | 0.038435 | NaN | 0.415000 | 2533 | 1467 | 0 |
| gaussian_medium | auto_median_degraded_error | 0.058828 | 0.062567 | 0.055695 | 0.048260 | 0.015238 | 0.069237 | 0.825000 | 334 | 2177 | 1489 |
| drift_medium | fixed_0.02 | 0.020000 | 0.017648 | 0.031591 | 0.021851 | 0.004500 | 0.038323 | 0.995000 | 764 | 2165 | 1071 |
| drift_medium | fixed_0.05 | 0.050000 | 0.017648 | 0.031591 | 0.018329 | 0.009868 | 0.028606 | 0.970000 | 2263 | 1726 | 11 |
| drift_medium | fixed_0.10 | 0.100000 | 0.017648 | 0.031591 | 0.017600 | 0.015493 | NaN | 0.930000 | 3678 | 322 | 0 |
| drift_medium | fixed_0.20 | 0.200000 | 0.017648 | 0.031591 | 0.017495 | 0.017495 | NaN | 0.920000 | 4000 | 0 | 0 |
| drift_medium | auto_median_degraded_error | 0.015678 | 0.017648 | 0.031591 | 0.023272 | 0.003635 | 0.036123 | 0.995000 | 581 | 1815 | 1604 |
| jump_medium | fixed_0.02 | 0.020000 | 0.263368 | 0.246523 | 0.239945 | 0.000000 | 0.318440 | 0.910000 | 986 | 0 | 3014 |
| jump_medium | fixed_0.05 | 0.050000 | 0.263368 | 0.246523 | 0.239976 | 0.000000 | 0.318481 | 0.785000 | 986 | 0 | 3014 |
| jump_medium | fixed_0.10 | 0.100000 | 0.263368 | 0.246523 | 0.240555 | 0.000000 | 0.319250 | 0.745000 | 986 | 0 | 3014 |
| jump_medium | fixed_0.20 | 0.200000 | 0.263368 | 0.246523 | 0.243320 | 0.000000 | 0.338172 | 0.630000 | 986 | 356 | 2658 |
| jump_medium | auto_median_degraded_error | 0.302825 | 0.263368 | 0.246523 | 0.246264 | 0.000000 | 0.399224 | 0.550000 | 986 | 1675 | 1339 |
| burst_medium | fixed_0.02 | 0.020000 | 0.074755 | 0.091885 | 0.069872 | 0.011309 | 0.215154 | 1.000000 | 707 | 2316 | 977 |
| burst_medium | fixed_0.05 | 0.050000 | 0.074755 | 0.091885 | 0.062785 | 0.012556 | 0.254290 | 1.000000 | 2533 | 672 | 795 |
| burst_medium | fixed_0.10 | 0.100000 | 0.074755 | 0.091885 | 0.061098 | 0.012693 | 0.269521 | 1.000000 | 3191 | 74 | 735 |
| burst_medium | fixed_0.20 | 0.200000 | 0.074755 | 0.091885 | 0.062245 | 0.012484 | 0.333567 | 0.995000 | 3213 | 285 | 502 |
| burst_medium | auto_median_degraded_error | 0.014054 | 0.074755 | 0.091885 | 0.073922 | 0.011275 | 0.148815 | 1.000000 | 351 | 2073 | 1576 |
| bias_medium | fixed_0.02 | 0.020000 | 0.188996 | 0.188868 | 0.188739 | NaN | 0.190493 | 0.715000 | 0 | 40 | 3960 |
| bias_medium | fixed_0.05 | 0.050000 | 0.188996 | 0.188868 | 0.188248 | NaN | 0.196029 | 0.550000 | 0 | 200 | 3800 |
| bias_medium | fixed_0.10 | 0.100000 | 0.188996 | 0.188868 | 0.187770 | 0.023088 | 0.224212 | 0.525000 | 60 | 1000 | 2940 |
| bias_medium | fixed_0.20 | 0.200000 | 0.188996 | 0.188868 | 0.187661 | 0.048542 | 0.304162 | 0.520000 | 280 | 2540 | 1180 |
| bias_medium | auto_median_degraded_error | 0.172925 | 0.188996 | 0.188868 | 0.187644 | 0.041958 | 0.284625 | 0.520000 | 220 | 2220 | 1560 |
| combined_medium | fixed_0.02 | 0.020000 | 0.199064 | 0.197429 | 0.196858 | 0.008928 | 0.198328 | 0.935000 | 5 | 31 | 3964 |
| combined_medium | fixed_0.05 | 0.050000 | 0.199064 | 0.197429 | 0.195396 | 0.015483 | 0.204360 | 0.840000 | 17 | 218 | 3765 |
| combined_medium | fixed_0.10 | 0.100000 | 0.199064 | 0.197429 | 0.194247 | 0.027909 | 0.228129 | 0.660000 | 82 | 860 | 3058 |
| combined_medium | fixed_0.20 | 0.200000 | 0.199064 | 0.197429 | 0.194260 | 0.049172 | 0.306263 | 0.585000 | 323 | 2422 | 1255 |
| combined_medium | auto_median_degraded_error | 0.188138 | 0.199064 | 0.197429 | 0.194204 | 0.046620 | 0.295732 | 0.585000 | 290 | 2257 | 1453 |

## Representative figures
- representative_gaussian_medium: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/representative_gaussian_medium_gaussian_medium_traj100.png gaussian_medium, trajectory 100
- representative_drift_medium: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/representative_drift_medium_drift_medium_traj53.png drift_medium, trajectory 53
- representative_jump_medium: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/representative_jump_medium_jump_medium_traj143.png jump_medium, trajectory 143
- representative_burst_medium: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/representative_burst_medium_burst_medium_traj107.png burst_medium, trajectory 107
- representative_bias_medium: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/representative_bias_medium_bias_medium_traj154.png bias_medium, trajectory 154
- representative_combined_medium: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/representative_combined_medium_combined_medium_traj123.png combined_medium, trajectory 123
- high_confidence_overcorrection: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/high_confidence_overcorrection_burst_medium_traj107.png burst_medium, trajectory 107
- low_confidence_preserved: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/low_confidence_preserved_burst_medium_traj85.png burst_medium, trajectory 85
- failure_or_ambiguous_case: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/figures/failure_or_ambiguous_case_burst_medium_traj165.png burst_medium, trajectory 165

## PASS / NO-PASS judgment
- C1: TRUE
- C1 note: E1 beats conditional in 6/6 formal conditions; attention wins ['drift_medium', 'burst_medium', 'combined_medium']; missing required per-frame outputs: []
- C2: FALSE
- C2 failures: ['drift_medium', 'burst_medium']
- C3: TRUE
- C3 failures: []
- Overall: NO-PASS

## Negative result analysis
If any criterion fails, the failure should be read as a limitation of linear confidence control rather than as a failure to load Stage 3 outputs. Bias and posterior anchoring remain outside E1's mechanism.

Boundary statement: E1 tests confidence control of residual correction. Bias and posterior anchoring are E2/E4 targets.
