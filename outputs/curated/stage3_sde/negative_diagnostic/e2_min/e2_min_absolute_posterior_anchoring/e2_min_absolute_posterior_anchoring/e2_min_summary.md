# Stage 4 E2-Min Absolute-Space Posterior Anchoring

## Objective and Hypothesis
E2-Min tests deterministic confidence-modulated absolute-space posterior anchoring after E1. E1 is complete: it showed confidence is useful, but scalar residual gating failed high-confidence no-harm for drift and burst.

## Selection Rule
Per the updated protocol, the selected shared setting is chosen only by six-condition ADE improvement versus Formal E1: maximize the number of degradations with ADE_E2 <= ADE_E1, with ties broken by mean ADE gain and then mean ADE.

## Selected Setting
- selected variant: V2_uniform_cond_motion
- alpha0: 0.500000
- beta: 0.050000
- sigma0: 1.000000
- six-condition ADE improvements: 4/6
- mean ADE gain vs Formal E1: 0.004450

## Formula
V1: alpha_t = 0. V2: alpha_t = alpha0. V3: alpha_t = alpha0 * (1 - c_t). Each trajectory is solved deterministically in absolute space from the quadratic objective with observation anchoring, optional conditional motion reference, and acceleration smoothness.

## Data Provenance
| degradation | conditional_source | conditional_path | degraded_path | shape_clean | shape_cond | protocol_validation_pass |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | archived_original | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/gaussian_cond_residual_t20_refined.npy | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/eval_degraded_gaussian.npy | (200, 20, 2) | (200, 20, 2) | True |
| drift_medium | archived_original | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/drift_cond_residual_t20_refined.npy | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_drift.npy | (200, 20, 2) | (200, 20, 2) | True |
| jump_medium | matched_protocol_recomputed | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/reconstructed_stage3_cond_outputs/jump_cond_residual_t20_refined_reconstructed.npy | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_jump.npy | (200, 20, 2) | (200, 20, 2) | True |
| burst_medium | archived_original | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/burst_cond_residual_t20_refined.npy | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_burst.npy | (200, 20, 2) | (200, 20, 2) | True |
| bias_medium | archived_original | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/bias_cond_residual_t20_refined.npy | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_bias.npy | (200, 20, 2) | (200, 20, 2) | True |
| combined_medium | matched_protocol_recomputed | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/reconstructed_stage3_cond_outputs/combined_cond_residual_t20_refined_reconstructed.npy | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_combined.npy | (200, 20, 2) | (200, 20, 2) | True |

## Full ADE Matrix
| degradation | V1_abs_anchor_smooth | V2_uniform_cond_motion | V3_conf_mod_cond_motion | formal_e1_linear_gate | noisy_input | stage3_cond_residual_t20 |
| --- | --- | --- | --- | --- | --- | --- |
| bias_medium | 0.191176 | 0.189430 | 0.189756 | 0.187644 | 0.188996 | 0.188868 |
| burst_medium | 0.039268 | 0.054175 | 0.055555 | 0.073922 | 0.074755 | 0.091885 |
| combined_medium | 0.193383 | 0.190018 | 0.190899 | 0.194204 | 0.199064 | 0.197429 |
| drift_medium | 0.031611 | 0.021083 | 0.022299 | 0.023272 | 0.017648 | 0.031591 |
| gaussian_medium | 0.049931 | 0.039722 | 0.041733 | 0.048260 | 0.062567 | 0.055695 |
| jump_medium | 0.254094 | 0.252441 | 0.252810 | 0.246264 | 0.263368 | 0.246523 |

## V1 / V2 / V3 Metrics
| degradation | method | ADE_mean | RMSE_mean | motion_usage_ratio | noisy_reversion_gap | bias_offset_error |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | V1_abs_anchor_smooth | 0.049931 | 0.056832 | 1.354784 | -0.012636 | nan |
| gaussian_medium | V2_uniform_cond_motion | 0.039722 | 0.044830 | 1.025168 | -0.022844 | nan |
| gaussian_medium | V3_conf_mod_cond_motion | 0.041733 | 0.046843 | 1.066452 | -0.020833 | nan |
| drift_medium | V1_abs_anchor_smooth | 0.031611 | 0.038271 | 3.010210 | 0.013963 | nan |
| drift_medium | V2_uniform_cond_motion | 0.021083 | 0.024335 | 1.433742 | 0.003435 | nan |
| drift_medium | V3_conf_mod_cond_motion | 0.022299 | 0.025897 | 1.619878 | 0.004651 | nan |
| burst_medium | V1_abs_anchor_smooth | 0.039268 | 0.057844 | 2.584914 | -0.035488 | nan |
| burst_medium | V2_uniform_cond_motion | 0.054175 | 0.102709 | 1.285020 | -0.020580 | nan |
| burst_medium | V3_conf_mod_cond_motion | 0.055555 | 0.102953 | 1.373067 | -0.019200 | nan |
| bias_medium | V1_abs_anchor_smooth | 0.191176 | 0.192905 | 3.050695 | 0.002181 | 0.188996 |
| bias_medium | V2_uniform_cond_motion | 0.189430 | 0.189796 | 1.480129 | 0.000435 | 0.188996 |
| bias_medium | V3_conf_mod_cond_motion | 0.189756 | 0.190298 | 1.720026 | 0.000760 | 0.188996 |
| jump_medium | V1_abs_anchor_smooth | 0.254094 | 0.292390 | 1.777168 | -0.009274 | nan |
| jump_medium | V2_uniform_cond_motion | 0.252441 | 0.288535 | 1.142333 | -0.010927 | nan |
| jump_medium | V3_conf_mod_cond_motion | 0.252810 | 0.289294 | 1.223394 | -0.010558 | nan |
| combined_medium | V1_abs_anchor_smooth | 0.193383 | 0.198717 | 1.286917 | -0.005681 | nan |
| combined_medium | V2_uniform_cond_motion | 0.190018 | 0.193587 | 0.997102 | -0.009046 | nan |
| combined_medium | V3_conf_mod_cond_motion | 0.190899 | 0.194731 | 1.024013 | -0.008164 | nan |

## Confidence-Bin Analysis
| degradation | confidence_bin | N_frames | ADE_mean | ADE_std | ADE_median | ADE_p25 | ADE_p75 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | high | 334 | 0.015872 | 0.009705 | 0.014666 | 0.008941 | 0.019998 |
| gaussian_medium | mid | 2177 | 0.032640 | 0.015531 | 0.031216 | 0.021339 | 0.042249 |
| gaussian_medium | low | 1489 | 0.055427 | 0.023095 | 0.053523 | 0.039368 | 0.068788 |
| drift_medium | high | 581 | 0.006796 | 0.006038 | 0.005079 | 0.003223 | 0.007943 |
| drift_medium | mid | 1815 | 0.016134 | 0.010497 | 0.013862 | 0.009065 | 0.019961 |
| drift_medium | low | 1604 | 0.031858 | 0.014448 | 0.028744 | 0.022101 | 0.038945 |
| burst_medium | high | 351 | 0.010821 | 0.008676 | 0.007896 | 0.004542 | 0.014850 |
| burst_medium | mid | 2073 | 0.015262 | 0.011968 | 0.011269 | 0.006856 | 0.020891 |
| burst_medium | low | 1576 | 0.115016 | 0.125225 | 0.053779 | 0.016198 | 0.187984 |
| bias_medium | high | 220 | 0.044288 | 0.015263 | 0.046171 | 0.034445 | 0.055624 |
| bias_medium | mid | 2220 | 0.134363 | 0.038602 | 0.133548 | 0.102911 | 0.163848 |
| bias_medium | low | 1560 | 0.288265 | 0.067095 | 0.275650 | 0.242788 | 0.320275 |
| jump_medium | high | 986 | 0.010637 | 0.011073 | 0.006516 | 0.003525 | 0.012803 |
| jump_medium | mid | 1675 | 0.273831 | 0.052014 | 0.273113 | 0.236846 | 0.313596 |
| jump_medium | low | 1339 | 0.403742 | 0.054915 | 0.403283 | 0.371865 | 0.443166 |
| combined_medium | high | 290 | 0.049146 | 0.020967 | 0.049089 | 0.033702 | 0.063069 |
| combined_medium | mid | 2257 | 0.143877 | 0.045090 | 0.141940 | 0.110237 | 0.178537 |
| combined_medium | low | 1453 | 0.289806 | 0.064312 | 0.278278 | 0.240959 | 0.331248 |

## Bias Offset Analysis
| method | bias_offset_error |
| --- | --- |
| noisy_input | 0.188996 |
| stage3_cond_residual_t20 | 0.188100 |
| formal_e1_linear_gate | 0.187441 |
| V1_abs_anchor_smooth | 0.188996 |
| V2_uniform_cond_motion | 0.188996 |
| V3_conf_mod_cond_motion | 0.188996 |

## Parameter Sweep Top Rows
| variant | alpha0 | beta | sigma0 | six_ADE_improved_count | mean_ADE_gain_vs_e1_across_6 | mean_ADE_e2_across_6 |
| --- | --- | --- | --- | --- | --- | --- |
| V2_uniform_cond_motion | 0.500000 | 0.050000 | 1.000000 | 4 | 0.004450 | 0.124478 |
| V2_uniform_cond_motion | 0.500000 | 0.010000 | 1.000000 | 4 | 0.003902 | 0.125025 |
| V2_uniform_cond_motion | 0.500000 | 0.100000 | 0.500000 | 4 | 0.003535 | 0.125393 |
| V3_conf_mod_cond_motion | 0.500000 | 0.010000 | 1.000000 | 4 | 0.003500 | 0.125428 |
| V3_conf_mod_cond_motion | 0.500000 | 0.050000 | 1.000000 | 4 | 0.003419 | 0.125509 |
| V2_uniform_cond_motion | 0.200000 | 0.010000 | 1.000000 | 4 | 0.003318 | 0.125610 |
| V2_uniform_cond_motion | 0.500000 | 0.050000 | 0.500000 | 4 | 0.003144 | 0.125784 |
| V2_uniform_cond_motion | 0.100000 | 0.010000 | 1.000000 | 4 | 0.002836 | 0.126092 |
| V3_conf_mod_cond_motion | 0.500000 | 0.100000 | 0.500000 | 4 | 0.002676 | 0.126252 |
| V3_conf_mod_cond_motion | 0.200000 | 0.010000 | 1.000000 | 4 | 0.002644 | 0.126284 |
| V3_conf_mod_cond_motion | 0.500000 | 0.050000 | 0.500000 | 4 | 0.002428 | 0.126500 |
| V2_uniform_cond_motion | 0.500000 | 0.010000 | 0.500000 | 4 | 0.002068 | 0.126860 |

## PASS / NO-PASS
| degradation | selected_variant | ADE_noisy | ADE_cond | ADE_e1 | ADE_e2 | improves_vs_e1 | E2_6ADE_improved_count | E2_6ADE_PASS | high_conf_no_harm_pass_1p05 | low_conf_preservation_pass | E2_C5_bias_offset_improves | overall_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | V2_uniform_cond_motion | 0.062567 | 0.055695 | 0.048260 | 0.039722 | True | 4 | False | False | True | False | NO-PASS |
| drift_medium | V2_uniform_cond_motion | 0.017648 | 0.031591 | 0.023272 | 0.021083 | True | 4 | False | False | True | False | NO-PASS |
| burst_medium | V2_uniform_cond_motion | 0.074755 | 0.091885 | 0.073922 | 0.054175 | True | 4 | False | False | True | False | NO-PASS |
| bias_medium | V2_uniform_cond_motion | 0.188996 | 0.188868 | 0.187644 | 0.189431 | False | 4 | False | True | True | False | NO-PASS |
| jump_medium | V2_uniform_cond_motion | 0.263368 | 0.246523 | 0.246264 | 0.252441 | False | 4 | False | False | True | False | NO-PASS |
| combined_medium | V2_uniform_cond_motion | 0.199064 | 0.197429 | 0.194204 | 0.190018 | True | 4 | False | False | True | False | NO-PASS |

## Diagnostics
| degradation | ADE_gain_vs_e1 | ADE_e2_high | ADE_noisy_high | high_conf_no_harm_pass_1p05 | ADE_e2_low | ADE_cond_low | low_conf_preservation_pass | motion_usage_ratio | noisy_reversion_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 0.008538 | 0.015872 | 0.014083 | False | 0.055427 | 0.068283 | True | 1.025168 | -0.022844 |
| drift_medium | 0.002190 | 0.006796 | 0.002408 | False | 0.031858 | 0.040013 | True | 1.433742 | 0.003435 |
| burst_medium | 0.019747 | 0.010821 | 0.003384 | False | 0.115016 | 0.153864 | True | 1.285020 | -0.020580 |
| bias_medium | -0.001786 | 0.044288 | 0.042336 | True | 0.288265 | 0.284157 | True | 1.480129 | 0.000435 |
| jump_medium | -0.006177 | 0.010637 | 0.000000 | False | 0.403742 | 0.390824 | True | 1.142333 | -0.010927 |
| combined_medium | 0.004186 | 0.049146 | 0.044971 | False | 0.289806 | 0.294118 | True | 0.997102 | -0.009046 |

## Figures
- drift_high_confidence_e1_failure: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_min_absolute_posterior_anchoring/figures/drift_high_confidence_e1_failure_drift_medium_traj183.png (drift_medium, trajectory 183)
- burst_boundary_failure: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_min_absolute_posterior_anchoring/figures/burst_boundary_failure_burst_medium_traj107.png (burst_medium, trajectory 107)
- bias_absolute_offset: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_min_absolute_posterior_anchoring/figures/bias_absolute_offset_bias_medium_traj83.png (bias_medium, trajectory 83)
- combined_degradation: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_min_absolute_posterior_anchoring/figures/combined_degradation_combined_medium_traj36.png (combined_medium, trajectory 36)
- low_confidence_preservation: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_min_absolute_posterior_anchoring/figures/low_confidence_preservation_burst_medium_traj165.png (burst_medium, trajectory 165)
- failure_or_ambiguous: FOUND /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_min_absolute_posterior_anchoring/figures/failure_or_ambiguous_jump_medium_traj2.png (jump_medium, trajectory 2)

## E2-DPS Decision
If six-condition ADE improves but drift/burst high-confidence no-harm or bias offset remain weak, E2-DPS remains motivated as the next probabilistic posterior anchoring step. If E2-Min satisfies six-condition ADE and repairs the E1 failures, E2-DPS can be scoped as confirmatory rather than rescue.

## Boundary Statement
E2-Min is deterministic posterior anchoring in absolute space. It does not train, resample conditional outputs, use SDEdit, or modify Stage 3 data.
