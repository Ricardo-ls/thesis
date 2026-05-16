# E1 Failure Diagnosis

This diagnosis explains why formal E1 improves overall ADE but fails high-confidence no-harm for drift_medium and burst_medium. It reads Stage 4 E1 outputs and Stage 3 saved per-frame conditional residual outputs only; it does not train or resample conditional residuals.

## High-confidence failure summary
| degradation | N_trajectories | N_high_frames | N_trajectories_with_high | N_high_violation_frames | N_high_violation_trajectories | high_violation_frame_fraction | high_violation_trajectory_fraction | ADE_noisy_high | ADE_cond_high | ADE_e1_high | C2_pass | total_high_violation_excess | N_trajectories_for_50pct_excess | N_trajectories_for_80pct_excess | N_trajectories_for_90pct_excess | boundary_excess_fraction_radius1 | wrong_direction_fraction_high | wrong_direction_fraction_violation | mean_cosine_high | mean_cosine_violation | mean_residual_norm_high | mean_noisy_error_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift_medium | 200 | 581 | 200 | 242 | 95 | 0.416523 | 0.475000 | 0.002408 | 0.014949 | 0.003635 | False | 0.775892 | 20 | 47 | 66 | 0.015613 | 0.290878 | 0.677686 | 0.086329 | -0.275305 | 0.014990 | 0.002408 |
| burst_medium | 200 | 351 | 162 | 286 | 144 | 0.814815 | 0.888889 | 0.003384 | 0.051637 | 0.011275 | False | 2.716295 | 31 | 68 | 89 | 0.478015 | 0.441595 | 0.527972 | 0.052383 | -0.055170 | 0.051494 | 0.003384 |

## Trajectory dominance
| degradation | total_excess | N_trajectories_for_50pct_excess | N_trajectories_for_80pct_excess | N_trajectories_for_90pct_excess |
| --- | --- | --- | --- | --- |
| drift_medium | 0.775892 | 20 | 47 | 66 |
| burst_medium | 2.716295 | 31 | 68 | 89 |

## Worst-10 high-confidence no-harm violation trajectories
| degradation | trajectory_id | N_high_frames | N_violation_high_frames | high_violation_frame_fraction | ADE_noisy_high | ADE_cond_high | ADE_e1_high | e1_over_1p1_noisy_high | high_excess_sum | high_excess_mean | boundary_high_frames | boundary_excess_sum | wrong_direction_fraction_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift_medium | 65 | 8 | 6 | 0.750000 | 0.003208 | 0.035586 | 0.007012 | 1.986992 | 0.027967 | 0.003496 | 0 | 0.000000 | 0.250000 |
| drift_medium | 156 | 6 | 4 | 0.666667 | 0.003652 | 0.035592 | 0.008486 | 2.112609 | 0.027544 | 0.004591 | 0 | 0.000000 | 0.333333 |
| drift_medium | 160 | 5 | 3 | 0.600000 | 0.003483 | 0.029245 | 0.009242 | 2.412441 | 0.027381 | 0.005476 | 0 | 0.000000 | 0.600000 |
| drift_medium | 19 | 6 | 5 | 0.833333 | 0.004059 | 0.023174 | 0.008628 | 1.932537 | 0.024979 | 0.004163 | 0 | 0.000000 | 0.833333 |
| drift_medium | 123 | 4 | 3 | 0.750000 | 0.003100 | 0.045521 | 0.009598 | 2.815106 | 0.024755 | 0.006189 | 0 | 0.000000 | 0.250000 |
| drift_medium | 25 | 8 | 7 | 0.875000 | 0.003717 | 0.028385 | 0.006913 | 1.690691 | 0.022593 | 0.002824 | 0 | 0.000000 | 0.250000 |
| drift_medium | 53 | 4 | 2 | 0.500000 | 0.002300 | 0.043854 | 0.007928 | 3.133688 | 0.022557 | 0.005639 | 0 | 0.000000 | 0.000000 |
| drift_medium | 10 | 9 | 8 | 0.888889 | 0.003880 | 0.017676 | 0.006604 | 1.547484 | 0.021029 | 0.002337 | 0 | 0.000000 | 0.777778 |
| drift_medium | 112 | 5 | 4 | 0.800000 | 0.003284 | 0.025327 | 0.007678 | 2.125410 | 0.020327 | 0.004065 | 0 | 0.000000 | 0.800000 |
| drift_medium | 187 | 4 | 3 | 0.750000 | 0.003176 | 0.025652 | 0.008505 | 2.434853 | 0.020048 | 0.005012 | 0 | 0.000000 | 0.750000 |
| burst_medium | 41 | 5 | 4 | 0.800000 | 0.003902 | 0.079832 | 0.020839 | 4.855046 | 0.083155 | 0.016631 | 0 | 0.000000 | 0.600000 |
| burst_medium | 172 | 3 | 3 | 1.000000 | 0.003463 | 0.134421 | 0.028693 | 7.532137 | 0.074650 | 0.024883 | 0 | 0.000000 | 0.333333 |
| burst_medium | 114 | 3 | 3 | 1.000000 | 0.003724 | 0.113018 | 0.027222 | 6.646134 | 0.069379 | 0.023126 | 1 | 0.017289 | 0.666667 |
| burst_medium | 55 | 3 | 3 | 1.000000 | 0.004139 | 0.084122 | 0.026187 | 5.752069 | 0.064904 | 0.021635 | 0 | 0.000000 | 1.000000 |
| burst_medium | 6 | 3 | 3 | 1.000000 | 0.002227 | 0.166838 | 0.023132 | 9.444821 | 0.062050 | 0.020683 | 1 | 0.021858 | 0.333333 |
| burst_medium | 120 | 4 | 3 | 0.750000 | 0.003883 | 0.077863 | 0.018709 | 4.380029 | 0.058184 | 0.014546 | 3 | 0.031612 | 0.500000 |
| burst_medium | 67 | 4 | 4 | 1.000000 | 0.003464 | 0.079490 | 0.017220 | 4.519827 | 0.053642 | 0.013410 | 0 | 0.000000 | 0.500000 |
| burst_medium | 54 | 3 | 3 | 1.000000 | 0.003593 | 0.099883 | 0.021377 | 5.407903 | 0.052271 | 0.017424 | 1 | 0.017929 | 0.333333 |
| burst_medium | 184 | 2 | 2 | 1.000000 | 0.002236 | 0.114998 | 0.028427 | 11.556514 | 0.051933 | 0.025967 | 2 | 0.051933 | 1.000000 |
| burst_medium | 177 | 3 | 3 | 1.000000 | 0.002664 | 0.112162 | 0.018956 | 6.469538 | 0.048079 | 0.016026 | 3 | 0.048079 | 0.333333 |

## Delta_0 C2 tradeoff
A delta_0 setting is treated as a clean rescue only if C2 passes and overall ADE is no more than 5% worse than the auto-delta0 formal E1 ADE.
| degradation | delta0_label | delta0 | ADE_e1 | ADE_auto_baseline | relative_ADE_change_vs_auto | not_significant_ADE_sacrifice_5pct | ADE_noisy_high | ADE_e1_high | C2_pass | N_high_frames | N_mid_frames | N_low_frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift_medium | fixed_0.02 | 0.020000 | 0.021851 | 0.023272 | -0.061059 | True | 0.003359 | 0.004500 | False | 764 | 2165 | 1071 |
| drift_medium | fixed_0.05 | 0.050000 | 0.018329 | 0.023272 | -0.212406 | True | 0.009278 | 0.009868 | True | 2263 | 1726 | 11 |
| drift_medium | fixed_0.10 | 0.100000 | 0.017600 | 0.023272 | -0.243748 | True | 0.015351 | 0.015493 | True | 3678 | 322 | 0 |
| drift_medium | fixed_0.20 | 0.200000 | 0.017495 | 0.023272 | -0.248240 | True | 0.017648 | 0.017495 | True | 4000 | 0 | 0 |
| drift_medium | auto_median_degraded_error | 0.015678 | 0.023272 | 0.023272 | 0.000000 | True | 0.002408 | 0.003635 | False | 581 | 1815 | 1604 |
| burst_medium | fixed_0.02 | 0.020000 | 0.069872 | 0.073922 | -0.054787 | True | 0.004749 | 0.011309 | False | 707 | 2316 | 977 |
| burst_medium | fixed_0.05 | 0.050000 | 0.062785 | 0.073922 | -0.150665 | True | 0.010042 | 0.012556 | False | 2533 | 672 | 795 |
| burst_medium | fixed_0.10 | 0.100000 | 0.061098 | 0.073922 | -0.173484 | True | 0.012544 | 0.012693 | True | 3191 | 74 | 735 |
| burst_medium | fixed_0.20 | 0.200000 | 0.062245 | 0.073922 | -0.157966 | True | 0.012835 | 0.012484 | True | 3213 | 285 | 502 |
| burst_medium | auto_median_degraded_error | 0.014054 | 0.073922 | 0.073922 | 0.000000 | True | 0.003384 | 0.011275 | False | 351 | 2073 | 1576 |

## Conservative gate ablation
Post-hoc ablation uses lambda_t = (1 - c_t)^gamma with the same oracle confidence and Stage 3 saved conditional output.
| degradation | gamma | ADE_noisy | ADE_cond | ADE_e1 | ADE_noisy_high | ADE_e1_high | ADE_noisy_low | ADE_cond_low | ADE_e1_low | C2_pass | C3_pass | N_high_frames | N_low_frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift_medium | 1 | 0.017648 | 0.031591 | 0.023272 | 0.002408 | 0.003635 | 0.029400 | 0.040013 | 0.036123 | False | True | 581 | 1604 |
| drift_medium | 2 | 0.017648 | 0.031591 | 0.020115 | 0.002408 | 0.002378 | 0.029400 | 0.040013 | 0.033505 | True | True | 581 | 1604 |
| drift_medium | 3 | 0.017648 | 0.031591 | 0.018865 | 0.002408 | 0.002384 | 0.029400 | 0.040013 | 0.031833 | True | True | 581 | 1604 |
| drift_medium | 4 | 0.017648 | 0.031591 | 0.018284 | 0.002408 | 0.002401 | 0.029400 | 0.040013 | 0.030800 | True | True | 581 | 1604 |
| burst_medium | 1 | 0.074755 | 0.091885 | 0.073922 | 0.003384 | 0.011275 | 0.174797 | 0.153864 | 0.148815 | False | True | 351 | 1576 |
| burst_medium | 2 | 0.074755 | 0.091885 | 0.066331 | 0.003384 | 0.004138 | 0.174797 | 0.153864 | 0.145235 | False | True | 351 | 1576 |
| burst_medium | 3 | 0.074755 | 0.091885 | 0.063066 | 0.003384 | 0.003409 | 0.174797 | 0.153864 | 0.142784 | True | True | 351 | 1576 |
| burst_medium | 4 | 0.074755 | 0.091885 | 0.061648 | 0.003384 | 0.003381 | 0.174797 | 0.153864 | 0.141201 | True | True | 351 | 1576 |

## Failure cause judgment
- drift_medium: gate too weak + residual direction wrong + drift global residual leakage. Evidence: gamma=2 passes C2 within 5% ADE of gamma=1; 67.77% of violating high-confidence residuals point opposite the oracle correction; high-confidence frames have tiny noisy error but conditional residual remains nontrivial across the trajectory; delta_0=0.05 passes C2 within 5% ADE
- burst_medium: gate too weak + residual direction wrong + burst boundary effect. Evidence: gamma=3 passes C2 within 5% ADE of gamma=1; 52.80% of violating high-confidence residuals point opposite the oracle correction; 47.80% of violation excess lies within one frame of low-confidence burst frames; delta_0=0.1 passes C2 within 5% ADE

## Figures
- drift_medium trajectory 65: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_failure_diagnosis/figures/drift_medium_failure_traj65.png`
- drift_medium trajectory 156: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_failure_diagnosis/figures/drift_medium_failure_traj156.png`
- drift_medium trajectory 160: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_failure_diagnosis/figures/drift_medium_failure_traj160.png`
- burst_medium trajectory 41: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_failure_diagnosis/figures/burst_medium_failure_traj41.png`
- burst_medium trajectory 172: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_failure_diagnosis/figures/burst_medium_failure_traj172.png`
- burst_medium trajectory 114: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_failure_diagnosis/figures/burst_medium_failure_traj114.png`

## Conclusion
E1's oracle confidence is directionally useful, but the linear gate lambda = 1 - c is too permissive for high-confidence frames. Drift additionally shows global residual leakage: even tiny input errors receive a nontrivial learned residual. Burst shows a boundary effect around corrupted spans, where neighboring high-confidence frames inherit residual corrections from the local burst context. A conservative gamma gate can diagnose this but should be treated as post-hoc evidence, not as a redesigned formal E1 result.
