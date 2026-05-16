# Bias Prior-Signal Sanity Check

## Data
This diagnostic reads existing Stage 3 protocol-validated per-frame conditional outputs only. It does not train, resample, run SDEdit, or modify Stage 3 files.

| degradation | conditional_source | conditional_path | degraded_path | clean_path | shape_clean | shape_degraded | shape_cond |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bias_medium | archived_original | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/bias_cond_residual_t20_refined.npy | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_bias.npy | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/data/stage3_indoor/clean_trajs.npy | (200, 20, 2) | (200, 20, 2) | (200, 20, 2) |

## Key Metrics
| metric | value |
| --- | --- |
| ADE_noisy | 0.188996 |
| ADE_cond | 0.188868 |
| ADE_improvement_noisy_minus_cond | 0.000127 |
| ADE_relative_improvement | 0.000674 |
| offset_norm_noisy_mean | 0.188996 |
| offset_norm_cond_mean | 0.188100 |
| offset_norm_improvement | 0.000895 |
| offset_norm_relative_improvement | 0.004738 |
| offset_reduced_fraction_trajectories | 0.520000 |
| shape_centered_ADE_noisy_mean | 0.000000 |
| shape_centered_ADE_cond_mean | 0.016599 |
| shape_centered_ADE_improvement | -0.016599 |
| frame_cosine_mean | 0.075298 |
| frame_cosine_median | 0.135319 |
| frame_cosine_p25 | -0.613285 |
| frame_cosine_p75 | 0.787928 |
| frame_cosine_fraction_gt_0 | 0.519750 |
| frame_cosine_fraction_gt_0p5 | 0.356000 |
| frame_cosine_fraction_lt_0 | 0.440000 |
| global_cosine_mean | 0.088136 |
| global_cosine_median | 0.166198 |
| global_ratio_mean | 0.166282 |
| global_ratio_median | 0.112625 |

## Direct Answers
1. Stage 3 cond_residual_t20 improves ADE on bias_medium: yes; ADE_noisy=0.188996, ADE_cond=0.188868.
2. It reduces absolute offset error: yes, but weakly; offset_noisy=0.188996, offset_cond=0.188100.
3. Residual direction alignment is very weak or absent bias-correcting signal: frame cosine mean=0.075298, median=0.135319; global cosine mean=0.088136, median=0.166198.
4. Correction source: some global offset correction.
5. E2-DPS prior-signal judgment: very weak or absent bias-correcting signal.
6. H5 recommendation: downgrade E2-DPS H5 to known limitation / future work unless DPS adds a stronger absolute-position likelihood.

## Offset-Reduction Cases
| trajectory_id | ADE_noisy | ADE_cond | offset_norm_noisy | offset_norm_cond | offset_norm_improvement | global_correction_cosine | global_correction_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | 0.283208 | 0.228835 | 0.283208 | 0.228769 | 0.054439 | 0.984759 | 0.195927 |
| 83 | 0.554288 | 0.509450 | 0.554288 | 0.509067 | 0.045222 | 0.926657 | 0.088695 |
| 88 | 0.146181 | 0.104351 | 0.146181 | 0.103317 | 0.042864 | 0.944659 | 0.318633 |
| 52 | 0.170289 | 0.132453 | 0.170289 | 0.131787 | 0.038502 | 0.939110 | 0.245675 |
| 103 | 0.215573 | 0.178816 | 0.215573 | 0.178605 | 0.036968 | 0.926738 | 0.188310 |

## Figures
- best_ADE_improvement trajectory 123: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_prior_signal_sanity_bias/figures/best_ADE_improvement_traj123.png
- worst_offset_regression trajectory 154: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_prior_signal_sanity_bias/figures/worst_offset_regression_traj154.png
- highest_global_alignment trajectory 7: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_prior_signal_sanity_bias/figures/highest_global_alignment_traj7.png
- median_global_alignment trajectory 62: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_prior_signal_sanity_bias/figures/median_global_alignment_traj62.png
