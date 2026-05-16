# Formal E1 Confidence Cache Validation Summary

Formal E1 did not originally save per-frame c_t arrays. This cache is reconstructed from the locked Formal E1 formula and validated against Formal E1 confidence-bin/statistical summaries. It is now the audited c_t source for E2-DPS.

Cache type: validated reconstructed Formal E1 c_t cache.

Formula:

`c_t = exp(- ||y_t - x_star_t|| / delta_0)`

Bins:

- high: c_t > 0.7
- low: c_t < 0.3
- mid: 0.3 <= c_t <= 0.7

## Validation Table
| degradation | shape | delta0 | n_high_conf_frames | n_mid_conf_frames | n_low_conf_frames | bin_counts_match | per_traj_confidence_stats_match | range_valid | validation_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | (200, 20) | 0.058828253 | 334 | 2177 | 1489 | True | True | True | True |
| drift_medium | (200, 20) | 0.015678257 | 581 | 1815 | 1604 | True | True | True | True |
| burst_medium | (200, 20) | 0.014054442 | 351 | 2073 | 1576 | True | True | True | True |
| bias_medium | (200, 20) | 0.172925338 | 220 | 2220 | 1560 | True | True | True | True |
| jump_medium | (200, 20) | 0.302824795 | 986 | 1675 | 1339 | True | True | True | True |
| combined_medium | (200, 20) | 0.188137785 | 290 | 2257 | 1453 | True | True | True | True |

## Output Files
- gaussian_medium: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/gaussian_medium_confidence.npy ; /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/gaussian_medium_confidence_validation.json
- drift_medium: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/drift_medium_confidence.npy ; /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/drift_medium_confidence_validation.json
- burst_medium: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/burst_medium_confidence.npy ; /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/burst_medium_confidence_validation.json
- bias_medium: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/bias_medium_confidence.npy ; /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/bias_medium_confidence_validation.json
- jump_medium: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/jump_medium_confidence.npy ; /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/jump_medium_confidence_validation.json
- combined_medium: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/combined_medium_confidence.npy ; /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache/combined_medium_confidence_validation.json
