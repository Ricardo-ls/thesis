# Reconstructed Missing Stage 3 Conditional Outputs

These files were reconstructed after Stage 3 using the original Stage 3 evaluation inputs and checkpoint. They are not original Stage 3 saved artifacts.

- checkpoint: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/best_ema_model.pt`
- normalization: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/rel_norm_params_v2.npz`
- clean eval: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/data/stage3_indoor/clean_trajs.npy[:200]`
- t_start: 20
- seeds: [42, 43, 44, 45, 46]

| degradation | output | shape | computed ADE | Stage 3 ADE | abs diff | match |
| --- | --- | --- | ---: | ---: | ---: | --- |
| jump_medium | `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/reconstructed_stage3_cond_outputs/jump_cond_residual_t20_refined_reconstructed.npy` | (200, 20, 2) | 0.246522859 | 0.246522859 | 0.000e+00 | True |
| combined_medium | `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/reconstructed_stage3_cond_outputs/combined_cond_residual_t20_refined_reconstructed.npy` | (200, 20, 2) | 0.197429359 | 0.197429359 | 0.000e+00 | True |
