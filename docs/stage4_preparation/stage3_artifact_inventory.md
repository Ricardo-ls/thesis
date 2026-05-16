# Stage 3 Artifact Inventory

Generated at: `2026-05-14T16:59:23`

Inventory only. No retraining, evaluation rerun, or Stage 3 output modification was performed.

## Phase 1: Repository Structure

| Directory | Status |
|---|---|
| `data/stage3_indoor` | FOUND |
| `outputs/stage3_indoor` | FOUND |
| `tools/stage3_indoor` | FOUND |
| `models` | FOUND |
| `docs` | FOUND |
| `docs/assets` | FOUND |
| `figure3` | MISSING |
| `figure1` | MISSING |

## Phase 2: Stage 3 Data Files

### Candidate NPY Files

| Path | Shape | Dtype | Min | Max | Representation | Likely role | Error |
|---|---:|---|---:|---:|---|---|---|
| `data/stage3_indoor/clean_trajs.npy` | `[2000, 20, 2]` | `float32` | 0.1338 | 2.8649 | absolute coordinates in approx. 3m room | clean source / likely eval subset via [:200] in Stage 3 reports |  |
| `data/stage3_indoor/degraded_bias_medium.npy` | `[1000, 20, 2]` | `float32` | -0.2397 | 3.2211 | absolute coordinates in approx. 3m room | degraded dataset / likely eval candidate |  |
| `data/stage3_indoor/degraded_burst_medium.npy` | `[1000, 20, 2]` | `float32` | -0.7278 | 3.5438 | likely absolute coordinates or normalized trajectory coordinates | degraded dataset / likely eval candidate |  |
| `data/stage3_indoor/degraded_combined_medium.npy` | `[1000, 20, 2]` | `float32` | -0.3468 | 3.3161 | likely absolute coordinates or normalized trajectory coordinates | degraded dataset / likely eval candidate |  |
| `data/stage3_indoor/degraded_drift_medium.npy` | `[1000, 20, 2]` | `float32` | 0.0713 | 2.9842 | absolute coordinates in approx. 3m room | degraded dataset / likely eval candidate |  |
| `data/stage3_indoor/degraded_gaussian_medium.npy` | `[1000, 20, 2]` | `float32` | 0.0154 | 2.9899 | absolute coordinates in approx. 3m room | degraded dataset / likely eval candidate |  |
| `data/stage3_indoor/degraded_jump_medium.npy` | `[1000, 20, 2]` | `float32` | -0.3430 | 3.3354 | likely absolute coordinates or normalized trajectory coordinates | degraded dataset / likely eval candidate |  |
| `data/stage3_indoor/train_trajs.npy` | `[10000, 20, 2]` | `float32` | 0.1319 | 2.8697 | absolute coordinates in approx. 3m room | train |  |
| `data/stage3_indoor/val_trajs.npy` | `[2000, 20, 2]` | `float32` | 0.1343 | 2.8651 | absolute coordinates in approx. 3m room | val |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/eval_degraded_gaussian.npy` | `[200, 20, 2]` | `float32` | 0.0154 | 2.9802 | absolute coordinates in approx. 3m room | val |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_bias.npy` | `[200, 20, 2]` | `float32` | -0.1644 | 3.1437 | absolute coordinates in approx. 3m room | eval |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_burst.npy` | `[200, 20, 2]` | `float32` | -0.6029 | 3.5229 | likely absolute coordinates or normalized trajectory coordinates | eval |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_combined.npy` | `[200, 20, 2]` | `float32` | -0.2533 | 3.2545 | likely absolute coordinates or normalized trajectory coordinates | eval |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_drift.npy` | `[200, 20, 2]` | `float32` | 0.1182 | 2.8825 | absolute coordinates in approx. 3m room | eval |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_gaussian.npy` | `[200, 20, 2]` | `float32` | 0.0154 | 2.9802 | absolute coordinates in approx. 3m room | eval |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_jump.npy` | `[200, 20, 2]` | `float32` | -0.3415 | 3.3443 | likely absolute coordinates or normalized trajectory coordinates | eval |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_full_degraded.npy` | `[200, 20, 2]` | `float32` | 0.0154 | 2.9802 | absolute coordinates in approx. 3m room | eval |  |
| `outputs/stage3_indoor/ddpm_prior/generated_abs.npy` | `[200, 20, 2]` | `float32` | -1.8286 | 5.4417 | unknown | generated prior sample diagnostic |  |
| `outputs/stage3_indoor/ddpm_prior/generated_rel.npy` | `[200, 19, 2]` | `float32` | -1.0008 | 0.9862 | relative displacement (inferred from name/shape) | generated prior sample diagnostic |  |
| `outputs/stage3_indoor/ddpm_prior_diagnostics/generated_abs_check.npy` | `[512, 20, 2]` | `float32` | -2.7851 | 5.2681 | unknown | generated prior sample diagnostic |  |
| `outputs/stage3_indoor/ddpm_prior_diagnostics/generated_rel_check.npy` | `[512, 19, 2]` | `float32` | -1.2167 | 0.9078 | relative displacement (inferred from name/shape) | generated prior sample diagnostic |  |
| `outputs/stage3_indoor/report/cache/bias_cond_residual_t20_refined.npy` | `[200, 20, 2]` | `float32` | -0.1742 | 3.1216 | absolute coordinates in approx. 3m room | cached method output / eval artifact |  |
| `outputs/stage3_indoor/report/cache/burst_cond_residual_t20_refined.npy` | `[200, 20, 2]` | `float32` | -0.4128 | 3.3676 | likely absolute coordinates or normalized trajectory coordinates | cached method output / eval artifact |  |
| `outputs/stage3_indoor/report/cache/drift_cond_residual_t20_refined.npy` | `[200, 20, 2]` | `float32` | 0.0724 | 2.9085 | absolute coordinates in approx. 3m room | cached method output / eval artifact |  |
| `outputs/stage3_indoor/report/cache/gaussian_cond_residual_t20_refined.npy` | `[200, 20, 2]` | `float32` | 0.0483 | 3.0010 | absolute coordinates in approx. 3m room | cached method output / eval artifact |  |
| `outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t2_refined.npy` | `[200, 20, 2]` | `float32` | 0.0333 | 2.9736 | absolute coordinates in approx. 3m room | cached method output / eval artifact |  |

### Data Split / Config Files

| Path | Type | Size bytes |
|---|---|---:|
| `data/stage3_indoor/degradation_config.json` | `.json` | 1352 |
| `data/stage3_indoor/trajs_metadata.json` | `.json` | 187022 |

### Normalization Files

| Path | Keys | Error |
|---|---|---|
| `data/stage3_indoor/rel_norm_params.npz` | `rel_mean, rel_std, rel_std_safe` |  |
| `data/stage3_indoor/rel_norm_params_v2.npz` | `rel_mean, rel_std` |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/rel_norm_params_v2.npz` | `rel_mean, rel_std` |  |
| `outputs/stage3_indoor/ddpm_indoor/seed42/rel_norm_params.npz` | `rel_mean, rel_std, rel_std_safe` |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/rel_norm_params_v2.npz` | `rel_mean, rel_std` |  |
| `outputs/stage3_indoor/receptive_field_expansion/rel_norm_params_v2.npz` | `rel_mean, rel_std` |  |

## Phase 3: Stage 3 Model Checkpoints

| Path | Role | Size MB | Modified | Keys | model_state_dict | ema_state_dict | epoch | val_loss | Load error |
|---|---|---:|---|---|---|---|---:|---:|---|
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/best_ema_model.pt` | conditional residual DDPM | 0.815 | 2026-05-01T14:57:28 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/best_model.pt` | conditional residual DDPM | 0.815 | 2026-05-01T14:57:28 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/final_ema_model.pt` | conditional residual DDPM | 0.815 | 2026-05-01T14:57:28 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/final_model.pt` | conditional residual DDPM | 0.815 | 2026-05-01T14:57:28 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/ddpm_indoor/seed42/best_model.pt` | indoor prior | 0.513 | 2026-04-30T22:07:10 | `time_mlp.0.weight, time_mlp.0.bias, time_mlp.2.weight, time_mlp.2.bias, conv1.weight, conv1.bias, conv2.weight, conv2.bias` | False | False | None | None |  |
| `outputs/stage3_indoor/ddpm_indoor/seed42/final_model.pt` | indoor prior | 0.513 | 2026-04-30T22:07:38 | `time_mlp.0.weight, time_mlp.0.bias, time_mlp.2.weight, time_mlp.2.bias, conv1.weight, conv1.bias, conv2.weight, conv2.bias` | False | False | None | None |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt` | indoor prior | 0.812 | 2026-05-01T08:56:43 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_model.pt` | indoor prior | 0.812 | 2026-05-01T08:56:43 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/final_ema_model.pt` | indoor prior | 0.812 | 2026-05-01T08:56:43 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/final_model.pt` | indoor prior | 0.812 | 2026-05-01T08:56:43 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/ddpm_prior/last_model.pt` | indoor prior | 0.513 | 2026-04-30T20:36:11 | `time_mlp.0.weight, time_mlp.0.bias, time_mlp.2.weight, time_mlp.2.bias, conv1.weight, conv1.bias, conv2.weight, conv2.bias` | False | False | None | None |  |
| `outputs/stage3_indoor/ddpm_prior/val_selected_model.pt` | indoor prior | 0.513 | 2026-04-30T20:36:11 | `time_mlp.0.weight, time_mlp.0.bias, time_mlp.2.weight, time_mlp.2.bias, conv1.weight, conv1.bias, conv2.weight, conv2.bias` | False | False | None | None |  |
| `outputs/stage3_indoor/receptive_field_expansion/ckpt_4block_best.pt` | 4-block RF expansion | 1.569 | 2026-05-06T12:49:13 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/receptive_field_expansion/ckpt_4block_best_ema.pt` | 4-block RF expansion | 1.57 | 2026-05-06T12:49:13 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/receptive_field_expansion/ckpt_4block_final.pt` | 4-block RF expansion | 1.569 | 2026-05-06T12:49:13 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |
| `outputs/stage3_indoor/receptive_field_expansion/ckpt_4block_final_ema.pt` | 4-block RF expansion | 1.57 | 2026-05-06T12:49:13 | `time_emb.embedding.weight, input_proj.weight, input_proj.bias, block1.0.weight, block1.0.bias, block1.2.weight, block1.2.bias, block2.0.weight` | False | False | None | None |  |

## Phase 4: Stage 3 Result CSVs

| Path | Rows | Columns | degradation | method | ADE/ade | seed | trajectory_id | Expected key CSV | Error |
|---|---:|---|---|---|---|---|---|---|---|
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/cond_residual_gaussian_per_traj.csv` | 1200 | `traj_idx, method, ADE, RMSE, smooth, noisy_ADE, uncond_ADE, delta_ADE_vs_noisy, delta_ADE_vs_uncond, improved_vs_noisy, improved_vs_uncond` | False | True | True | False | False | False |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/cond_residual_gaussian_summary.csv` | 6 | `method, N, ADE_mean, ADE_std, RMSE_mean, RMSE_std, smooth_mean, smooth_std, delta_ADE_vs_noisy_mean, improved_fraction_vs_noisy, wilcoxon_p_vs_noisy, delta_ADE_vs_uncond_mean, improved_fraction_vs_uncond, wilcoxon_p_vs_uncond` | False | True | True | False | False | False |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_per_traj.csv` | 3600 | `degradation, degradation_group, traj_idx, method, ADE, smooth, noisy_ADE, noisy_smooth, delta_ADE_vs_noisy, delta_smooth_vs_noisy, improved_ADE_vs_noisy, improved_smooth_vs_noisy` | True | True | True | False | False | True |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_summary.csv` | 18 | `degradation, degradation_group, method, N, ADE_mean, ADE_std, smooth_mean, delta_ADE_vs_noisy_mean, improved_fraction, wilcoxon_p_vs_noisy` | True | True | True | False | False | True |  |
| `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/loss_curve.csv` | 60 | `epoch, train_loss, val_loss_ema` | False | False | False | False | False | False |  |
| `outputs/stage3_indoor/ddpm_indoor/seed42/loss_curve.csv` | 100 | `epoch, train_loss, val_loss` | False | False | False | False | False | False |  |
| `outputs/stage3_indoor/ddpm_indoor/seed42/oor_diagnosis_summary.csv` | 4 | `strategy, oor_ratio, oor_mean_dev, oor_max_dev, endpoint_drift_mean, endpoint_drift_p95` | False | False | False | False | False | False |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/loss_curve.csv` | 60 | `epoch, train_loss, val_loss_ema` | False | False | False | False | False | False |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_per_traj.csv` | 1200 | `degradation, traj_idx, method, t_start, ADE, RMSE, smooth, noisy_ADE, noisy_RMSE, noisy_smooth, kalman_ADE, kalman_RMSE, kalman_smooth, delta_ADE_vs_noisy, delta_RMSE_vs_noisy, delta_smooth_vs_noisy, delta_ADE_vs_kalman, delta_RMSE_vs_kalman, delta_smooth_vs_kalman, improved_ADE_vs_noisy, improved_RMSE_vs_noisy, improved_smooth_vs_noisy, improved_ADE_vs_kalman, improved_RMSE_vs_kalman, improved_smooth_vs_kalman` | True | True | True | False | False | False |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_summary.csv` | 6 | `degradation, method, t_start, N, ADE_mean, ADE_std, ADE_median, ADE_p25, ADE_p75, RMSE_mean, RMSE_std, smooth_mean, smooth_std, ADE_vs_noisy_percent, delta_ADE_mean, delta_ADE_median, delta_ADE_std, improved_fraction_ADE, wilcoxon_p_vs_noisy, delta_RMSE_mean, improved_fraction_RMSE, delta_smooth_mean, smooth_improved_fraction, ADE_vs_kalman_percent, delta_ADE_vs_kalman_mean, delta_ADE_vs_kalman_median, improved_fraction_ADE_vs_kalman, wilcoxon_p_vs_kalman` | True | True | True | False | False | False |  |
| `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_scout_results.csv` | 10 | `degradation, method, t_start, N, ADE_mean, ADE_std, smooth_mean, ADE_vs_noisy_percent, wilcoxon_p_vs_noisy` | True | True | True | False | False | False |  |
| `outputs/stage3_indoor/ddpm_prior/loss_curve.csv` | 100 | `epoch, train_loss, val_loss` | False | False | False | False | False | False |  |
| `outputs/stage3_indoor/ddpm_prior_diagnostics/one_step_denoise.csv` | 6 | `t, noisy_error, denoised_error, improvement, improvement_pct` | False | False | False | False | False | False |  |
| `outputs/stage3_indoor/ddpm_prior_diagnostics/overfit_loss_curve.csv` | 10 | `iteration, loss` | False | False | False | False | False | False |  |
| `outputs/stage3_indoor/receptive_field_expansion/ablation_results.csv` | 18 | `method, degradation, ADE_mean, ADE_std, ADE_median, RMSE_mean, smooth_mean, OOR_rate, n_traj, n_seed, source, paired_wilcoxon_p_vs_current_2block` | True | True | True | False | False | True |  |
| `outputs/stage3_indoor/receptive_field_expansion/raw_4block_eval.csv` | 18000 | `trajectory_id, seed, degradation, method, ADE, RMSE, smooth, OOR` | True | True | True | True | True | True |  |
| `outputs/stage3_indoor/receptive_field_expansion/train_4block_log.csv` | 60 | `epoch, train_loss, val_loss, elapsed_seconds, is_best` | False | False | False | False | False | True |  |
| `outputs/stage3_indoor/report/tables/table1_gaussian_medium.csv` | 6 | `method, ADE_mean, ADE_std, delta_vs_noisy_pct, improved_fraction, wilcoxon_p, smooth_mean` | False | True | True | False | False | False |  |
| `outputs/stage3_indoor/report/tables/table2_generalization.csv` | 6 | `degradation, noisy_input_ADE, kalman_cv_ADE, uncond_sdedit_ADE, cond_residual_ADE, cond_delta_vs_noisy, cond_p_vs_noisy, interpretation` | True | False | True | False | False | False |  |
| `outputs/stage3_indoor/report/tables/table3_glossary.csv` | 11 | `term, definition` | False | False | False | False | False | False |  |
| `outputs/stage3_indoor/sdedit_diagnostic/diagnostic_per_traj.csv` | 12000 | `degradation, method, t_start, traj_index, ADE, RMSE, smooth, step_mean, step_p95, step_max, acc_mean, acc_p95, out_of_room_ratio` | True | True | True | False | False | False |  |
| `outputs/stage3_indoor/sdedit_diagnostic/diagnostic_summary.csv` | 12 | `degradation, method, t_start, ADE_mean, ADE_std, ADE_median, ADE_p25, ADE_p75, RMSE_mean, smooth_mean, step_mean, step_p95_mean, step_max_mean, acc_mean, acc_p95_mean, out_of_room_ratio_mean, n_traj` | True | True | True | False | False | False |  |

## Explicitly Missing Expected Artifacts

- Missing directory: `figure3`
- Missing directory: `figure1`
- Missing explicit file named `eval_trajs.npy`; Stage 3 appears to use `clean_trajs.npy[:200]` / `val_trajs.npy` and generated degraded arrays instead.
