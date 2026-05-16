# Stage 4 E1 Oracle Residual-Gating Smoke Test

## Setup
- checkpoint path: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/best_ema_model.pt`
- normalization path: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/rel_norm_params_v2.npz`
- data source: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/data/stage3_indoor/clean_trajs.npy[:10]`
- original degraded path: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/eval_degraded_gaussian.npy`
- original Stage 3 conditional output path: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3_indoor/report/cache/gaussian_cond_residual_t20_refined.npy`
- degradation: `gaussian_medium`
- number of trajectories: 10
- seed: 42
- T: 20
- conditional residual start_t: 20
- conditional residual source: cached Stage 3 cond_residual_t20 output
- Stage 3 cached conditional output was generated as the mean over seeds [42, 43, 44, 45, 46]

## Stage 3 output verification
- conditional output shape type: aggregated_prediction
- recomputed cond_residual_t20 ADE mean: 0.055695
- Stage 3 table cond_residual_t20 ADE mean: 0.055695
- recomputed cond_residual_t20 ADE std: 0.016248
- Stage 3 table cond_residual_t20 ADE std: 0.016248
- cond ADE mean absolute diff: 0.000e+00
- recomputed noisy ADE mean: 0.062567
- Stage 3 table noisy ADE mean: 0.062567
- reproduced Stage 3 metrics: TRUE

## Oracle confidence
- delta_0: 0.052043
- confidence min / mean / max: 0.048153 / 0.370073 / 0.919539
- correlation between confidence and per-frame delta_t: -0.945280

Expected: confidence should be negatively correlated with delta_t.

## Mean metrics over 10 trajectories
- mean ADE_noisy: 0.059325
- mean ADE_cond: 0.056626
- mean ADE_e1: 0.046121
- mean ADE_high_noisy: 0.013403
- mean ADE_high_cond: 0.033257
- mean ADE_high_e1: 0.012165
- mean ADE_low_noisy: 0.091977
- mean ADE_low_cond: 0.065692
- mean ADE_low_e1: 0.064202

## Smoke conclusion
- E1 oracle residual gating reduces ADE vs noisy_input by -22.26%.
- E1 oracle residual gating reduces ADE vs Stage 3 cond_residual_t20 by -18.55%.
- On high-confidence frames, Stage 3 cond_residual_t20 changes ADE vs noisy_input by +148.13%, showing the over-correction failure mode.
- On high-confidence frames, oracle gating changes ADE vs noisy_input by -9.24%, showing that suppressing residuals on reliable observations works in this smoke setting.
- On low-confidence frames, oracle gating keeps the useful residual behavior: Stage 3 cond_residual_t20 is -28.58% vs noisy, and E1 is -30.20% vs noisy.
- Conclusion: PASS. This supports Stage 4 E1's core mechanism, but it is still only an oracle smoke test, not a final scientific result.

## Smoke decision
- PASS: TRUE

Checks:
- script runs end-to-end: TRUE
- output shapes are valid: TRUE
- no NaN/Inf in full trajectories: TRUE
- confidence is negatively correlated with delta_t: TRUE
- x_e1 shape matches y and x_cond: TRUE
- ADE_e1 is finite: TRUE

Do not judge full scientific success from this smoke test.

## Saved artifacts
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_smoke/smoke_metrics.csv`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_smoke/smoke_examples.npz`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_smoke/smoke_examples.png`
