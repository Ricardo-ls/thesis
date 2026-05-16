# E2-DPS NaN Diagnosis Summary

Pre-registration: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/docs/stage4/E2_DPS_hypothesis.md

## Main Findings
- First non-finite observed at condition=drift_medium, zeta=3.0, trajectory=0, step=93, stage=likelihood_loss.
- zeta=0 no-guidance finite for all checked cases: True
- guided finite for all checked cases: False
- max grad_norm observed: 5.02611e+17
- max update_to_x_ratio observed: 24775.3
- median t=99 loss_sum / loss_mean ratio: 20
- first_nonfinite_stage counts: {'likelihood_loss': 12}
- diagnosis: guidance update scale / pre-registered hyperparameter instability
- recommendation: Do not continue Stage 8A as interpretable. If changing normalization, scaling, or clipping is needed, write a pre-registration amendment or audit finding first.

## Interpretation
This diagnostic does not modify the formal E2-DPS method. No clipping, robust likelihood, altered sigma, altered kappa, or expanded zeta was introduced.

If the diagnosis points to scale mismatch, Stage 8A should not be treated as a PASS/FAIL scientific result. Any method change such as loss normalization, guidance scaling, or clipping requires a pre-registration amendment / audit finding before rerunning.

## Figures
- drift_medium traj 0 zeta=0.3: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_nan_diagnosis/figures/drift_medium_traj0_zeta0p3_trace.png
- burst_medium traj 0 zeta=0.3: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_nan_diagnosis/figures/burst_medium_traj0_zeta0p3_trace.png
- gaussian_medium traj 0 zeta=0.3: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_nan_diagnosis/figures/gaussian_medium_traj0_zeta0p3_trace.png
- bias_medium traj 0 zeta=0.3: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_nan_diagnosis/figures/bias_medium_traj0_zeta0p3_trace.png
