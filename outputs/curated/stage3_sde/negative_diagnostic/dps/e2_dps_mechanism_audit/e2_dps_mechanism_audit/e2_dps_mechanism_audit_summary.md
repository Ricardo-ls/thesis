# E2-DPS Mechanism Audit Summary

This is a mechanism audit only. Stage 8B was not started, zeta was not expanded, and the method was not changed.

Pre-registration: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/docs/stage4/E2_DPS_hypothesis.md
Amendment 001: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/docs/stage4/E2_DPS_hypothesis_amendment_001.md
Amendment 002: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/docs/stage4/E2_DPS_hypothesis_amendment_002.md

## A. Guidance Variable Audit
- current implementation applies measurement loss to: x0_hat_abs_from_xt(x_t, eps_pred, t), the denoised absolute trajectory estimate derived from the current reverse latent x_t
- the DDPM stochastic base step is computed as `x_base = ddpm_step(x_t, eps_pred, t)`.
- the guidance update is then applied to `x_base`: `x_guided = x_base - zeta * grad`.
- therefore the gradient is computed through the denoised estimate from the pre-step latent, then applied after the DDPM base step.

## B. DPS Canonical Interface Audit
- denoised x0_hat is available: True
- x0_hat shape: (1, 20, 2) absolute trajectory per selected case
- x0_hat scale: absolute coordinate scale after adding normalized residual displacement to degraded relative displacement and integrating from y[0]
- x0_hat can be converted to absolute trajectory by the existing A/operator reconstruction inside `x0_hat_abs_from_xt`.

## C. Step-Level Likelihood vs ADE Audit
- fraction of selected guided steps where likelihood_norm decreases after guidance: 0.599
- fraction of selected guided steps where ADE increases after guidance: 0.512
- fraction of selected guided steps where acceleration RMS increases after guidance: 0.617

## D. zeta=0 Base Sampler Audit
- zeta=0 no-guidance mean final ADE over selected cases: 1.534886
- zeta=0 no-guidance mean acceleration RMS over selected cases: 0.282327

## E. Monotonicity Audit
| zeta | final_ADE | final_acceleration_RMS | final_likelihood_norm | motion_usage_ratio |
| --- | --- | --- | --- | --- |
| 0.000000 | 1.534886 | 0.282327 | 8.652581 | 2.645523 |
| 0.300000 | 2.090522 | 0.334204 | 121.858727 | 7.230235 |
| 1.000000 | 5.763478 | 0.807710 | 402.766174 | 25.282640 |
| 3.000000 | 15.825772 | 2.143804 | 1222.245076 | 78.384595 |

- zeta increases monotonically worsen ADE: True
- zeta increases monotonically lower likelihood: False

## Failure Attribution
- classification: objective mismatch + over-guidance scale problem
- objective mismatch: guidance often lowers measurement likelihood while increasing ADE
- over-guidance / roughening: guidance often increases acceleration RMS
- over-guidance scale problem: ADE and motion usage increase monotonically with zeta

## Stage 8B Decision
Stage 8B should not be started from this audited implementation as a positive-signal continuation. The Stage 8A result is finite but mechanism-negative.

Recommended next step: write a new amendment before any further run. Based on this audit, the first amendment to consider is not zeta expansion; it is a variable-interface / guidance-timing correction that computes and applies guidance consistently with the denoised x0_hat interface, or a pre-registered timestep schedule if the interface is judged correct.

Trust-region or clipping should only be considered after the variable-interface audit is resolved, because they would hide rather than explain the current harmful guidance mechanism.

## Figures
- drift_medium traj 0: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_mechanism_audit/figures/drift_medium_traj0_mechanism_trace.png
- burst_medium traj 0: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_mechanism_audit/figures/burst_medium_traj0_mechanism_trace.png
- gaussian_medium traj 0: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_dps_mechanism_audit/figures/gaussian_medium_traj0_mechanism_trace.png
