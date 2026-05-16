# E2-DPS Stage 8A Pilot Audit Failed

Timestamp: 2026-05-15T11:14:15

Failure: Missing Formal E1 per-frame confidence cache

## Details
- The Stage 8A protocol requires c_t_from_E1_cache to be read directly from Formal E1 saved cache/output.
- Searched Formal E1 directory: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions
- No per-frame confidence .npy/.npz cache was found.
- Available Formal E1 files are summary CSV/Markdown files only; they contain bin counts and per-trajectory confidence min/mean/max, not per-frame c_t arrays.
- The requirement forbids recomputing c_t and then comparing array_equal as a workaround.
- An E2-Min trajectory NPZ exists at /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e2_min_absolute_posterior_anchoring/e2_min_optimized_trajectories.npz, but it is not a Formal E1 cache/output and therefore is not accepted for this audit.

## Completed Checks
- Pre-registration file exists: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/docs/stage4/E2_DPS_hypothesis.md
- Required Formal E1 summary files, val_trajs.npy, and Stage 3 checkpoint exist
- Loaded six protocol-validated degraded inputs for first 50 trajectories
- A operator round-trip passed for one representative trajectory per condition: gaussian_medium=0.00e+00, drift_medium=0.00e+00, burst_medium=0.00e+00, bias_medium=0.00e+00, jump_medium=0.00e+00, combined_medium=0.00e+00

## Action
Pilot sampling was not started. Stage 8B was not started. This follows the locked E2-DPS pre-registration and the Stage 8A instruction to stop on path or data problems.
