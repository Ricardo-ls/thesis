# E2-DPS Stage 8A Pilot Audit Summary

Pre-registration: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/docs/stage4/E2_DPS_hypothesis.md
Amendment 001: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/docs/stage4/E2_DPS_hypothesis_amendment_001.md
Amendment 002: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/docs/stage4/E2_DPS_hypothesis_amendment_002.md
Confidence cache: /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_oracle_residual_gating_6conditions/confidence_cache
Device: cpu

All audit checks passed before pilot sampling.

Method: Amendment 002 canonical norm-based DPS guidance with heteroscedastic confidence weighting.

## A Operator Round-Trip
- gaussian_medium: max_err=0.00e+00
- drift_medium: max_err=0.00e+00
- burst_medium: max_err=0.00e+00
- bias_medium: max_err=0.00e+00
- jump_medium: max_err=0.00e+00
- combined_medium: max_err=0.00e+00

## Confidence Cache Counts
- gaussian_medium: full high/mid/low=334/2177/1489; pilot high/mid/low=82/580/338
- drift_medium: full high/mid/low=581/1815/1604; pilot high/mid/low=171/461/368
- burst_medium: full high/mid/low=351/2073/1576; pilot high/mid/low=87/545/368
- bias_medium: full high/mid/low=220/2220/1560; pilot high/mid/low=20/700/280
- jump_medium: full high/mid/low=986/1675/1339; pilot high/mid/low=251/429/320
- combined_medium: full high/mid/low=290/2257/1453; pilot high/mid/low=64/606/330

## Sigma Obs Squared
- gaussian_medium: min/mean/max=0.002617/0.003981/0.004899
- drift_medium: min/mean/max=0.002500/0.003928/0.004938
- burst_medium: min/mean/max=0.002527/0.004091/0.005000
- bias_medium: min/mean/max=0.003109/0.003981/0.004829
- jump_medium: min/mean/max=0.002500/0.003763/0.004520
- combined_medium: min/mean/max=0.002726/0.003992/0.004750

## Norm-Based Likelihood Sign Check
- drift_medium: loss before=3.174130, after=3.172921, passed=True

Ground truth val_trajs.npy[:50] is used only for metrics, not guidance, anchor, or sigma_obs.
