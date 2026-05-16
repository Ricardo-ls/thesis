# Stage 3 SDE / SDEdit Legacy Audit

Date: 2026-05-15

Scope: static audit only. No experiment was run. No model was trained. No checkpoint or Stage 3 output was modified.

## Executive Answer

Yes, the old Stage 3 indoor unconditional-prior SDEdit line exists.

The strongest relevant record is:

- prior: `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`
- normalization: `data/stage3_indoor/rel_norm_params_v2.npz`
- main script: `tools/stage3_indoor/sdedit_gaussian_full.py`
- six-condition comparison script: `tools/stage3_indoor/generalization_diagnostic.py`
- main result table: `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_summary.csv`
- report table: `outputs/stage3_indoor/report/tables/table2_generalization.csv`

It was observation-initialized SDEdit-style DDPM refinement:

```text
degraded absolute trajectory y_abs
-> relative displacement dy
-> normalize dy
-> q_sample(dy, t_start)
-> unconditional reverse DDPM denoise
-> denormalize relative displacement
-> reconstruct absolute trajectory anchored at y_abs[0]
```

It did not use confidence. It did not use the conditional residual model. It was prior-only refinement.

The remembered “有改进但有限” is accurate for the final indoor-v2 SDEdit result: under `gaussian_medium`, best `t_start=2` improved ADE from `0.062567` to `0.060812`, a `2.804624%` reduction. But across six degradations, the effect was limited and unstable: it improved `gaussian_medium` and `jump_medium`, worsened `drift_medium` and `burst_medium`, and was almost unchanged for `bias_medium` / `combined_medium`.

## Q1. 3x3 Room / 6-Condition Data

### Found Data

The 3m x 3m indoor simulator data exists:

| file | shape | dtype | min | max | role |
|---|---:|---|---:|---:|---|
| `data/stage3_indoor/clean_trajs.npy` | `(2000, 20, 2)` | `float32` | `0.133755` | `2.864851` | clean trajectories |
| `data/stage3_indoor/train_trajs.npy` | `(10000, 20, 2)` | `float32` | `0.131867` | `2.869740` | prior training source |
| `data/stage3_indoor/val_trajs.npy` | `(2000, 20, 2)` | `float32` | `0.134277` | `2.865081` | prior validation / later Stage 4 source |
| `data/stage3_indoor/degraded_gaussian_medium.npy` | `(1000, 20, 2)` | `float32` | `0.015415` | `2.989899` | degraded data |
| `data/stage3_indoor/degraded_bias_medium.npy` | `(1000, 20, 2)` | `float32` | `-0.239748` | `3.221137` | degraded data |
| `data/stage3_indoor/degraded_drift_medium.npy` | `(1000, 20, 2)` | `float32` | `0.071285` | `2.984233` | degraded data |
| `data/stage3_indoor/degraded_jump_medium.npy` | `(1000, 20, 2)` | `float32` | `-0.342967` | `3.335413` | degraded data |
| `data/stage3_indoor/degraded_burst_medium.npy` | `(1000, 20, 2)` | `float32` | `-0.727769` | `3.543796` | degraded data |
| `data/stage3_indoor/degraded_combined_medium.npy` | `(1000, 20, 2)` | `float32` | `-0.346843` | `3.316102` | degraded data |

The six found categories are degradation conditions, not six path classes:

- `gaussian_medium`
- `bias_medium`
- `drift_medium`
- `jump_medium`
- `burst_medium`
- `combined_medium`

The generator `tools/stage3_indoor/generate_indoor_trajs.py` defines a 3m x 3m room with five behavior types:

- `goal_directed`
- `multi_goal`
- `pacing`
- `stationary`
- `boundary_walk`

So the precise audit wording is: the 3x3 room data exists, and the “six” appears in the saved degradation/condition set. I did not find evidence that the indoor simulator uses exactly six path behavior classes.

## Q2. Old Indoor Prior

### Main Indoor Unconditional Prior

Found:

- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_model.pt`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/final_ema_model.pt`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/final_model.pt`

Training script:

- `tools/stage3_indoor/train_indoor_ddpm_v2.py`

Prior representation:

- normalized relative displacement
- storage before model shape: `(N, 19, 2)`
- model input shape: `[B, 2, 19]`
- model: `TemporalDenoiser1D(max_timesteps=100, in_channels=2, hidden_dim=128)`
- timesteps: `100`
- training data: `train_trajs.npy`, expanded by 6-fold geometric augmentation to `(60000, 20, 2)`

Normalization:

- `data/stage3_indoor/rel_norm_params_v2.npz`
- `rel_mean = [3.9385077e-09, -1.5433420e-09]`
- `rel_std = [0.1478053, 0.14835621]`

Prior check:

- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/prior_check_v2.json`
- `best_epoch = 60`
- `best_val_loss = 0.26114753794670104`
- `passed = true`

This is different from the Stage 2 ETH+UCY prior. The Stage 2 prior under `outputs/prior/train/...` is ETH+UCY relative displacement without Stage 3 indoor normalization; the indoor-v2 prior is trained on synthetic indoor trajectories and uses `rel_norm_params_v2.npz`.

### Older Indoor Prior

Also found:

- `outputs/stage3_indoor/ddpm_prior/val_selected_model.pt`
- `outputs/stage3_indoor/ddpm_prior/last_model.pt`
- `outputs/stage3_indoor/ddpm_indoor/seed42/best_model.pt`
- `outputs/stage3_indoor/ddpm_indoor/seed42/final_model.pt`

The older SDEdit diagnostic used `outputs/stage3_indoor/ddpm_prior/val_selected_model.pt`. The final cleaner gaussian/generalization line used the v2 prior.

## Q3. What the SDE / SDEdit Experiment Was

### Final Indoor-v2 SDEdit

Script:

- `tools/stage3_indoor/sdedit_gaussian_full.py`

Protocol:

- `N = 200`
- `TIMESTEPS = 100`
- `T_LIST = [1, 2, 3, 5]`
- `SDEDIT_SEEDS = [42, 43, 44, 45, 46]`
- degradation: `gaussian_medium`
- checkpoint: `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`
- normalization: `data/stage3_indoor/rel_norm_params_v2.npz`

Interface:

```text
degraded_abs[:, 1:, :] - degraded_abs[:, :-1, :]
-> normalize with rel_mean / rel_std
-> transpose to [B, 2, 19]
-> q_sample(x0, t_start)
-> reverse DDPM from t_start to 0
-> denormalize
-> reconstruct absolute trajectory with start point degraded_abs[:, 0, :]
```

So yes: this was observation-initialized SDEdit. It was not pure generation from Gaussian noise.

It did not use:

- confidence
- DPS posterior guidance
- conditional residual model
- clean target during inference
- oracle `t_start`

### Six-Condition Unconditional SDEdit Diagnostic

Script:

- `tools/stage3_indoor/generalization_diagnostic.py`

Protocol:

- `N = 200`
- `SAMPLE_SEEDS = [42, 43, 44, 45, 46]`
- `uncond_sdedit_t2` only
- six conditions: gaussian, drift, jump, burst, bias, combined
- method output is seed-averaged absolute trajectory

This is the record that compares prior-only SDEdit with the final conditional residual DDPM.

### Older Diagnostic

Script:

- `tools/stage3_indoor/diagnose_sdedit.py`

Protocol:

- `N_EVAL = 1000`
- degradations: `gaussian_medium`, `burst_medium`
- `t_start = [1, 3, 5, 10, 20]`
- checkpoint: `outputs/stage3_indoor/ddpm_prior/val_selected_model.pt`
- input representation: relative displacement
- absolute reconstruction start: `degraded_abs[:,0,:]`

This older run showed severe drift as `t_start` increased.

## Q4. Result Locations

Key scripts:

- `tools/stage3_indoor/train_indoor_ddpm_v2.py`
- `tools/stage3_indoor/sdedit_scout.py`
- `tools/stage3_indoor/sdedit_gaussian_full.py`
- `tools/stage3_indoor/diagnose_sdedit.py`
- `tools/stage3_indoor/generalization_diagnostic.py`
- `tools/stage3/refinement/run_refinement_interface.py`
- `tools/stage3/refinement/run_alpha_sweep.py`

Key checkpoints/configs:

- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/prior_check_v2.json`
- `data/stage3_indoor/rel_norm_params_v2.npz`
- `data/stage3_indoor/degradation_config.json`

Key metrics:

- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_scout_results.csv`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_summary.csv`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_per_traj.csv`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_conclusion.json`
- `outputs/stage3_indoor/sdedit_diagnostic/diagnostic_summary.csv`
- `outputs/stage3_indoor/sdedit_diagnostic/diagnostic_per_traj.csv`
- `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_summary.csv`
- `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_per_traj.csv`
- `outputs/stage3_indoor/report/tables/table1_gaussian_medium.csv`
- `outputs/stage3_indoor/report/tables/table2_generalization.csv`

Key figures/cache:

- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_scout.png`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_diagnostic.png`
- `outputs/stage3_indoor/sdedit_diagnostic/diagnostic_examples.png`
- `outputs/stage3_indoor/report/figures/fig5_unconditional_diagnostic.png`
- `outputs/stage3_indoor/report_v2/figures/fig06_tstart_sweep.png`
- `outputs/stage3_indoor/report_v2/figures/fig07_unconditional_diagnostic.png`
- `outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t2_refined.npy`

Related older missing-span refinement:

- `outputs/stage3/refinement/refinement_report.md`
- `outputs/stage3/refinement/eval/refinement_metrics.csv`
- `outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv`
- `outputs/stage3/refinement/alpha_sweep/alpha_sweep_report.md`

## Q5. What “Improved But Limited” Meant

### Gaussian Full SDEdit Sweep

Source:

- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_summary.csv`

| method | t_start | ADE | RMSE | smooth | ADE vs noisy | improved fraction |
|---|---:|---:|---:|---:|---:|---:|
| noisy_input | - | `0.062567` | `0.069788` | `0.231923` | `0.000000%` | `0.000` |
| kalman_cv | - | `0.085002` | `0.100040` | `0.105186` | `-35.858036%` | `0.235` |
| sdedit_t1 | 1 | `0.061277` | `0.068435` | `0.227724` | `2.060877%` | `0.795` |
| sdedit_t2 | 2 | `0.060812` | `0.067994` | `0.224015` | `2.804624%` | `0.765` |
| sdedit_t3 | 3 | `0.060951` | `0.068256` | `0.219811` | `2.582767%` | `0.700` |
| sdedit_t5 | 5 | `0.063093` | `0.070846` | `0.210903` | `-0.841318%` | `0.575` |

Best gaussian SDEdit was `t_start=2`:

- ADE: `0.062567 -> 0.060812`
- absolute ADE change: `-0.001755`
- relative improvement: `2.804624%`
- Wilcoxon p vs noisy: `1.674860e-13`

This is real but small.

### Six-Condition Generalization

Source:

- `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_summary.csv`

| condition | noisy ADE | uncond SDEdit t2 ADE | delta vs noisy | improved fraction | reading |
|---|---:|---:|---:|---:|---|
| gaussian_medium | `0.062567` | `0.060812` | `-0.001755` | `0.765` | small improvement |
| drift_medium | `0.017648` | `0.019486` | `+0.001838` | `0.350` | worsened |
| jump_medium | `0.263368` | `0.257183` | `-0.006185` | `0.740` | small improvement |
| burst_medium | `0.074755` | `0.082752` | `+0.007996` | `0.250` | worsened |
| bias_medium | `0.188996` | `0.189277` | `+0.000281` | `0.470` | no change / slight worse |
| combined_medium | `0.199064` | `0.198966` | `-0.000098` | `0.515` | negligible |

Smoothness generally decreases under SDEdit, but lower smoothness is not always better: for burst/drift, SDEdit can become smoother while ADE worsens.

High-confidence / low-confidence ADE was not reported in the old SDEdit outputs. Confidence did not exist in that line.

### Older 1000-Trajectory Diagnostic

Source:

- `outputs/stage3_indoor/sdedit_diagnostic/diagnostic_summary.csv`

This older diagnostic used `outputs/stage3_indoor/ddpm_prior/val_selected_model.pt`, not the final indoor-v2 EMA prior. It is still useful as a failure-mode warning:

| condition | noisy ADE | t1 ADE | t3 ADE | t5 ADE | t10 ADE | t20 ADE |
|---|---:|---:|---:|---:|---:|---:|
| gaussian_medium | `0.062373` | `0.094780` | `0.167567` | `0.247442` | `0.431539` | `0.719654` |
| burst_medium | `0.072314` | `0.118281` | `0.186684` | `0.259817` | `0.435451` | `0.719650` |

This shows that stronger noising quickly drifts away from the degraded observation when the prior/interface is weak.

### Related Older Missing-Span Alpha Sweep

This is not full-trajectory SDEdit. It is an earlier missing-segment reconstruction/refinement interface using a Stage 2 prior candidate and masked blending.

Sources:

- `outputs/stage3/refinement/refinement_report.md`
- `outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv`
- `outputs/stage3/refinement/alpha_sweep/alpha_sweep_report.md`

Mean masked-ADE best alpha:

| coarse method | best alpha | mean masked_ADE |
|---|---:|---:|
| linear_interp | `0.00` | `0.027261` |
| savgol_w5_p2 | `0.00` | `0.027930` |
| kalman_cv_dt1.0_q1e-3_r1e-2 | `0.10` | `0.049230` |

Main reading in the old report: larger alpha values generally hurt missing-segment reconstruction. The pure DDPM prior interface was much worse than naive coarse reconstruction, e.g. mean masked_ADE around `0.102` for DDPM prior v0 / masked replace.

This old alpha-sweep failure is related evidence that direct unconditional prior injection can be harmful when the interface is not carefully matched.

## Q6. Failure Cause Classification

| cause | verdict | evidence |
|---|---|---|
| A. prior domain mismatch | not primary for indoor-v2 SDEdit | final SDEdit used an indoor prior trained on `train_trajs.npy`; older missing-span alpha sweep used Stage 2 ETH+UCY prior and does show domain/interface mismatch risk |
| B. start_t not swept / bad start_t | partially no | gaussian final sweep did test `t_start=[1,2,3,5]`, and older diagnostic tested `[1,3,5,10,20]`; best gaussian was `t=2`; failure is not simply “never swept” |
| C. only ADE, no confidence-bin view | yes | old SDEdit reported ADE/RMSE/smoothness but not high/low confidence bins; confidence did not exist |
| D. absolute / relative interface mismatch | partially yes | code correctly converts abs to normalized relative and reconstructs abs, but relative prior cannot directly observe/correct global bias |
| E. anchor issue | yes for bias/global offset | reconstruction anchors at `degraded_abs[:,0,:]`; if the first point has bias, the absolute offset is preserved |
| F. implementation bug | no clear evidence | final v2 prior check passed, q_sample/reverse interface is explicit, gaussian improvement is reproducible |
| G. method limitation | yes | prior-only SDEdit has no observation conditioning during reverse denoising; larger `t_start` increases drift; small `t_start` barely changes the input |

## Bottom Line For Replanning

Do not treat old Stage 3 as if SDEdit was never tried. It was tried.

The old evidence says:

1. An indoor unconditional prior existed and passed its own prior check.
2. Observation-initialized SDEdit existed and used degraded trajectories as the noised starting point.
3. `t_start` was swept at least for gaussian/burst diagnostics.
4. The best final gaussian setting was `t_start=2`, with only a `2.8%` ADE gain.
5. Across six conditions, prior-only SDEdit was not robust.
6. Bias/global offset could not be fixed by the relative-displacement prior with degraded-start anchoring.
7. The likely failure is not a missing experiment; it is the combination of weak prior-only observation coupling, relative/anchor limits, and no per-frame reliability model.

## Recommendation

I do not recommend simply “redoing Stage 3-SDEdit” as the next main experiment unless the redo changes the scientific question.

A useful redo would need to be explicitly framed as one of these:

- an observation-initialized refinement with a better interface audit and preserved degraded anchor assumptions;
- a controlled comparison between Stage 2 ETH+UCY prior and Stage 3 indoor prior;
- a confidence-aware or likelihood-anchored variant that addresses the old failure modes;
- an absolute-space anchoring/bias-aware route if bias correction is the target.

Repeating the old prior-only SDEdit protocol would likely reproduce the same pattern: small gaussian gains, poor burst/drift behavior, no bias repair.

## Result Index

The machine-readable index for the artifacts above is:

- `outputs/stage4/stage3_sde_legacy_audit/stage3_sde_legacy_results_index.csv`
