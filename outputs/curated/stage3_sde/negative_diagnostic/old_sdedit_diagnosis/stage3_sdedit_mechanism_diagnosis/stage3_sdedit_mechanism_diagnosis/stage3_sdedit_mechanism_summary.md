# Stage 3 SDEdit Mechanism Diagnosis

Static mechanism diagnosis only. No model was trained, no checkpoint was modified, and no old SDEdit method was rerun.

## 1. Do old SDEdit refined trajectories exist?

- full per-frame availability for all requested diagnostics: `False`
- found unique per-frame SDEdit refined outputs: `1`

- Found: `outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t2_refined.npy` shape `(200, 20, 2)`

The archive is incomplete: only `gaussian_medium`, `t_start=2` has a per-frame SDEdit cache in the checked locations.

## 2. Coverage

Summary-level metrics cover six-condition `uncond_sdedit_t2` plus gaussian/burst legacy sweeps. Per-frame confidence-bin analysis only covers gaussian t=2.

## 3. ADE Improvement By Condition

| condition | t_start | noisy_ADE | sdedit_ADE | ADE_delta | ADE_relative_improvement_pct | improved_fraction | noisy_acceleration | sdedit_acceleration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 2 | 0.062567 | 0.060812 | -0.001755 | 2.804624 | 0.765000 | 0.231923 | 0.224015 |
| drift_medium | 2 | 0.017648 | 0.019486 | 0.001838 | -10.413415 | 0.350000 | 0.134721 | 0.133044 |
| burst_medium | 2 | 0.074755 | 0.082752 | 0.007996 | -10.696698 | 0.250000 | 0.315608 | 0.307113 |
| bias_medium | 2 | 0.188996 | 0.189277 | 0.000281 | -0.148692 | 0.470000 | 0.133433 | 0.132023 |
| jump_medium | 2 | 0.263368 | 0.257183 | -0.006185 | 2.348544 | 0.740000 | 0.236007 | 0.229932 |
| combined_medium | 2 | 0.199064 | 0.198966 | -0.000098 | 0.049345 | 0.515000 | 0.232119 | 0.224224 |

Main improvements: `gaussian_medium, jump_medium, combined_medium`.
Main degradations: `drift_medium, burst_medium, bias_medium`.

## 4. High / Mid / Low Confidence Bin Changes

partial gaussian t=2: high delta=0.006115, low delta=-0.004159

| condition | t_start | bin | N_frames | noisy_ADE | sdedit_ADE | ADE_delta |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 2 | high | 334 | 0.014083 | 0.020198 | 0.006115 |
| gaussian_medium | 2 | mid | 2177 | 0.046816 | 0.045498 | -0.001318 |
| gaussian_medium | 2 | low | 1489 | 0.096470 | 0.092311 | -0.004159 |

Full six-condition confidence-bin diagnosis is unavailable because six-condition per-frame SDEdit outputs were not archived.

## 5. t_start Curve Shape

- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_summary.csv::gaussian_medium`: `small_U_shaped_sweet_spot`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_scout_results.csv::burst_medium`: `monotonic_worse`
- `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_scout_results.csv::gaussian_medium`: `small_U_shaped_sweet_spot`
- `outputs/stage3_indoor/sdedit_diagnostic/diagnostic_summary.csv::burst_medium`: `monotonic_worse`
- `outputs/stage3_indoor/sdedit_diagnostic/diagnostic_summary.csv::gaussian_medium`: `monotonic_worse`

Interpretation: the final gaussian-v2 curve is a small U-shaped sweet spot with best `t_start=2`; scout/older burst curves are monotonic worse; the older prior diagnostic is monotonic worse for both gaussian and burst.

## 6. Drift / Burst Failure Size

- `drift_medium`: ADE delta `0.001838` (-10.413% relative improvement; negative means worse in the table convention).
- `burst_medium`: ADE delta `0.007996` (-10.697% relative improvement; negative means worse in the table convention).

Using the saved six-condition table, drift degradation is small in absolute ADE but meaningful relative to its low noisy baseline; burst degradation is larger and clearly destructive.

## 7. Prior Sample Quality

old saved prior sample step mean ratio=1.081, acceleration RMS ratio=1.100, off-room=0.245

| source | exists | group | n | step_norm_mean | step_norm_p95 | total_length_mean | acceleration_rms_mean | off_room_ratio | nearest_neighbor_l2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outputs/stage3_indoor/ddpm_prior/generated_abs.npy | 1 | prior_sample | 200 | 0.192680 | 0.420024 | 3.660911 | 0.279213 | 0.245250 | 0.342822 |
| outputs/stage3_indoor/ddpm_prior_diagnostics/generated_abs_check.npy | 1 | prior_sample | 512 | 0.191244 | 0.420006 | 3.633640 | 0.272099 | 0.261816 | 0.361126 |
| data/stage3_indoor/clean_trajs.npy | 1 | clean_reference | 2000 | 0.178316 | 0.321809 | 3.388010 | 0.253911 | 0.000000 | 0.000000 |

The saved unconditional samples found here are from the older `ddpm_prior` line, not the final indoor-v2 EMA prior. The final v2 prior has `prior_check_v2.json`, but no saved generated trajectory array was found in the checked locations.

## 8. Most Likely Mechanisms

- `vanilla SDEdit lacks confidence awareness`: supported. It applies the prior uniformly after noising and has no per-frame reliability control.
- `start_t not adaptive`: supported. Gaussian has a tiny sweet spot; burst worsens as intervention grows.
- `relative representation cannot fix bias`: supported by no-change bias result.
- `anchor preserves absolute offset`: supported by reconstruction with degraded `y[0]` anchor.
- `prior too weak`: plausible but not fully decidable from archived final-v2 samples because final-v2 generated arrays were not saved; `prior_check_v2.json` says the prior passed its internal checks.
- `domain / scale mismatch`: not primary for final indoor-v2 SDEdit because the prior is indoor-normalized; relevant for older Stage 2 prior blending.
- `implementation issue`: no direct evidence from archived outputs; gaussian t=2 improvement and prior check suggest the pipeline ran coherently.

## 9. Recommendation

Recommendation: do not continue vanilla SDEdit as-is.

A confidence-aware SDEdit follow-up is scientifically plausible only if it is explicitly framed as addressing the old failure mode: high-reliability frames should be protected while low-reliability frames receive prior intervention. But because the old archive lacks six-condition per-frame SDEdit outputs, first recreate the missing old outputs only if the goal is a strict apples-to-apples mechanism study.

For reporting, the current old SDEdit result should be treated as a negative / limited prior-only result: small gaussian/jump gains, drift/burst harm, and no bias repair.

## Figures

- t_start sweep figures are under `outputs/stage4/stage3_sdedit_mechanism_diagnosis/figures/`.
- gaussian partial representative figure written: `True`.
