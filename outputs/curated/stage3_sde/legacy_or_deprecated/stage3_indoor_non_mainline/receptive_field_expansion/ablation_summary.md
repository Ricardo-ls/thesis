# Stage 3 Receptive-Field Expansion Ablation Summary

## Purpose

This ablation tests whether expanding the current short-horizon Conv1D
denoiser from 2 residual blocks to 4 residual blocks provides measurable
gains for T=20 indoor trajectory refinement.

This experiment does not claim that the custom denoiser is
architecturally superior to SSSD-S4 or Diffusion-TS. It only tests
whether receptive-field expansion is a primary bottleneck in the current
short-window setting.

## Receptive field analysis

- Current 2-block Conv1D:
  - block-only receptive field = 9 frames
  - including-projection receptive field = 13 frames
  - coverage = 65.0% of T=20

- Expanded 4-block Conv1D:
  - block-only receptive field = 17 frames
  - including-projection receptive field = 21 frames
  - coverage = 105.0% of T=20

- Current 2-block parameters: 212354
- Expanded 4-block parameters: 409474
- Parameter ratio: expanded / current = 1.9283x

## Complete ADE matrix

| Method | gaussian | bias | drift | jump | burst | combined |
|--------|----------|------|-------|------|-------|----------|
| noisy_input | 0.0626 | 0.1890 | 0.0176 | 0.2634 | 0.0748 | 0.1991 |
| current 2-block Conv1D | 0.0786 | 0.1953 | 0.0584 | 0.2543 | 0.1121 | 0.2050 |
| expanded 4-block Conv1D | 0.0752 | 0.1949 | 0.0549 | 0.2510 | 0.1125 | 0.2051 |

## Headline numbers: gaussian_medium

| Method | ADE (m) | Δ vs current 2-block | p-value |
|--------|---------|----------------------|---------|
| noisy_input | 0.0626 | — | — |
| current 2-block Conv1D | 0.0786 | — | — |
| expanded 4-block Conv1D | 0.0752 | +4.3% | 1.2e-20 |

## Statistical note

- ADE_mean and ADE_std are computed across paired `trajectory_id × seed` evaluation rows in `raw_4block_eval.csv`.
- Wilcoxon p-values are computed only from paired per-trajectory/per-seed ADE values.
- Wilcoxon p-values are computed from paired `trajectory_id × seed` ADE rows in `raw_4block_eval.csv`.
- Any p-value shown as `p < 1e-300` was under numerical precision rather than literally zero.

## Key finding

Expanding the Conv1D backbone from the current 2-block model to the full-window 4-block model yields statistically detectable but practically small improvements. The median relative ADE improvement is only approximately 0.73%, and the expanded model still underperforms the noisy input baseline in five out of six degradation conditions. Therefore, the Stage 3 failure mode is not primarily caused by insufficient receptive-field capacity. The main bottleneck is the refinement/posterior interface, motivating Stage 4 to focus on sensor-anchored or confidence-aware posterior refinement rather than full SSSD-S4 porting.

Additional summary:
- Median relative ADE improvement of 4-block over 2-block: 0.73%
- Degradations where 4-block beats 2-block: 4/6
- Degradations where 4-block beats noisy_input: 1/6

Conclusion:

Expanding the Conv1D receptive field from the current 2-block setting to the expanded 4-block setting does not produce a practically meaningful improvement under the current T=20 indoor refinement protocol. This suggests that receptive-field capacity is not the primary bottleneck. The Stage 4 effort should therefore focus on the sensor-conditioning and posterior refinement interface.

## Backbone choice implication

The custom TemporalDenoiser1D is not presented as a universal
replacement for SSSD-S4 or Diffusion-TS. It is retained here as a
controlled short-horizon denoising backbone for T=20 indoor
trajectories. The Stage 4 contribution will focus on sensor-anchored
posterior refinement, because Stage 3 failure analysis indicates that
the main limitation is the missing observation anchor rather than
long-range temporal modeling capacity.
