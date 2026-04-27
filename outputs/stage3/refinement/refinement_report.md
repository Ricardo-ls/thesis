# Stage 3 Refinement Report

## Purpose

Test whether a lightweight refinement stage improves coarse reconstruction in the current controlled benchmark.

## Setup

- Degradation types: missing_only, missing_noise, missing_drift, missing_noise_drift
- Coarse methods: Linear, SG, Kalman
- Refiners: Identity, Light SG

## Metric Interpretation

Full-trajectory metrics measure overall consistency, while masked metrics measure reconstruction quality on the missing segment itself.
When the two views differ, both should be reported explicitly.

## Mean Results By Refiner

| coarse_method | refiner | ADE | RMSE | masked_ADE | masked_RMSE | improvement_ADE | improvement_masked_ADE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | Identity | 0.029192 | 0.023746 | 0.027261 | 0.022286 | 0.000000 | 0.000000 |
| Linear | Light SG | 0.026051 | 0.021100 | 0.027930 | 0.022847 | -0.032289 | -0.012001 |
| SG | Identity | 0.026051 | 0.021100 | 0.027930 | 0.022847 | 0.000000 | 0.000000 |
| SG | Light SG | 0.025491 | 0.020638 | 0.027648 | 0.022568 | 0.005052 | 0.008574 |
| Kalman | Identity | 0.032124 | 0.028721 | 0.049908 | 0.044530 | 0.000000 | 0.000000 |
| Kalman | Light SG | 0.031004 | 0.027351 | 0.047303 | 0.041511 | 0.022256 | 0.054381 |

## Interpretation

This report keeps full-trajectory metrics and masked metrics as complementary views.
The key question for refinement is whether masked_ADE improves, because that directly reflects missing-segment reconstruction quality.

Light SG does not behave as a universal improvement layer. Its effect depends on the coarse method and the degradation type.

- For Linear coarse trajectories, Light SG can improve full-trajectory ADE while slightly worsening masked_ADE under noise-heavy settings.
- For SG coarse trajectories, Light SG produces only small gains on average.
- For Kalman coarse trajectories, Light SG is the most helpful and improves both ADE and masked_ADE on average.

This means the refinement stage should not be judged only from the full-trajectory view. In the current benchmark, some settings show better overall consistency together with weaker missing-segment reconstruction. The figure `full_vs_masked_refinement_improvement.png` is included specifically to visualize that direction mismatch.

## Figures

- `figures/coarse_vs_refined_ADE.png`
- `figures/coarse_vs_refined_masked_ADE.png`
- `figures/improvement_bar_chart.png`
- `figures/full_vs_masked_refinement_improvement.png`
- `figures/trajectory_example_coarse_refined.png`
