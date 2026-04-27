# Stage 3 Refinement Report

## Purpose

Test whether a lightweight refinement stage improves coarse reconstruction in the current controlled benchmark.

## Setup

- Degradation types: missing_only, missing_noise, missing_drift, missing_noise_drift
- Coarse methods: Linear, SG, Kalman
- Refiners: Identity, Light SG, DDPM prior interface v0

## Metric Interpretation

Full-trajectory metrics measure overall consistency, while masked metrics measure reconstruction quality on the missing segment itself.
When the two views differ, both should be reported explicitly.

## Mean Results By Refiner

| coarse_method | refiner | ADE | RMSE | masked_ADE | masked_RMSE | improvement_ADE | improvement_masked_ADE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | Identity | 0.029192 | 0.023746 | 0.027261 | 0.022286 | 0.000000 | 0.000000 |
| Linear | Light SG | 0.026051 | 0.021100 | 0.027930 | 0.022847 | -0.032289 | -0.012001 |
| Linear | DDPM prior v0 | 0.103835 | 0.092082 | 0.102050 | 0.081497 | -15.447375 | -4.273860 |
| SG | Identity | 0.026051 | 0.021100 | 0.027930 | 0.022847 | 0.000000 | 0.000000 |
| SG | Light SG | 0.025491 | 0.020638 | 0.027648 | 0.022568 | 0.005052 | 0.008574 |
| SG | DDPM prior v0 | 0.103463 | 0.091919 | 0.102437 | 0.081710 | -11.184825 | -4.337088 |
| Kalman | Identity | 0.032124 | 0.028721 | 0.049908 | 0.044530 | 0.000000 | 0.000000 |
| Kalman | Light SG | 0.031004 | 0.027351 | 0.047303 | 0.041511 | 0.022256 | 0.054381 |
| Kalman | DDPM prior v0 | 0.102565 | 0.091189 | 0.102030 | 0.082180 | -3.467971 | -1.199473 |

## Interpretation

This report keeps full-trajectory metrics and masked metrics as complementary views.
The key question for refinement is whether masked_ADE improves, because that directly reflects missing-segment reconstruction quality.

## First learned-prior refinement interface

The `ddpm_prior_interface_v0` refiner is the first Stage 3 interface-level connection to the Stage 2 learned prior.
It is not yet a fully optimized conditional diffusion refinement model.
Instead, it uses a one-shot prior projection in relative displacement space and then maps the result back to absolute trajectories.
Its purpose is to verify that Stage 2 prior checkpoints can be connected cleanly to Stage 3 coarse reconstructions.

In the current benchmark, the interface is operational but the v0 prior refinement does not improve over the simple baselines.
Its mean ADE and mean masked_ADE are both worse than Identity and Light SG across all three coarse-method families.
This is still a useful result because it validates the integration path without overclaiming performance.

## Figures

- `figures/coarse_vs_refined_ADE.png`
- `figures/coarse_vs_refined_masked_ADE.png`
- `figures/improvement_bar_chart.png`
- `figures/full_vs_masked_refinement_improvement.png`
- `figures/ddpm_vs_naive_refinement_improvement.png`
- `figures/trajectory_example_coarse_refined.png`
