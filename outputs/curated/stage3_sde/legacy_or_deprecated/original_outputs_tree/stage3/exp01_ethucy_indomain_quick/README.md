# exp01_ethucy_indomain_quick

## 1. Experiment name

`exp01_ethucy_indomain_quick`

## 2. Purpose

Quick in-domain sanity check of Stage 3 missing reconstruction on ETH+UCY public trajectories.

## 3. What this experiment tests

Whether the existing DDPM v3 reconstruction interface works when evaluation data come from the same public trajectory domain as the prior.

## 4. What this experiment does NOT test

- real Room3 performance
- real sensor deployment
- strict held-out generalization, unless a held-out split is explicitly implemented
- geometry-aware reconstruction

## 5. Dataset paths used

- absolute trajectories: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/datasets/processed/data_eth_ucy_20.npy`
- relative trajectories reference: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/datasets/processed/data_eth_ucy_20_rel.npy`
- metadata csv: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/datasets/processed/data_eth_ucy_20_meta.csv`
- summary csv: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/datasets/processed/data_eth_ucy_20_summary.csv`

Dataset notes:
- natural coordinate scale, no Room3 normalization
- first `1024` trajectories used for this quick run
- sequence length `T=20`
- contiguous missing span length `4` frames under `fixed` placement

## 6. Prior checkpoint path used

- objective: `optimization_best`
- recommended prior variant: `none`
- checkpoint path: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-100epoch/best_model.pt`

## 7. Method list

- `linear_interp`
- `savgol_w5_p2`
- `kalman_cv_dt1.0_q1e-3_r1e-2`
- `ddpm_v3_inpainting`
- `ddpm_v3_inpainting_anchored`

## 8. Missing condition

- `missing_only` only
- no added observation noise
- no added drift

## 9. Metric definitions

| metric | definition |
| --- | --- |
| masked_ADE | Mean Euclidean point error on masked frames only. |
| masked_RMSE | Root mean square coordinate error on masked frames only. |
| endpoint_error | Euclidean error at the last missing frame of the contiguous span. |
| path_length_error | Absolute path-length difference on the bounded subtrajectory covering the missing span and its two observed anchors. |
| acceleration_error | RMSE between full-trajectory second finite differences, used as a smoothness proxy. |

## 10. Statistical population

- deterministic methods: `N = 1024` trajectories
- DDPM methods: `N = 1024 × 5 = 5120` trajectory-seed cases
- one contiguous missing span is generated per trajectory in this run
- `per_case_results.csv` stores deterministic rows at trajectory level and DDPM rows at trajectory-seed level

## 11. Known limitations

- Existing checkpoint may have been trained on the same public trajectory corpus, so this is a pipeline sanity check, not a strict generalization test.
- The run is intentionally quick and therefore uses a subset of the available ETH+UCY windows rather than a full-corpus exhaustive evaluation.
- `ddpm_v3_inpainting` currently generates directly from degraded observed input; the saved five-column figure uses `linear_interp` as a reference coarse reconstruction column rather than a true upstream dependency of v3.

## Key Stats Snapshot

| method | missing_condition | metric | N | mean | std | median | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_interp | missing_only | masked_ADE | 1024 | 0.084269 | 0.085497 | 0.063745 | 0.027149 | 0.110830 |
| linear_interp | missing_only | masked_RMSE | 1024 | 0.063204 | 0.063442 | 0.048571 | 0.020577 | 0.084440 |
| linear_interp | missing_only | endpoint_error | 1024 | 0.069458 | 0.078720 | 0.048801 | 0.017085 | 0.090332 |
| savgol_w5_p2 | missing_only | masked_ADE | 1024 | 0.081871 | 0.083248 | 0.061651 | 0.026192 | 0.107683 |
| savgol_w5_p2 | missing_only | masked_RMSE | 1024 | 0.061897 | 0.062250 | 0.046990 | 0.019740 | 0.081946 |
| savgol_w5_p2 | missing_only | endpoint_error | 1024 | 0.064755 | 0.074673 | 0.044343 | 0.017091 | 0.083059 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | masked_ADE | 1024 | 0.277678 | 0.253865 | 0.224637 | 0.103496 | 0.376745 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | masked_RMSE | 1024 | 0.216318 | 0.197787 | 0.173482 | 0.082727 | 0.289861 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | endpoint_error | 1024 | 0.442780 | 0.417500 | 0.350486 | 0.149969 | 0.605453 |
| ddpm_v3_inpainting | missing_only | masked_ADE | 5120 | 0.304263 | 0.230202 | 0.251038 | 0.128246 | 0.429333 |
| ddpm_v3_inpainting | missing_only | masked_RMSE | 5120 | 0.239656 | 0.179509 | 0.198310 | 0.101651 | 0.340630 |
| ddpm_v3_inpainting | missing_only | endpoint_error | 5120 | 0.488641 | 0.373420 | 0.408063 | 0.200453 | 0.690761 |
| ddpm_v3_inpainting_anchored | missing_only | masked_ADE | 5120 | 0.089707 | 0.068702 | 0.073027 | 0.041708 | 0.117839 |
| ddpm_v3_inpainting_anchored | missing_only | masked_RMSE | 5120 | 0.068152 | 0.051644 | 0.055786 | 0.032026 | 0.089227 |
| ddpm_v3_inpainting_anchored | missing_only | endpoint_error | 5120 | 0.075014 | 0.070176 | 0.055867 | 0.028002 | 0.098790 |

## Representative Figures

- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/exp01_ethucy_indomain_quick/figures/median_case_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/exp01_ethucy_indomain_quick/figures/best_ddpm_improvement_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/exp01_ethucy_indomain_quick/figures/worst_ddpm_degradation_five_column.png`

## Selected Case Notes

- coarse reference method for figure columns: `linear_interp`
- missing exact intermediate note: the current v3 interface does not expose a separate coarse-dependent DDPM candidate generation stage; see `selected_cases.json` for the explicit note stored with each case.
