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
- first `128` trajectories used for this quick run
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

- deterministic methods: `N = 128` trajectories
- DDPM methods: `N = 128 × 5 = 640` trajectory-seed cases
- one contiguous missing span is generated per trajectory in this run
- `per_case_results.csv` stores deterministic rows at trajectory level and DDPM rows at trajectory-seed level

## 11. Known limitations

- Existing checkpoint may have been trained on the same public trajectory corpus, so this is a pipeline sanity check, not a strict generalization test.
- The run is intentionally quick and therefore uses a subset of the available ETH+UCY windows rather than a full-corpus exhaustive evaluation.
- `ddpm_v3_inpainting` currently generates directly from degraded observed input; the saved five-column figure uses `linear_interp` as a reference coarse reconstruction column rather than a true upstream dependency of v3.

## Key Stats Snapshot

| method | missing_condition | metric | N | mean | std | median | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_interp | missing_only | masked_ADE | 128 | 0.117882 | 0.096267 | 0.090939 | 0.038711 | 0.193341 |
| linear_interp | missing_only | masked_RMSE | 128 | 0.088840 | 0.071852 | 0.065301 | 0.029549 | 0.142321 |
| linear_interp | missing_only | endpoint_error | 128 | 0.098784 | 0.097943 | 0.070128 | 0.025865 | 0.146326 |
| savgol_w5_p2 | missing_only | masked_ADE | 128 | 0.116141 | 0.094873 | 0.089454 | 0.036871 | 0.187464 |
| savgol_w5_p2 | missing_only | masked_RMSE | 128 | 0.088048 | 0.071053 | 0.064654 | 0.028697 | 0.139962 |
| savgol_w5_p2 | missing_only | endpoint_error | 128 | 0.096247 | 0.096406 | 0.068057 | 0.031690 | 0.139311 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | masked_ADE | 128 | 0.320332 | 0.257278 | 0.259567 | 0.115428 | 0.524409 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | masked_RMSE | 128 | 0.248345 | 0.197583 | 0.201005 | 0.087067 | 0.426697 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | missing_only | endpoint_error | 128 | 0.496470 | 0.413126 | 0.384893 | 0.175911 | 0.885380 |
| ddpm_v3_inpainting | missing_only | masked_ADE | 640 | 0.307332 | 0.230151 | 0.254322 | 0.121792 | 0.457011 |
| ddpm_v3_inpainting | missing_only | masked_RMSE | 640 | 0.239805 | 0.176179 | 0.206190 | 0.099791 | 0.355524 |
| ddpm_v3_inpainting | missing_only | endpoint_error | 640 | 0.469053 | 0.355806 | 0.394472 | 0.180067 | 0.673784 |
| ddpm_v3_inpainting_anchored | missing_only | masked_ADE | 640 | 0.126066 | 0.085908 | 0.115917 | 0.052599 | 0.185448 |
| ddpm_v3_inpainting_anchored | missing_only | masked_RMSE | 640 | 0.095714 | 0.064546 | 0.090295 | 0.040173 | 0.140949 |
| ddpm_v3_inpainting_anchored | missing_only | endpoint_error | 640 | 0.105415 | 0.093445 | 0.081451 | 0.035026 | 0.143482 |

## Representative Figures

- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/median_case_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/best_ddpm_improvement_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/worst_ddpm_degradation_five_column.png`

## Selected Case Notes

- coarse reference method for figure columns: `linear_interp`
- missing exact intermediate note: the current v3 interface does not expose a separate coarse-dependent DDPM candidate generation stage; see `selected_cases.json` for the explicit note stored with each case.
