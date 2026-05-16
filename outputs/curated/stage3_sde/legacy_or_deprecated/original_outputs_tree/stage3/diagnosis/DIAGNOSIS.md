# v3 Failure Diagnosis

*Generated in 771.2 s on 1024 trajectories × 5 seeds for `missing_only`.*

## Step A: Scale Comparison

- ETH+UCY mean step: `0.241194`
- room3 mean step: `0.034471`
- ratio `r = mean_step_eth / mean_step_room3`: `6.997057`

| dataset | mean_step | std_step | p50_step | p95_step |
| --- | ---: | ---: | ---: | ---: |
| eth_ucy_train | 0.241194 | 0.189435 | 0.233449 | 0.551196 |
| room3_clean | 0.034471 | 0.026174 | 0.034873 | 0.076318 |

## Step D: Missing-only Comparison

| method | n_trajectories | n_seeds | masked_ADE_mean | masked_ADE_std | masked_RMSE_mean | masked_RMSE_std | off_map_ratio_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_interp | 1024 | 1 | 0.012931 | 0.012993 | 0.009712 | 0.009662 | 0.000000 |
| ddpm_v3_inpainting | 1024 | 5 | 0.124652 | 0.126183 | 0.098345 | 0.098749 | 0.011416 |
| ddpm_v3_inpainting_scaled | 1024 | 5 | 0.013796 | 0.012142 | 0.010392 | 0.009044 | 0.000000 |
| ddpm_v3_inpainting_anchored | 1024 | 5 | 0.031920 | 0.027569 | 0.024276 | 0.020879 | 0.000020 |

## Step E: Final Diagnosis

**判决：scale mismatch 占主导，但 endpoint conditioning 也明显失效**

- 原始 `ddpm_v3_inpainting` masked_ADE mean: `0.124652`
- `ddpm_v3_inpainting_scaled` masked_ADE mean: `0.013796`
- `ddpm_v3_inpainting_anchored` masked_ADE mean: `0.031920`

- `ddpm_v3_inpainting_scaled` 与 `ddpm_v3_inpainting_anchored` 都把 masked_ADE 降到原始 v3 的 50% 以下。
- 这说明两个因素都在起作用，需要同时控制尺度与端点条件。

## Notes

- `ddpm_v3_inpainting_scaled`: v3 sample → relative displacement → divide by `r` → absolute trajectory → missing-span endpoint anchoring.
- `ddpm_v3_inpainting_anchored`: no scale calibration; only apply missing-span endpoint anchoring.
- Evaluation metrics are computed only on the existing `missing_only` controlled benchmark subset; Stage 2 training is unchanged.
