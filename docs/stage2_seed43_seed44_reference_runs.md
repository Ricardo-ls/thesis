# Stage 2 Seed43 and Seed44 Reference Runs

This note summarizes the seeded `100`-epoch follow-up runs for the four official Stage 2 variants:

- `none`
- `q10`
- `q20`
- `q30`

The purpose of this page is to record the exact sample and evaluation outputs for:

- `seed43-100epoch`
- `seed44-100epoch`

under one shared public figure protocol:

- `sample_seed = 42`
- `vis_seed = 42`
- `num_generate = 512`
- `num_show = 16`
- `denoise_selection = endpoint_quantile`
- `denoise_quantile = 0.5`
- `reference_tag = reference_seed42`

These runs are not a replacement for the official Stage 2 registry interpretation. They are follow-up seeded comparisons that preserve:

- the same model family
- the same sampling math
- the same evaluation metrics
- the same seeded public-figure selection rule

## What Was Produced

For each of the `8` run combinations (`4` variants x `2` train seeds), the repository now contains:

- one `sample/` directory with seeded qualitative figures and `manifest.json`
- one `eval/` directory with distribution-level diagnostics, `summary_metrics.csv`, and `manifest.json`

Each sample directory contains:

- `generated_rel_samples.npy`
- `generated_abs_samples.npy`
- `real_vs_generated.png`
- `denoise_check.png`
- `manifest.json`

Each eval directory contains:

- `summary_metrics.csv`
- `analysis_config.txt`
- `manifest.json`
- `hist_step_norm.png`
- `hist_avg_speed.png`
- `hist_total_length.png`
- `hist_endpoint_displacement.png`
- `hist_moving_ratio_global.png`
- `hist_propulsion_ratio.png`
- `hist_acc_rms.png`
- `scatter_endpoint.png`
- `scatter_speed_vs_length.png`

## How to Read the Table

The table below reports four ratios from `summary_metrics.csv`:

- `endpoint ratio`: generated mean / real mean for `endpoint_displacement`
- `propulsion ratio`: generated mean / real mean for `propulsion_ratio`
- `moving ratio`: generated mean / real mean for `moving_ratio_global`
- `acc rms ratio`: generated mean / real mean for `acc_rms`

These are diagnostic ratios, not standalone rankings.

- Ratios near `1.0` indicate closer mean-level agreement with the real data on that metric.
- Values below `1.0` indicate an under-shoot relative to the real mean.
- Values above `1.0` indicate an over-shoot relative to the real mean.

Because the figure protocol fixes `vis_seed = 42`, the realized real-trajectory index list and generated-trajectory display index list are deterministic within each variant. The training seed changes the trained checkpoint and therefore the generated samples, but not the selection rule itself.

## Seed43 Summary

| Variant | Endpoint Ratio | Propulsion Ratio | Moving Ratio | Acc RMS Ratio | Sample Dir | Eval Dir |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `none` | `0.785708` | `0.858539` | `1.100980` | `1.428723` | `outputs/prior/sample/ddpm_eth_ucy_none_h128/seed43-100epoch/reference_seed42/` | `outputs/prior/eval/ddpm_eth_ucy_none_h128/seed43-100epoch/reference_seed42/` |
| `q10` | `0.789015` | `0.852560` | `1.086692` | `1.426239` | `outputs/prior/sample/ddpm_eth_ucy_q10_h128/seed43-100epoch/reference_seed42/` | `outputs/prior/eval/ddpm_eth_ucy_q10_h128/seed43-100epoch/reference_seed42/` |
| `q20` | `0.800575` | `0.828861` | `1.035139` | `1.520124` | `outputs/prior/sample/ddpm_eth_ucy_q20_h128/seed43-100epoch/reference_seed42/` | `outputs/prior/eval/ddpm_eth_ucy_q20_h128/seed43-100epoch/reference_seed42/` |
| `q30` | `0.778668` | `0.850613` | `0.959589` | `1.371479` | `outputs/prior/sample/ddpm_eth_ucy_q30_h128/seed43-100epoch/reference_seed42/` | `outputs/prior/eval/ddpm_eth_ucy_q30_h128/seed43-100epoch/reference_seed42/` |

### Seed43 Notes

- `none` remains relatively strong on propulsion retention while overestimating motion frequency.
- `q20` stays closer to `1.0` on moving ratio than `none` and keeps the filtered-prior interpretation intact, but it still undershoots endpoint progression and propulsion.
- `q30` pulls moving ratio below `1.0`, consistent with a stronger filtering regime.
- `q10` behaves as a weak-filter reference rather than a clean replacement for either `none` or `q20`.

## Seed44 Summary

| Variant | Endpoint Ratio | Propulsion Ratio | Moving Ratio | Acc RMS Ratio | Sample Dir | Eval Dir |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `none` | `0.843156` | `0.835438` | `1.113001` | `1.546005` | `outputs/prior/sample/ddpm_eth_ucy_none_h128/seed44-100epoch/reference_seed42/` | `outputs/prior/eval/ddpm_eth_ucy_none_h128/seed44-100epoch/reference_seed42/` |
| `q10` | `0.773150` | `0.852775` | `1.081541` | `1.398343` | `outputs/prior/sample/ddpm_eth_ucy_q10_h128/seed44-100epoch/reference_seed42/` | `outputs/prior/eval/ddpm_eth_ucy_q10_h128/seed44-100epoch/reference_seed42/` |
| `q20` | `0.782708` | `0.826793` | `1.027929` | `1.445800` | `outputs/prior/sample/ddpm_eth_ucy_q20_h128/seed44-100epoch/reference_seed42/` | `outputs/prior/eval/ddpm_eth_ucy_q20_h128/seed44-100epoch/reference_seed42/` |
| `q30` | `0.743636` | `0.838368` | `0.944836` | `1.313747` | `outputs/prior/sample/ddpm_eth_ucy_q30_h128/seed44-100epoch/reference_seed42/` | `outputs/prior/eval/ddpm_eth_ucy_q30_h128/seed44-100epoch/reference_seed42/` |

### Seed44 Notes

- `none` improves its endpoint ratio relative to the `seed43` run but still overshoots moving ratio and acceleration RMS.
- `q20` again stays closer than `none` on moving ratio, while continuing to undershoot endpoint and propulsion means.
- `q30` shows the strongest endpoint under-shoot among the four variants, which is consistent with an over-filtered reference reading.
- `q10` remains intermediate and does not overturn the current official narrative.

## Manifest and Figure Traceability

The sample manifests for this batch are:

- `outputs/prior/sample/ddpm_eth_ucy_none_h128/seed43-100epoch/reference_seed42/manifest.json`
- `outputs/prior/sample/ddpm_eth_ucy_q10_h128/seed43-100epoch/reference_seed42/manifest.json`
- `outputs/prior/sample/ddpm_eth_ucy_q20_h128/seed43-100epoch/reference_seed42/manifest.json`
- `outputs/prior/sample/ddpm_eth_ucy_q30_h128/seed43-100epoch/reference_seed42/manifest.json`
- `outputs/prior/sample/ddpm_eth_ucy_none_h128/seed44-100epoch/reference_seed42/manifest.json`
- `outputs/prior/sample/ddpm_eth_ucy_q10_h128/seed44-100epoch/reference_seed42/manifest.json`
- `outputs/prior/sample/ddpm_eth_ucy_q20_h128/seed44-100epoch/reference_seed42/manifest.json`
- `outputs/prior/sample/ddpm_eth_ucy_q30_h128/seed44-100epoch/reference_seed42/manifest.json`

The eval manifests for this batch are:

- `outputs/prior/eval/ddpm_eth_ucy_none_h128/seed43-100epoch/reference_seed42/manifest.json`
- `outputs/prior/eval/ddpm_eth_ucy_q10_h128/seed43-100epoch/reference_seed42/manifest.json`
- `outputs/prior/eval/ddpm_eth_ucy_q20_h128/seed43-100epoch/reference_seed42/manifest.json`
- `outputs/prior/eval/ddpm_eth_ucy_q30_h128/seed43-100epoch/reference_seed42/manifest.json`
- `outputs/prior/eval/ddpm_eth_ucy_none_h128/seed44-100epoch/reference_seed42/manifest.json`
- `outputs/prior/eval/ddpm_eth_ucy_q10_h128/seed44-100epoch/reference_seed42/manifest.json`
- `outputs/prior/eval/ddpm_eth_ucy_q20_h128/seed44-100epoch/reference_seed42/manifest.json`
- `outputs/prior/eval/ddpm_eth_ucy_q30_h128/seed44-100epoch/reference_seed42/manifest.json`

These manifests preserve:

- the resolved registry variant
- the train seed
- the train epoch budget
- the seeded figure-selection protocol
- the resolved denoise sample index
- the generated sample path used by evaluation

## Interpretation Boundary

This summary is intentionally conservative.

- It does not redefine the official Stage 2 conclusion.
- It does not claim that one filtered variant becomes the single global winner across all seeds and all diagnostics.
- It should be read as a traceable seeded follow-up layer on top of the official Stage 2 registry interpretation.

The current high-level reading remains:

- `none` is the optimization-best baseline under the unified protocol
- `q20` is the most balanced motion-focused prior among filtered variants
- `q10` and `q30` remain useful reference points for understanding weak and strong filtering behavior
