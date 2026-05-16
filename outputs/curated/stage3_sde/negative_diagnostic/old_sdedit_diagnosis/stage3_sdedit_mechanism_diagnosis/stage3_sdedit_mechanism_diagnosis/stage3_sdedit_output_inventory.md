# Stage 3 SDEdit Output Inventory

Static inventory only. No SDEdit trajectory was regenerated.

## Refined Trajectory Outputs

- expected refined-output entries checked: 10
- found unique files: 1
- missing entries: 8

### Found

- `outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t2_refined.npy` shape `(200, 20, 2)`

### Missing

- `six_condition_t2_expected` condition=`drift_medium` t_start=`2` candidates=`outputs/stage3_indoor/report/cache/drift_uncond_sdedit_t2_refined.npy`
- `six_condition_t2_expected` condition=`burst_medium` t_start=`2` candidates=`outputs/stage3_indoor/report/cache/burst_uncond_sdedit_t2_refined.npy`
- `six_condition_t2_expected` condition=`bias_medium` t_start=`2` candidates=`outputs/stage3_indoor/report/cache/bias_uncond_sdedit_t2_refined.npy`
- `six_condition_t2_expected` condition=`jump_medium` t_start=`2` candidates=`outputs/stage3_indoor/report/cache/jump_uncond_sdedit_t2_refined.npy`
- `six_condition_t2_expected` condition=`combined_medium` t_start=`2` candidates=`outputs/stage3_indoor/report/cache/combined_uncond_sdedit_t2_refined.npy`
- `gaussian_tstart_sweep_expected` condition=`gaussian_medium` t_start=`1` candidates=`outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_t1_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_sdedit_t1_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_uncond_sdedit_t1_refined.npy; outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t1_refined.npy`
- `gaussian_tstart_sweep_expected` condition=`gaussian_medium` t_start=`3` candidates=`outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_t3_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_sdedit_t3_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_uncond_sdedit_t3_refined.npy; outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t3_refined.npy`
- `gaussian_tstart_sweep_expected` condition=`gaussian_medium` t_start=`5` candidates=`outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_t5_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_sdedit_t5_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_uncond_sdedit_t5_refined.npy; outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t5_refined.npy`

## Other Legacy Artifacts Found

- keyword-matched files: 40

The detailed artifact search is reflected in `missing_output_inventory.md` and in the generated CSV diagnostics.
