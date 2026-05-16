# Missing Output Inventory

The old Stage 3 SDEdit refined trajectory archive is incomplete for full mechanism diagnosis.

Only existing outputs were read. No old SDEdit script was rerun.

## Found Per-Frame SDEdit Refined Outputs

- `outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t2_refined.npy` shape `(200, 20, 2)`

## Missing Per-Frame Outputs Required For Full Diagnosis

- scope=`six_condition_t2_expected`, condition=`drift_medium`, t_start=`2`; checked `outputs/stage3_indoor/report/cache/drift_uncond_sdedit_t2_refined.npy`
- scope=`six_condition_t2_expected`, condition=`burst_medium`, t_start=`2`; checked `outputs/stage3_indoor/report/cache/burst_uncond_sdedit_t2_refined.npy`
- scope=`six_condition_t2_expected`, condition=`bias_medium`, t_start=`2`; checked `outputs/stage3_indoor/report/cache/bias_uncond_sdedit_t2_refined.npy`
- scope=`six_condition_t2_expected`, condition=`jump_medium`, t_start=`2`; checked `outputs/stage3_indoor/report/cache/jump_uncond_sdedit_t2_refined.npy`
- scope=`six_condition_t2_expected`, condition=`combined_medium`, t_start=`2`; checked `outputs/stage3_indoor/report/cache/combined_uncond_sdedit_t2_refined.npy`
- scope=`gaussian_tstart_sweep_expected`, condition=`gaussian_medium`, t_start=`1`; checked `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_t1_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_sdedit_t1_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_uncond_sdedit_t1_refined.npy; outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t1_refined.npy`
- scope=`gaussian_tstart_sweep_expected`, condition=`gaussian_medium`, t_start=`3`; checked `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_t3_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_sdedit_t3_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_uncond_sdedit_t3_refined.npy; outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t3_refined.npy`
- scope=`gaussian_tstart_sweep_expected`, condition=`gaussian_medium`, t_start=`5`; checked `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_t5_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_sdedit_t5_refined.npy; outputs/stage3_indoor/ddpm_indoor_v2/seed42/gaussian_uncond_sdedit_t5_refined.npy; outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t5_refined.npy`

## Consequence

Full confidence-bin ADE decomposition and representative trajectory failure-mode plots cannot be computed for all six conditions or for every `t_start` without the per-frame SDEdit outputs.

The script therefore reports:

- summary-level condition and `t_start` metrics from saved CSV files;
- a partial per-frame confidence-bin diagnosis only for `gaussian_medium`, `t_start=2`, because `outputs/stage3_indoor/report/cache/gaussian_uncond_sdedit_t2_refined.npy` exists;
- prior sample diagnostics only from old saved unconditional samples.

## To Reproduce Missing Per-Frame Outputs Later

Do not do this inside the current audit. If explicitly approved later, the closest old scripts are:

- `tools/stage3_indoor/sdedit_gaussian_full.py` for gaussian `t_start=[1,2,3,5]`, checkpoint `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt`, seeds `[42,43,44,45,46]`.
- `tools/stage3_indoor/generalization_diagnostic.py` for six-condition `uncond_sdedit_t2`, same indoor-v2 prior and normalization.
- `tools/stage3_indoor/diagnose_sdedit.py` for older diagnostic `t_start=[1,3,5,10,20]`, checkpoint `outputs/stage3_indoor/ddpm_prior/val_selected_model.pt`.
