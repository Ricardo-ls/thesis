# Stage 3 Experiment Map

## 00. Old Task: Missing-Segment Inpainting

Purpose: document the original task where a segment was removed and reconstructed.

Key aliases:

- `results__fixed_gap_inpainting/` -> `outputs/stage3/inpainting_experiment/`
- `results__controlled_missing_benchmark/` -> `outputs/stage3/controlled_benchmark/`
- `results__ddpm_v3_diagnosis/` -> `outputs/stage3/diagnosis/`
- `code__old_refinement_tools/` -> `tools/stage3/refinement/`

Primary evidence:

- `full_results.csv`
- `REPORT.md`
- `v3_diagnosis_table.csv`

## 01. Data: Synthetic Indoor Full-Trajectory Setting

Purpose: controlled indoor clean/degraded pairs for full-trajectory refinement.

Key aliases:

- `data__clean_degraded_splits/` -> `data/stage3_indoor/`
- `code__generate_indoor_trajs.py`
- `code__generate_indoor_trajs_large.py`
- `code__degradation_generator.py`

Primary evidence:

- `clean_trajs.npy`
- `train_trajs.npy`
- `val_trajs.npy`
- `degraded_*_medium.npy`
- `rel_norm_params_v2.npz`

## 02. Prior: Unconditional Basic Sanity Check

Purpose: earliest unconditional DDPM prior check without SDEdit.

Key aliases:

- `results__basic_prior_training/` -> `outputs/stage3_indoor/ddpm_prior/`
- `results__basic_prior_diagnostics/` -> `outputs/stage3_indoor/ddpm_prior_diagnostics/`
- `code__train_ddpm_prior.py`
- `code__diagnose_ddpm_prior.py`

Primary evidence:

- `prior_check.json`
- `sampling_distribution.json`
- `one_step_denoise.csv`

## 03. Prior v2 and Unconditional SDEdit

Purpose: retrained indoor v2 prior, t_start sweep, and gaussian SDEdit diagnostic.

Key aliases:

- `results__indoor_v2_prior_and_sdedit/` -> `outputs/stage3_indoor/ddpm_indoor_v2/seed42/`
- `results__early_sdedit_diagnostic/` -> `outputs/stage3_indoor/sdedit_diagnostic/`
- `code__train_indoor_ddpm_v2.py`
- `code__sdedit_scout.py`
- `code__sdedit_gaussian_full.py`

Primary evidence:

- `prior_check_v2.json`
- `sdedit_gaussian_full_summary.csv`
- `sdedit_gaussian_full_conclusion.json`
- `sdedit_gaussian_full_per_traj.csv`

## 04. Baseline: Full-Trajectory Degradation

Purpose: clean-vs-degraded baseline before any refinement method.

Key aliases:

- `table__noisy_input_baselines_by_degradation.csv`
- `table__generalization_summary_with_noisy_rows.csv`
- `results__n200_degraded_arrays_and_generalization/`
- `data__n1000_degraded_arrays/`

Primary evidence:

- `table2_generalization.csv`
- `generalization_summary.csv`
- `generalization_degraded_*.npy`
- `degraded_*_medium.npy`

## 05. Model: Conditional Residual DDPM

Purpose: gaussian-trained observation-conditioned residual DDPM and generalization diagnostics.

Key aliases:

- `results__gaussian_trained_conditional_residual/`
- `code__train_cond_residual_gaussian.py`
- `code__generalization_diagnostic.py`
- `model__temporal_denoiser_conditional.py`

Primary evidence:

- `cond_residual_gaussian_summary.csv`
- `cond_residual_gaussian_conclusion.json`
- `generalization_summary.csv`
- `generalization_conclusion.json`
- `best_ema_model.pt`

## 06. Ablation: Receptive-Field Expansion

Purpose: test whether expanding Conv1D receptive field fixes the Stage 3 failure mode.

Key aliases:

- `results__rf_expansion_ablation/`
- `code__compute_rf_expansion.py`
- `code__train_4block.py`
- `code__eval_rf_expansion.py`
- `code__rf_expansion_report.py`
- `model__temporal_denoiser_conditional_4block.py`

Primary evidence:

- `ablation_summary.md`
- `ablation_results.csv`
- `raw_4block_eval.csv`
- `param_count.txt`
- `figs/rf_expansion_comparison.png`

## 07. Ablation: SSSD Smoke Test

Purpose: lightweight SSSD-related feasibility artifacts.

Key aliases:

- `results__sssd_smoke_test/`
- `code__sssd_smoke_test.py`
- `model__sssd_denoiser.py`

Primary evidence:

- `param_count.txt`

## 08. Reports: Stage 3 Indoor Closing

Purpose: report-level figures, tables, document outputs, and report builders.

Key aliases:

- `results__closing_report_v1/`
- `results__final_report_v2/`
- `code__build_stage3_closing_report.py`
- `code__build_stage3_final_report_v2.py`

Primary evidence:

- `report/tables/*.csv`
- `report/figures/*.png`
- `report_v2/figures/*.png`
- `report_v2/stage3_final_report.docx`

## 09. Code and Stage 4 Inventory Reference

Purpose: quick jump point for all Stage 3 indoor tools, models, and the Stage 4 preparation inventory.

Key aliases:

- `code__all_stage3_indoor_tools/`
- `models__stage3_backbones/`
- `docs__stage4_preparation_inventory_reference/`

