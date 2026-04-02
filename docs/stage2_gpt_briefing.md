# Stage 2 GPT Briefing

This file is a factual handoff for writing Stage 2 documentation. It only records paths, public assets, and verified run metadata.

## Repository Scope

- Project: trajectory DDPM thesis codebase
- Stage 2 focus: trajectory-only diffusion prior pre-training
- Stage 3 and downstream sensor-conditioned work are separate

## Official Registry Semantics

The registry in [`utils/prior/ablation_paths.py`](../utils/prior/ablation_paths.py) is the source of truth.

- `optimization_best -> none`
- `motion_balanced -> q20`

Official Stage 2 roles:

- `none`: optimization-best baseline
- `q10`: weak-filter reference point
- `q20`: most balanced motion-focused prior among filtered variants
- `q30`: strong-filter reference point

## Official Seeded Reference Runs

Fixed public figure protocol:

- `sample_seed = 42`
- `vis_seed = 42`
- `num_generate = 512`
- `num_show = 16`
- `denoise_selection = endpoint_quantile`
- `denoise_quantile = 0.5`
- `reference_tag = reference_seed42`
- `device = cpu`

Reference run output layout:

- `outputs/prior/sample/ddpm_eth_ucy_none_h128/reference_seed42/`
- `outputs/prior/sample/ddpm_eth_ucy_q10_h128/reference_seed42/`
- `outputs/prior/sample/ddpm_eth_ucy_q20_h128/reference_seed42/`
- `outputs/prior/sample/ddpm_eth_ucy_q30_h128/reference_seed42/`

Eval output layout:

- `outputs/prior/eval/ddpm_eth_ucy_none_h128/reference_seed42/`
- `outputs/prior/eval/ddpm_eth_ucy_q10_h128/reference_seed42/`
- `outputs/prior/eval/ddpm_eth_ucy_q20_h128/reference_seed42/`
- `outputs/prior/eval/ddpm_eth_ucy_q30_h128/reference_seed42/`

Public assets layout:

- `docs/assets/stage2/none/reference_seed42/`
- `docs/assets/stage2/q10/reference_seed42/`
- `docs/assets/stage2/q20/reference_seed42/`
- `docs/assets/stage2/q30/reference_seed42/`

Each public asset directory contains:

- `sample/real_vs_generated.png`
- `sample/denoise_check.png`
- `sample/manifest.json`
- `eval/summary_metrics.csv`
- `eval/manifest.json`
- `eval/*.png`

## Verified Training Records

These are the verified official Stage 2 training records currently used in the docs and registry.

### none

- `rel_path = datasets/processed/data_eth_ucy_20_rel.npy`
- `ckpt_path = outputs/prior/train/ddpm_eth_ucy_none_h128/best_model.pt`
- `samples = 36073`
- `best_val_loss = 0.090071`
- `best_epoch = 38`
- `denoise_index_resolved = 1190`

### q10

- `rel_path = datasets/processed/data_eth_ucy_20_rel_q10.npy`
- `ckpt_path = outputs/prior/train/ddpm_eth_ucy_q10_h128/best_model.pt`
- `samples = 32465`
- `best_val_loss = 0.09671`
- `best_epoch = 36`
- `denoise_index_resolved = 12921`

### q20

- `rel_path = datasets/processed/data_eth_ucy_20_rel_q20.npy`
- `ckpt_path = outputs/prior/train/ddpm_eth_ucy_q20_h128/best_model.pt`
- `samples = 28858`
- `best_val_loss = 0.10341`
- `best_epoch = 41`
- `denoise_index_resolved = 12116`

### q30

- `rel_path = datasets/processed/data_eth_ucy_20_rel_q30.npy`
- `ckpt_path = outputs/prior/train/ddpm_eth_ucy_q30_h128/best_model.pt`
- `samples = 25251`
- `best_val_loss = 0.104672`
- `best_epoch = 39`
- `denoise_index_resolved = 7087`

## Verified Evaluation Ratios

### none

- `step_norm_all = 0.917176`
- `endpoint_displacement = 0.815246`
- `moving_ratio_global = 1.107620`
- `propulsion_ratio = 0.857671`
- `acc_rms = 1.380279`

### q10

- `step_norm_all = 0.928575`
- `endpoint_displacement = 0.802554`
- `moving_ratio_global = 1.098594`
- `propulsion_ratio = 0.840675`
- `acc_rms = 1.516248`

### q20

- `step_norm_all = 0.945826`
- `endpoint_displacement = 0.826408`
- `moving_ratio_global = 1.045439`
- `propulsion_ratio = 0.845482`
- `acc_rms = 1.353333`

### q30

- `step_norm_all = 0.946900`
- `endpoint_displacement = 0.817517`
- `moving_ratio_global = 0.969652`
- `propulsion_ratio = 0.831879`
- `acc_rms = 1.337866`

## Loss Curves

Public loss curve assets:

- `docs/assets/prior/none_loss_curve.svg`
- `docs/assets/prior/q20_loss_curve.svg`

Verified references:

- none: `best_val_loss = 0.090071`, `best_epoch = 38`
- q20: `best_val_loss = 0.10341`, `best_epoch = 41`

## Selected Public Figure Paths

### Quality checks

- `docs/assets/stage2/none/reference_seed42/sample/real_vs_generated.png`
- `docs/assets/stage2/q20/reference_seed42/sample/real_vs_generated.png`
- `docs/assets/stage2/none/reference_seed42/sample/denoise_check.png`
- `docs/assets/stage2/q20/reference_seed42/sample/denoise_check.png`

### Distribution diagnostics

- `docs/assets/stage2/none/reference_seed42/eval/hist_endpoint_displacement.png`
- `docs/assets/stage2/q20/reference_seed42/eval/hist_endpoint_displacement.png`
- `docs/assets/stage2/q20/reference_seed42/eval/hist_propulsion_ratio.png`
- `docs/assets/stage2/q20/reference_seed42/eval/hist_acc_rms.png`

### Legacy prior figures still present in docs

- `docs/assets/prior/eval_none_hist_endpoint_displacement.png`
- `docs/assets/prior/eval_none_hist_propulsion_ratio.png`
- `docs/assets/prior/eval_q20/hist_endpoint_displacement.png`
- `docs/assets/prior/eval_q20/hist_propulsion_ratio.png`
- `docs/assets/prior/eval_q20/hist_acc_rms.png`

## Reproducibility Notes

- Public reference figures are seeded reference runs, not training reruns.
- Fixed seeds are for sample and visualization reproducibility.
- The public workflow does not change model architecture, training protocol, sampling math, or metric definitions.
- Public repository contains code, selected figures, manifests, and lightweight docs only.
- No raw pedestrian trajectory files, processed corpora, checkpoints, or large outputs are committed.

