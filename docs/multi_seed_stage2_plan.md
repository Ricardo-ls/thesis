# Stage 2 Multi-Seed Plan

This plan keeps the Stage 2 prior experiments organized and easy to recover.

## Recommended seed policy

- `none`: 3 seeds
- `q20`: 3 seeds
- `q10`: 1 reference seed
- `q30`: 1 reference seed

## Suggested seeds

- `42`
- `43`
- `44`

## Directory convention

Store each run in a seed-and-epoch-labeled directory under the variant folder:

- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed43-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed44-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-150epoch/`
- `outputs/prior/train/ddpm_eth_ucy_q20_h128/seed42-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_q20_h128/seed43-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_q20_h128/seed44-100epoch/`

Reference-only runs:

- `outputs/prior/train/ddpm_eth_ucy_q10_h128/seed42-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_q30_h128/seed42-100epoch/`

## Naming rule

Inside each run directory, keep one snapshot per run:

- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`
- `RUN_NOTE_*.md`

## Current status

This plan document defines the folder structure to use for future runs. The training script now writes directly into the seed-and-epoch-labeled path and generates the loss curve automatically from epoch 10 onward.
