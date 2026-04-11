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

Store each run in a seed-labeled directory under the variant folder:

- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed42/`
- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed43/`
- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed44/`
- `outputs/prior/train/ddpm_eth_ucy_q20_h128/seed42/`
- `outputs/prior/train/ddpm_eth_ucy_q20_h128/seed43/`
- `outputs/prior/train/ddpm_eth_ucy_q20_h128/seed44/`

Reference-only runs:

- `outputs/prior/train/ddpm_eth_ucy_q10_h128/seed42/`
- `outputs/prior/train/ddpm_eth_ucy_q30_h128/seed42/`

## Naming rule

Inside each seed directory, keep one snapshot per run:

- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- optional `RUN_NOTE_*.md`

## Current status

This plan document defines the folder structure to use for future runs. Existing historical runs remain unchanged.
