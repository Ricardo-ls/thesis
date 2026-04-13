# Stage 2 Multi-Seed Plan

This plan keeps the Stage 2 prior experiments organized and easy to recover.

## Active Expansion Set

The active Phase A expansion now follows the explicit seed list:

- `2`
- `3`
- `4`
- `12`
- `13`
- `14`
- `22`
- `23`
- `24`
- `32`
- `33`
- `34`
- `42`
- `43`
- `44`

Although this is sometimes described informally as a "10-seed extension", the concrete list above contains `15` seeds. The list itself should be treated as authoritative.

## Variants

All four Stage 2 variants follow the same seed schedule:

- `none`
- `q10`
- `q20`
- `q30`

## Directory Convention

Store each run in a seed-and-epoch-labeled directory under the variant folder:

- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed2-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed12-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-150epoch/`
- `outputs/prior/train/ddpm_eth_ucy_q10_h128/seed24-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_q20_h128/seed33-100epoch/`
- `outputs/prior/train/ddpm_eth_ucy_q30_h128/seed44-100epoch/`

The same naming rule applies to all variants and all seeds in the active set.

## Naming Rule

Inside each completed run directory, keep one snapshot per run:

- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`
- `RUN_NOTE_<variant>_ep<epochs>_seed<seed>.md`

## Execution Rule

The training script now writes directly into the seed-and-epoch-labeled path and generates the loss curve automatically from epoch `10` onward.

For the current expansion, every new `100`-epoch run should land in:

- `seed2-100epoch`
- `seed3-100epoch`
- `seed4-100epoch`
- `seed12-100epoch`
- `seed13-100epoch`
- `seed14-100epoch`
- `seed22-100epoch`
- `seed23-100epoch`
- `seed24-100epoch`
- `seed32-100epoch`
- `seed33-100epoch`
- `seed34-100epoch`
- `seed42-100epoch`
- `seed43-100epoch`
- `seed44-100epoch`

## Current Status

The directory skeleton for the active expansion set has been created for all four variants. Existing completed runs are preserved in place, and empty future run directories are tracked with `.gitkeep` so they remain visible in the repository layout.
