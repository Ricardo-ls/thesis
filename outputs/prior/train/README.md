# Prior Training Outputs

This directory preserves the training-side archive for the Stage 2 ETH+UCY prior.

## Conceptual Layout

The four official variants are the organizing principle:

- `ddpm_eth_ucy_none_h128/`
- `ddpm_eth_ucy_q10_h128/`
- `ddpm_eth_ucy_q20_h128/`
- `ddpm_eth_ucy_q30_h128/`

Each variant folder contains only completed runs and their run notes. The higher-level `outputs/prior/variants/` directory is the preferred browsing entry point.

## Run Naming

Completed runs are stored as `seed<k>-<epochs>epoch/` directories.

Common examples:

- `seed2-100epoch/`
- `seed3-100epoch/`
- `seed4-100epoch/`
- `seed12-100epoch/`
- `seed13-100epoch/`
- `seed14-100epoch/`
- `seed22-100epoch/`
- `seed23-100epoch/`
- `seed24-100epoch/`
- `seed32-100epoch/`
- `seed33-100epoch/`
- `seed34-100epoch/`
- `seed42-100epoch/`
- `seed43-100epoch/`
- `seed44-100epoch/`
- `seed42-150epoch/`

The `none` variant additionally preserves the `seed42-150epoch/` reference snapshot used in the narrative and comparisons.

## Required Files

Every completed run should contain:

- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`
- `RUN_NOTE_<variant>_ep<epochs>_seed<seed>.md`

## Archival Rule

Empty placeholders are intentionally omitted. The archive should contain only completed results, so the directory tree reads like a curated evidence table rather than a scratch workspace.
