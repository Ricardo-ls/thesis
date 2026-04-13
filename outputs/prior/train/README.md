# Prior Training Outputs

This directory stores Stage 2 training artifacts for the ETH+UCY prior pipeline.

## Layout

- `ddpm_eth_ucy_none_h128/` - none variant training snapshots
- `ddpm_eth_ucy_q10_h128/` - q10 variant training snapshots
- `ddpm_eth_ucy_q20_h128/` - q20 variant training snapshots
- `ddpm_eth_ucy_q30_h128/` - q30 variant training snapshots

Each variant folder has its own `README.md` so the run folders and run notes stay easy to read.

## Multi-seed convention

For the active multi-seed study, keep each run in a seed-and-epoch-labeled subdirectory.

The current seed expansion set is:

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

Recommended focus:

- all four variants now follow the same `100`-epoch seed expansion
- `none` additionally preserves the `seed42-150epoch/` reference run

Use consistent names such as:

- `seed2-100epoch`
- `seed12-100epoch`
- `seed42-100epoch`
- `seed42-150epoch`

Keep one run per folder so the results stay easy to compare and recover. Every completed run should save:

- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`
- `RUN_NOTE_<variant>_ep<epochs>_seed<seed>.md`
