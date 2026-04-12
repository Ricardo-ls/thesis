# Prior Training Outputs

This directory stores Stage 2 training artifacts for the ETH+UCY prior pipeline.

## Layout

- `ddpm_eth_ucy_none_h128/` - none variant training snapshots
- `ddpm_eth_ucy_q10_h128/` - q10 variant training snapshots
- `ddpm_eth_ucy_q20_h128/` - q20 variant training snapshots
- `ddpm_eth_ucy_q30_h128/` - q30 variant training snapshots

Each variant folder has its own `README.md` so the run folders and run notes stay easy to read.

## Multi-seed convention

For the multi-seed study, keep each run in a seed-and-epoch-labeled subdirectory:

- `seed42-100epoch/`
- `seed43-100epoch/`
- `seed44-100epoch/`
- `seed42-150epoch/`

Recommended focus:

- `none`: multi-seed, with both 100-epoch and 150-epoch reference material preserved
- `q20`: multi-seed
- `q10`: multi-seed or reference follow-up, depending on what you are comparing
- `q30`: multi-seed or reference follow-up, depending on what you are comparing

Use consistent names such as:

- `seed42-100epoch`
- `seed43-100epoch`
- `seed44-100epoch`
- `seed42-150epoch`

Keep one run per folder so the results stay easy to compare and recover. Every completed run should save:

- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`
- `RUN_NOTE_<variant>_ep<epochs>_seed<seed>.md`
