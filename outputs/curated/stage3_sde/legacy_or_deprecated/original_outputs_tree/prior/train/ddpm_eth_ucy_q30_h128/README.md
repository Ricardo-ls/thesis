# `q30` Variant Training Snapshots

This directory is part of the current evidence layer for the `q30` Stage 2 prior.

## Current Evidence

`q30` is the strong-filtering variant and serves as the high-constraint counterpoint to `q10` and `q20`.

## Run Inventory

- `seed42-100epoch/` contains the first 100-epoch run.
- `seed43-100epoch/` and `seed44-100epoch/` are part of the archived 15-seed sweep.
- `seed2-100epoch/`, `seed3-100epoch/`, `seed4-100epoch/`, `seed12-100epoch/`, `seed13-100epoch/`, `seed14-100epoch/`, `seed22-100epoch/`, `seed23-100epoch/`, `seed24-100epoch/`, `seed32-100epoch/`, `seed33-100epoch/`, and `seed34-100epoch/` preserve the remaining archived runs.

## Required Files

Each completed run should contain:

- `RUN_NOTE_q30_ep<epochs>_seed<seed>.md`
- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`

## Reading Order

1. Start with `seed42-100epoch/RUN_NOTE_q30_ep100_seed42.md`.
2. Use the seed-labeled filenames directly when restoring or copying files.
