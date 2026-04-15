# `q10` Variant Training Snapshots

This directory preserves the `q10` Stage 2 prior archive.

## What It Represents

`q10` is the weak-filtering reference. It removes only a modest amount of low-quality motion while keeping the data manifold relatively broad.

## Run Inventory

- `seed42-100epoch/` contains the first 100-epoch reference run.
- `seed43-100epoch/` contains the follow-up 100-epoch run.
- `seed44-100epoch/` contains the later 100-epoch run.
- `seed2-100epoch/`, `seed3-100epoch/`, `seed4-100epoch/`, `seed12-100epoch/`, `seed13-100epoch/`, `seed14-100epoch/`, `seed22-100epoch/`, `seed23-100epoch/`, `seed24-100epoch/`, `seed32-100epoch/`, `seed33-100epoch/`, and `seed34-100epoch/` preserve the remaining archived runs.

## Required Files

Each completed run should contain:

- `RUN_NOTE_q10_ep<epochs>_seed<seed>.md`
- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`

## Reading Order

1. Start with `seed42-100epoch/` for the initial q10 reference run.
2. Continue with `seed43-100epoch/RUN_NOTE_q10_ep100_seed43.md`.
3. Use the seed-labeled filenames directly when restoring or copying files.
