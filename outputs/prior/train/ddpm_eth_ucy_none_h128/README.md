# `none` Variant Training Snapshots

This directory is the training archive for the `none` Stage 2 prior.

## What It Represents

`none` is the optimization-best baseline under the unified protocol. It is the reference against which the filtered variants are interpreted.

## Run Inventory

- `seed42-100epoch/` contains the canonical 100-epoch reference run.
- `seed42-150epoch/` contains the longer 150-epoch snapshot used in the narrative comparison.
- `seed43-100epoch/` and `seed44-100epoch/` capture the follow-up 100-epoch runs.
- `seed2-100epoch/`, `seed3-100epoch/`, `seed4-100epoch/`, `seed12-100epoch/`, `seed13-100epoch/`, `seed14-100epoch/`, `seed22-100epoch/`, `seed23-100epoch/`, `seed24-100epoch/`, `seed32-100epoch/`, and `seed33-100epoch/` preserve the remaining archived runs.

## Required Files

Each completed run should contain:

- `RUN_NOTE_none_ep<epochs>_seed<seed>.md`
- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`

## Reading Order

1. Start with `seed42-100epoch/RUN_NOTE_none_ep100_seed42.md`.
2. Read `seed42-150epoch/RUN_NOTE_none_ep150_seed42.md` for the extended run.
3. Use the seed-labeled filenames directly when restoring or copying files.
