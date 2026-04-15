# q10 variant training snapshots

This directory keeps the labeled `q10` Stage 2 runs in seed-and-epoch-labeled subfolders.

## Layout

- `seed42-100epoch/` keeps the 100-epoch reference run:
  - `best_model.pt`
  - `last_model.pt`
  - `loss_history.csv`
- `seed43-100epoch/` keeps the 100-epoch follow-up run:
  - `RUN_NOTE_q10_ep100_seed43.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_history.csv`
- `seed2-100epoch/`, `seed12-100epoch/`, `seed13-100epoch/`, `seed14-100epoch/`, `seed22-100epoch/`, `seed23-100epoch/`, `seed24-100epoch/`, `seed3-100epoch/`, `seed32-100epoch/`, `seed33-100epoch/`, `seed34-100epoch/`, `seed4-100epoch/`, `seed44-100epoch/` are all seed-aligned training snapshots.
- Each completed seed directory contains the standard training artifacts:
  - `RUN_NOTE_q10_ep100_seed<seed>.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_curve_epoch10plus.png`
  - `loss_curve_epoch10plus.svg`
  - `loss_history.csv`

## Reading order

1. Read `seed42-100epoch/` if you want the first q10 reference run.
2. Read `seed43-100epoch/RUN_NOTE_q10_ep100_seed43.md` for the second q10 run.
3. Use the seed-labeled filenames directly when restoring or copying files.
