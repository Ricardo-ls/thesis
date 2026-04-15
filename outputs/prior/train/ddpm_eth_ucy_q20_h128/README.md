# q20 variant training snapshots

This directory keeps the labeled `q20` Stage 2 runs in seed-and-epoch-labeled subfolders.

## Layout

- `seed42-100epoch/` keeps the 100-epoch reference run:
  - `RUN_NOTE_q20_ep100_seed42.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_curve_epoch10plus.png`
  - `loss_curve_epoch10plus.svg`
  - `loss_history.csv`
- `seed43-100epoch/` keeps the 100-epoch follow-up run:
  - `RUN_NOTE_q20_ep100_seed43.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_curve_epoch10plus.png`
  - `loss_curve_epoch10plus.svg`
  - `loss_history.csv`
- `seed44-100epoch/` keeps the 100-epoch follow-up run:
  - `RUN_NOTE_q20_ep100_seed44.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_curve_epoch10plus.png`
  - `loss_curve_epoch10plus.svg`
  - `loss_history.csv`
- `seed2-100epoch/`, `seed12-100epoch/`, `seed13-100epoch/`, `seed14-100epoch/`, `seed22-100epoch/`, `seed23-100epoch/`, `seed24-100epoch/`, `seed3-100epoch/`, `seed32-100epoch/`, `seed33-100epoch/`, `seed34-100epoch/`, and `seed4-100epoch/` are placeholder directories kept for seed alignment.

## Reading order

1. Read `seed42-100epoch/` for the first q20 reference run.
2. Use the seed-labeled filenames directly when restoring or copying files.
