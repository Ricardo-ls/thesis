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
- `seed44-100epoch/` is reserved for the next run and only contains `.gitkeep` for now.

## Reading order

1. Read `seed42-100epoch/` if you want the first q10 reference run.
2. Read `seed43-100epoch/RUN_NOTE_q10_ep100_seed43.md` for the second q10 run.
3. Use the seed-labeled filenames directly when restoring or copying files.
