# q30 variant training snapshots

This directory keeps the labeled `q30` Stage 2 runs in seed-and-epoch-labeled subfolders.

## Layout

- `seed42-100epoch/` keeps the 100-epoch reference run:
  - `RUN_NOTE_q30_ep100_seed42.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_history.csv`
- `seed43-100epoch/` is reserved for the next 100-epoch run and only contains `.gitkeep` for now.
- `seed44-100epoch/` is reserved for the next 100-epoch run and only contains `.gitkeep` for now.

## Reading order

1. Read `seed42-100epoch/RUN_NOTE_q30_ep100_seed42.md` for the first q30 reference run.
2. Use the seed-labeled filenames directly when restoring or copying files.
