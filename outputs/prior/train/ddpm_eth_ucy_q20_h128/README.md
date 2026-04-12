# q20 variant training snapshots

This directory keeps the labeled `q20` Stage 2 runs in seed-and-epoch-labeled subfolders.

## Layout

- `seed42-100epoch/` keeps the 100-epoch reference run:
  - `best_model.pt`
  - `last_model.pt`
  - `loss_history.csv`
- `seed43-100epoch/` is reserved for the next 100-epoch run and only contains `.gitkeep` for now.
- `seed44-100epoch/` is reserved for the next 100-epoch run and only contains `.gitkeep` for now.

## Reading order

1. Read `seed42-100epoch/` for the first q20 reference run.
2. Use the seed-labeled filenames directly when restoring or copying files.
