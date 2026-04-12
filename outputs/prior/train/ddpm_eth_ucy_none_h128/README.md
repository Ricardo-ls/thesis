# none variant training snapshots

This directory is the labeled archive for the `none` Stage 2 runs.

## Layout

- `seed42-100epoch/` keeps the 100-epoch reference run:
  - `RUN_NOTE_none_ep100_seed42.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_history.csv`
- `seed42-150epoch/` keeps the 150-epoch snapshot:
  - `RUN_NOTE_none_ep150_seed42.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_history.csv`
- `seed43-100epoch/` keeps the additional 100-epoch run:
  - `RUN_NOTE_none_ep100_seed43.md`
  - `best_model.pt`
  - `last_model.pt`
  - `loss_history.csv`
- `seed44-100epoch/` is reserved for the next 100-epoch run and only contains `.gitkeep` for now.

## Reading order

1. Read `seed42-100epoch/RUN_NOTE_none_ep100_seed42.md` for the 100-epoch reference entry.
2. Read `seed42-150epoch/RUN_NOTE_none_ep150_seed42.md` for the 150-epoch snapshot entry.
3. Read `seed43-100epoch/RUN_NOTE_none_ep100_seed43.md` for the second 100-epoch run.
4. Use the seed-labeled filenames directly when restoring or copying files.
