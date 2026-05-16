# Stage 2 Prior Outputs

This directory is the current evidence root for Stage 2 prior outputs. It preserves the training, sampling, and evaluation artifacts that support the current mainline interpretation.

## Current Evidence

The current evidence is organized around the four official variants:

- `none`
- `q10`
- `q20`
- `q30`

The main human-facing entry point is [`variants/`](variants). Within each variant, outputs are separated by role:

- `train/` for optimization snapshots and loss traces
- `sample/` for reverse-sampling outputs and seed manifests
- `eval/` for distributional diagnostics, ratios, and summary metrics

## Documentation Classification

For clarity, files in this area are grouped into the following classes:

- `README.md` files provide directory-level interpretation and should be treated as current navigation aids.
- `RUN_NOTE_*.md` files are run records produced during training.
- `analysis_config.txt` and `manifest.json` files are execution metadata and should be treated as provenance records.
- PNG and SVG files are figure artifacts, not narrative documents.

When reading this directory, prefer the current mainline documentation in [`docs/`](../../docs) and use the files here as evidence-bearing outputs.

## Training Snapshot Convention

Completed training runs are stored as `seed<k>-<epochs>epoch/` folders.

Typical examples:

- `seed2-100epoch/`
- `seed12-100epoch/`
- `seed14-100epoch/`
- `seed42-100epoch/`
- `seed42-150epoch/`

Inside each completed run directory, the expected artifacts are:

- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`
- `RUN_NOTE_<variant>_ep<epochs>_seed<seed>.md`

The loss curve begins at epoch 10 to match the project plotting convention and avoid the visually noisy warm-up region.

## Formal Reading Order

1. [`README.md`](../../README.md) for the thesis-level framing.
2. [`docs/prior_stage2.md`](../../docs/prior_stage2.md) for the current Stage 2 interpretation.
3. [`docs/stage2_phaseA_multiseed_100epoch_report.md`](../../docs/stage2_phaseA_multiseed_100epoch_report.md) for the completed 15-seed, multi-seed, 100-epoch mainline result.
4. [`utils/prior/ablation_paths.py`](../../utils/prior/ablation_paths.py) for the canonical path registry.
