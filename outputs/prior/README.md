# Stage 2 Prior Outputs

This directory stores the experiment artifacts for the Stage 2 trajectory prior pipeline.

## Scope

Stage 2 corresponds to trajectory-only diffusion prior pre-training on ETH+UCY under the unified DDPM protocol used in this repository. The contents here are organized to support:

- training snapshot recovery
- seeded qualitative figure generation
- distribution-level evaluation
- archival backup of intermediate and formal runs

## Layout

- `train/` stores checkpoint-level training artifacts
- `sample/` stores reverse-sampling outputs and seeded figure manifests
- `eval/` stores distribution-level diagnostics and summary metrics

## Training artifact convention

Each training run is stored in a directory named by seed and epoch budget:

- `seed42-100epoch/`
- `seed43-100epoch/`
- `seed44-100epoch/`
- `seed42-150epoch/`

Inside each completed run directory, the expected artifacts are:

- `best_model.pt`
- `last_model.pt`
- `loss_history.csv`
- `loss_curve_epoch10plus.png`
- `loss_curve_epoch10plus.svg`
- `RUN_NOTE_<variant>_ep<epochs>_seed<seed>.md`

The loss curve starts from epoch 10 and follows the project plotting convention:

- `train_loss` in deep blue
- `val_loss` in orange

## Interpretation

These outputs are archival experiment artifacts rather than lightweight public figures alone. They are intended to preserve the exact training state that produced the current Stage 2 conclusions and follow-up comparisons.

## Seeded Follow-up Batch

The seeded `100`-epoch follow-up sample/eval batch for `seed43` and `seed44` has been completed for all four official variants:

- `none`
- `q10`
- `q20`
- `q30`

The formal run summary is documented in:

- `docs/stage2_seed43_seed44_reference_runs.md`

That page records:

- the eight completed run combinations
- the sample and eval output paths
- the main mean-ratio diagnostics from `summary_metrics.csv`
- the interpretation boundary for these seeded follow-up runs
