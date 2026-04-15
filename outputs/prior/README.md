# Stage 2 Prior Outputs

This directory is the artifact archive for Stage 2 of the thesis project. It preserves the exact training, sampling, and evaluation outputs that support the written conclusions.

## Design Principle

The archive is organized around the four official variants:

- `none`
- `q10`
- `q20`
- `q30`

The main human-facing entry point is [`variants/`](variants). Within each variant, outputs are separated by role:

- `train/` for optimization snapshots and loss traces
- `sample/` for reverse-sampling outputs and seed manifests
- `eval/` for distributional diagnostics, ratios, and summary metrics

## Archived Phase-A Material

The earlier multi-seed Stage 2 sweep is intentionally treated as folded archive material rather than a live operating mode.

- `outputs/prior/archive/stage2_phaseA_multiseed_100epoch/eval/`
- `docs/stage2_phaseA_multiseed_100epoch_report.md`

Those assets are retained for traceability, but the main entry point for the repository is the four-variant ETH+UCY prior archive.

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
2. [`docs/prior_stage2.md`](../../docs/prior_stage2.md) for the scientific interpretation.
3. [`utils/prior/ablation_paths.py`](../../utils/prior/ablation_paths.py) for the canonical path registry.

## Follow-Up Batch

The seeded `100`-epoch follow-up sample/eval batch for `seed43` and `seed44` is available for all four official variants.

The formal summary lives in:

- [`docs/stage2_seed43_seed44_reference_runs.md`](../../docs/stage2_seed43_seed44_reference_runs.md)

That page records:

- the eight completed run combinations
- the sample and eval output paths
- the main ratio diagnostics from `summary_metrics.csv`
- the interpretation boundary for the follow-up batch
