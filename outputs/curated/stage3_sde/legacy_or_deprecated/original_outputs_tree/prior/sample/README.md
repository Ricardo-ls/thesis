# Prior Sampling Outputs

This directory is part of the current evidence layer for Stage 2 sampling outputs.

## Current Evidence

Sampling artifacts are organized by variant and seed to mirror the training archive.

- `none/`
- `q10/`
- `q20/`
- `q30/`

Within each variant directory, sampling runs are stored as `seed<k>-<epochs>epoch/reference_seed<sample_seed>/`.

## Legacy Reference

Older sampling batches may appear under the archive tree when they are retained for traceability. They do not alter the current evidence layer.

## Interpretation Rule

Use this directory for current sampling evidence and provenance. Historical comparison material belongs under `outputs/prior/archive/`.
