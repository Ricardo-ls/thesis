# Variant-First Stage 2 Entry

This directory is the preferred top-level browsing entry for the Stage 2 prior archive.

## Layout

- `none/`
- `q10/`
- `q20/`
- `q30/`

Each variant directory contains three views:

- `train/`
- `sample/`
- `eval/`

These views point back to the canonical artifact stores under `outputs/prior/train`, `outputs/prior/sample`, and `outputs/prior/eval` so the repository stays compatible with the existing scripts.
