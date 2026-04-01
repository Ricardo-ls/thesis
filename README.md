# Thesis: Stage 2 Trajectory Prior

This repository contains the codebase for a trajectory diffusion thesis project focused on Stage 2: pre-training an unconditional pedestrian motion prior from public ETH+UCY trajectory data.

## Project Scope

The thesis is organized in stages:

- Stage 2 is the current core focus: trajectory-only diffusion prior pre-training.
- Stage 3 and any downstream sensor-conditioned localization or filtering are separate follow-up stages.
- The current codebase is intentionally centered on motion prior learning, not sensor-to-position supervision.

## Stage 2 Summary

Stage 2 uses a registry-driven evaluation and documentation pipeline defined in [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py). The official semantic entry points are:

- `optimization_best -> none`
- `motion_balanced -> q20`

Official Stage 2 roles:

- `none`: optimization-best baseline under the unified protocol
- `q10`: filtering too weak; gains are limited
- `q20`: most balanced motion-focused prior among filtered variants
- `q30`: filtering too strong; not the default motion prior candidate

Scripts may accept either raw variants or semantic names through the registry. Paths, checkpoints, sample directories, evaluation directories, and official records are all resolved through the same source of truth in [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py).

## Public Documentation

- [Stage 2 documentation and figures](docs/prior_stage2.md)

## Repository Safety

This repository is intentionally limited to public-facing code, documentation, and selected publication-oriented figures.

- No raw pedestrian trajectory files are committed.
- No processed training corpora are committed.
- No checkpoints or large outputs are committed.
- Only selected figures needed to explain Stage 2 are kept in `docs/`.
