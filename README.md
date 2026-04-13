# Thesis: Stage 2 Trajectory Prior

This repository contains the codebase for a trajectory diffusion thesis project focused on Stage 2: pre-training an unconditional pedestrian motion prior from public ETH+UCY trajectory data.

## Start Here

If you are new to the project, read in this order:

1. This README for the project-level scope and the official Stage 2 roles.
2. [`docs/prior_stage2.md`](docs/prior_stage2.md) for the full Stage 2 interpretation, figures, and reproducibility notes.
3. [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py) for the registry that defines the official semantics and paths.

## Project Scope

The thesis is organized in stages:

- Stage 2 is the current core focus: trajectory-only diffusion prior pre-training.
- Stage 3 and any downstream sensor-conditioned localization or filtering are separate follow-up stages.
- The current codebase is intentionally centered on motion prior learning, not sensor-to-position supervision.

## Stage 2 Summary

Stage 2 uses a registry-driven evaluation and documentation pipeline defined in [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py). That file is the single source of truth for Stage 2 paths, checkpoints, sample directories, evaluation directories, train/eval records, and narrative labels. The official semantic entry points are:

- `optimization_best -> none`
- `motion_balanced -> q20`

Official Stage 2 roles:

- `none`: optimization-best baseline under the unified protocol
- `q10`: filtering too weak; gains are limited
- `q20`: most balanced motion-focused prior among filtered variants
- `q30`: filtering too strong; not the default motion prior candidate

This summary is intentionally compact. Detailed figures and interpretation live in [`docs/prior_stage2.md`](docs/prior_stage2.md).

Scripts may accept either raw variants or semantic names through the registry, depending on the entry point implementation.

## Public Documentation

- [Stage 2 documentation and figures](docs/prior_stage2.md)
- [Backup snapshot and restore notes](BACKUP.md)

## Local Environment Setup

To make the repository portable across computers, a local bootstrap script is included in the repository root:

- `bootstrap_vscode_env.sh`

From a fresh clone, run:

```bash
chmod +x bootstrap_vscode_env.sh
./bootstrap_vscode_env.sh
```

This script will:

- create a repo-local `.venv`
- install the Python dependencies from `requirements.txt`
- write `.vscode/settings.json`, `.vscode/tasks.json`, and `.vscode/extensions.json`
- open a shell with the environment activated

After that, training commands can be launched directly from the repository root with:

```bash
PYTHONPYCACHEPREFIX=/tmp MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -u -m tools.prior.train.train_ddpm_eth_ucy_h128 --variant none --epochs 100 --batch_size 128 --timesteps 100 --hidden_dim 128 --random_seed 42
```

## Official Reference Figures

The public Stage 2 figures are reproducible from the code in this repository without committing raw trajectory data, checkpoints, or large outputs.

Official seeded reference runs:

```bash
PYTHONPYCACHEPREFIX=/tmp ../.venv/bin/python -m tools.prior.sample.reverse_sample_ddpm_eth_ucy_h128 \
  --variant motion_balanced \
  --sample_seed 42 \
  --vis_seed 42 \
  --num_generate 512 \
  --num_show 16 \
  --denoise_selection endpoint_quantile \
  --denoise_quantile 0.5 \
  --reference_tag reference_seed42 \
  --save_manifest \
  --device cpu

PYTHONPYCACHEPREFIX=/tmp ../.venv/bin/python -m tools.prior.eval.analyze_generated_vs_real_eth_ucy_h128 \
  --variant motion_balanced \
  --num_generate 512 \
  --generated_rel_path outputs/prior/sample/ddpm_eth_ucy_q20_h128/reference_seed42/generated_rel_samples.npy \
  --reference_tag reference_seed42 \
  --save_manifest

../.venv/bin/python -m tools.prior.export_reference_figures \
  --variant motion_balanced \
  --reference_tag reference_seed42 \
  --include both
```

For the optimization-best baseline, replace `motion_balanced` with `optimization_best`.

## Repository Safety

This public repository includes code, selected figures, and lightweight docs only.

- No raw pedestrian trajectory files are committed.
- No processed training corpora are committed.
- No checkpoints or large training outputs are committed.
- Only selected publication-oriented figures are kept in `docs/`.

## Backup Branch

For rollback and recovery, a full snapshot is also stored on GitHub in the branch `backup/full-snapshot-2026-04-07`.

Use [`BACKUP.md`](BACKUP.md) for the shortest file-restore command and backup branch notes.
