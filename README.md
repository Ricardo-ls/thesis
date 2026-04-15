# Thesis: Stage 2 Trajectory Prior

This repository is the research and archival workspace for Stage 2 of the thesis: learning a high-fidelity, trajectory-only diffusion prior on ETH+UCY. The emphasis is not on a generic demo pipeline, but on a reproducible scientific record: data filtering, prior pre-training, seeded evaluation, and the written narrative that explains why each variant behaves the way it does.

## Reading Order

If you are new to the project, read in this order:

1. This README for the project thesis, structure, and repository map.
2. [`docs/prior_stage2.md`](docs/prior_stage2.md) for the full Stage 2 interpretation, figures, and reproducibility notes.
3. [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py) for the canonical registry that binds names to paths and semantics.

## Research Scope

The codebase is organized around a narrow scientific question:

- Can a diffusion prior trained only on pedestrian motion capture the geometry, smoothness, and dynamic plausibility of ETH+UCY trajectories?
- Which filtering policy yields the strongest balance between sample coverage and motion realism?
- How do the official variants compare when the protocol, seed, horizon, and evaluation budget are held fixed?

Stage 2 is the current center of gravity.

- Stage 2: trajectory-only diffusion prior pre-training and evaluation.
- Stage 3: downstream sensor-conditioned localization or filtering, treated as a separate future line of work.

## Canonical Variants

The registry in [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py) is the source of truth for semantics, directories, and narrative labels. The official Stage 2 names are:

- `none`: optimization-best baseline under the unified protocol
- `q10`: weakest filtering regime, useful as a low-pressure reference
- `q20`: balanced motion-focused prior and the recommended filtered variant
- `q30`: strongest filtering regime, more selective but less favorable on the full trade-off

Semantic aliases:

- `optimization_best -> none`
- `motion_balanced -> q20`

## Repository Layout

The repository is intentionally split by function:

- [`docs/`](docs) contains the paper-facing narrative, figures, and archived notes.
- [`outputs/prior/variants/`](outputs/prior/variants) is the top-level Stage 2 entry point, with one directory per official variant.
- [`outputs/prior/train|sample|eval/`](outputs/prior) remain the canonical artifact stores that the code still reads from.
- [`outputs/prior/archive/`](outputs/prior/archive) holds folded early-phase material.
- [`tools/prior/`](tools/prior) contains training, sampling, and evaluation entry points.
- [`utils/prior/`](utils/prior) contains the registry and shared semantic helpers.

## Stage 2 Artifacts

The Stage 2 archive is now organized so the four official variants are the primary conceptual units. The new outer directory is `outputs/prior/variants/`, and each variant exposes three task views:

- `train/` for checkpoints and training curves
- `sample/` for reverse-sampling artifacts
- `eval/` for distributional diagnostics and summary metrics

The early phase-A multi-seed material is folded under `outputs/prior/archive/` rather than treated as a first-class operating mode.

See:

- [`outputs/prior/README.md`](outputs/prior/README.md)
- [`outputs/prior/train/README.md`](outputs/prior/train/README.md)
- [`outputs/prior/variants/`](outputs/prior/variants)
- [`outputs/prior/archive/stage2_phaseA_multiseed_100epoch/eval`](outputs/prior/archive/stage2_phaseA_multiseed_100epoch/eval)
- [`docs/multi_seed_stage2_plan.md`](docs/multi_seed_stage2_plan.md)

## Local Setup

To make the repository portable across machines, a local bootstrap script is included at the repository root:

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

After that, training commands can be launched from the repository root, for example:

```bash
PYTHONPYCACHEPREFIX=/tmp MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -u -m tools.prior.train.train_ddpm_eth_ucy_h128 --variant none --epochs 100 --batch_size 128 --timesteps 100 --hidden_dim 128 --random_seed 42
```

## Reference Figures

The public Stage 2 figures are reproducible from the code in this repository without committing raw trajectory data, checkpoints, or large outputs.

Reference sampling and evaluation for the balanced prior:

```bash
PYTHONPYCACHEPREFIX=/tmp ./.venv/bin/python -m tools.prior.sample.reverse_sample_ddpm_eth_ucy_h128 \
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

PYTHONPYCACHEPREFIX=/tmp ./.venv/bin/python -m tools.prior.eval.analyze_generated_vs_real_eth_ucy_h128 \
  --variant motion_balanced \
  --num_generate 512 \
  --generated_rel_path outputs/prior/sample/ddpm_eth_ucy_q20_h128/reference_seed42/generated_rel_samples.npy \
  --reference_tag reference_seed42 \
  --save_manifest

./.venv/bin/python -m tools.prior.export_reference_figures \
  --variant motion_balanced \
  --reference_tag reference_seed42 \
  --include both
```

For the optimization-best baseline, replace `motion_balanced` with `optimization_best`.

## Safety Model

This repository is a public scientific archive, not a raw data dump.

- No raw pedestrian trajectory files are committed.
- No processed training corpora are committed.
- No checkpoints or large training outputs are committed beyond the curated snapshot archive.
- Publication-oriented figures and narrative docs live in `docs/`.

## Backup Branch

For rollback and recovery, a full snapshot is also stored on GitHub in the branch `backup/full-snapshot-2026-04-12`.

Use [`BACKUP.md`](BACKUP.md) for the shortest file-restore command and backup branch notes.
