# Thesis: Stage 2 Trajectory Prior

This repository is the research and archival workspace for Stage 2 of the thesis: learning a high-fidelity, trajectory-only diffusion prior on ETH+UCY. The emphasis is on a reproducible scientific record, not a generic demo pipeline.

## What This Repo Contains

- Stage 2 training, sampling, and evaluation code
- Canonical registry logic for variant and path resolution
- Seeded training snapshots and evaluation artifacts
- Paper-facing figures and narrative documentation

## Read In This Order

1. This README for the project-level scope and repository map.
2. [`docs/prior_stage2.md`](docs/prior_stage2.md) for the Stage 2 interpretation and figures.
3. [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py) for the canonical registry of variants and paths.

## Stage 2 Scope

Stage 2 is the current center of gravity.

- Stage 2: trajectory-only diffusion prior pre-training and evaluation
- Stage 3: downstream sensor-conditioned localization or filtering, treated as future work

The scientific question is narrow:

- Can a diffusion prior trained only on pedestrian motion capture the geometry, smoothness, and dynamic plausibility of ETH+UCY trajectories?
- Which filtering policy gives the best balance between motion realism and sample coverage?
- How do the four official variants compare when protocol and model are held fixed?

## Stage 2 Interpretation

The repository now distinguishes between the current mainline evidence, legacy reference material, and phase-working provenance.

Current mainline:

- `docs/stage2_phaseA_multiseed_100epoch_report.md` is the primary interpretation page.
- The completed 15-seed, multi-seed, 100-epoch Phase A sweep is the current evidence layer.
- It spans 15 train seeds: `2`, `3`, `4`, `12`, `13`, `14`, `22`, `23`, `24`, `32`, `33`, `34`, `42`, `43`, and `44`.
- The current mainline shortlist is `none`, with `q10` as the secondary candidate carried forward for the next stage.

Legacy reference layer:

- The original single-seed `50`-epoch interpretation is retained only for historical comparison.
- `none`, `q10`, `q20`, and `q30` remain the canonical variant names in the registry.

Semantic aliases still work:

- `optimization_best -> none`
- `motion_balanced -> q20`

Compatibility aliases are retained for traceability only and do not define the current planning layer.

The authoritative registry lives in [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py), but the current result reading should follow the completed 15-seed multi-seed report rather than the old single-seed summary.

## Repository Layout

- [`docs/`](docs): paper-facing narrative, figures, and archived notes
- [`docs/archive/`](docs/archive): historical and phase-specific documentation
- [`outputs/prior/train/`](outputs/prior/train): training snapshots organized by variant and seed
- [`outputs/prior/sample/`](outputs/prior/sample): reverse-sampling artifacts organized to mirror train
- [`outputs/prior/eval/`](outputs/prior/eval): evaluation artifacts organized to mirror train
- [`outputs/prior/variants/`](outputs/prior/variants): browsing entry point for the four official variants
- [`outputs/prior/archive/`](outputs/prior/archive): legacy reference outputs and folded historical material
- [`tools/prior/`](tools/prior): training, sampling, and evaluation entry points
- [`tools/legacy/`](tools/legacy): older exploratory scripts retained for traceability
- [`utils/prior/`](utils/prior): registry and shared semantic helpers

## Documentation Hygiene

The repository contains current mainline documents, legacy reference material, and phase-working provenance.

- Current mainline documents define the active Stage 2 reading.
- Phase-working provenance documents record project progression and are retained for traceability.
- Legacy reference material remains in the tree so the thesis record is auditable, but it does not participate in the current planning layer.

For a formal classification of the documentation and archive layers, see [`docs/DOCUMENT_CATALOG.md`](docs/DOCUMENT_CATALOG.md).

## Artifact Layout

The current evidence layer is seed-labeled and aligned across train, sample, and eval:

- `outputs/prior/train/ddpm_eth_ucy_{variant}_h128/seed{seed}-{epoch_tag}/`
- `outputs/prior/sample/ddpm_eth_ucy_{variant}_h128/seed{seed}-{epoch_tag}/reference_seed{sample_seed}/`
- `outputs/prior/eval/ddpm_eth_ucy_{variant}_h128/seed{seed}-{epoch_tag}/reference_seed{sample_seed}/`

Here:

- `train_seed` identifies the checkpoint source
- `sample_seed` controls the sampling and visualization protocol
- `reference_seed42` is a fixed protocol tag, not a training seed

The current evidence layer is read from:

- `outputs/prior/train/`
- `outputs/prior/sample/`
- `outputs/prior/eval/`

The `outputs/prior/archive/` tree is legacy reference material and should not be treated as the active evidence layer.

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

## Reference Workflow

Compatibility aliases are preserved for traceability only:

```bash
optimization_best -> none
motion_balanced -> q20
```

If you need a historical workflow example, consult the archived documentation rather than using these aliases as the default execution path.

## Safety Model

This repository is a public scientific archive, not a raw data dump.

- No raw pedestrian trajectory files are committed.
- No processed training corpora are committed.
- No checkpoints or large training outputs are committed beyond the curated snapshot archive.
- Publication-oriented figures and narrative docs live in `docs/`.

## Backup Branches

For rollback and recovery, full snapshots are kept in GitHub backup branches.

- `backup/full-snapshot-2026-04-16` recommended current backup target
- `backup/full-snapshot-2026-04-12` historical backup
- `backup/full-snapshot-2026-04-11` historical backup
- `backup/full-snapshot-2026-04-07` historical backup

If you do not see the backup in GitHub, switch branches in the repository view. The backup is not the default branch, so it will not appear unless you select it explicitly. New uploads should go to `backup/full-snapshot-2026-04-16`.

Use [`docs/archive/reference/BACKUP.md`](docs/archive/reference/BACKUP.md) for the shortest restore command and backup notes.
