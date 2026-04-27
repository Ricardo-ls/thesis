# Thesis: Stage 2 Prior and Stage 3 Benchmark Scaffold

This repository is the research and archival workspace for the thesis trajectory track.

- Stage 2 is the completed trajectory-only diffusion prior study on ETH+UCY.
- Stage 3 now includes a minimal Phase 1 indoor trajectory imputation benchmark scaffold.

The emphasis remains a reproducible scientific record rather than a generic demo pipeline.

## What This Repo Contains

- Stage 2 training, sampling, and evaluation code
- Stage 3 phase-1 benchmark scripts for indoor missing-trajectory completion
- Canonical registry logic for variant and path resolution
- Seeded training snapshots and evaluation artifacts
- Paper-facing figures and narrative documentation

## Read In This Order

1. This README for the project-level scope and repository map.
2. [`docs/prior_stage2.md`](docs/prior_stage2.md) for the Stage 2 interpretation and figures.
3. [`docs/stage3/README.md`](docs/stage3/README.md) for Stage 3 navigation.
4. [`docs/stage3/planning/stage3_phase1_formal_spec.md`](docs/stage3/planning/stage3_phase1_formal_spec.md) for the formal Stage 3 phase-1 problem definition and scope boundary.
5. [`docs/stage3/planning/stage3_phase1_plan.md`](docs/stage3/planning/stage3_phase1_plan.md) for the current Stage 3 phase-1 benchmark boundary.
6. [`变更记录.md`](变更记录.md) for the required Chinese modification record.
7. [`utils/prior/ablation_paths.py`](utils/prior/ablation_paths.py) for the canonical registry of Stage 2 variants and paths.

## Repository Scope

The repository currently contains two distinct layers:

- Stage 2: trajectory-only diffusion prior pre-training and evaluation
- Stage 3 Phase 1: indoor trajectory imputation benchmark with one contiguous missing span

The Stage 2 scientific question is narrow:

- Can a diffusion prior trained only on pedestrian motion capture the geometry, smoothness, and dynamic plausibility of ETH+UCY trajectories?
- Which filtering policy gives the best balance between motion realism and sample coverage?
- How do the four official variants compare when protocol and model are held fixed?

The current Stage 3 Phase 1 engineering question is intentionally minimal:

- Given a degraded coarse absolute indoor trajectory with one contiguous missing span, how well do simple completion baselines reconstruct the clean trajectory?
- How often do reconstructed trajectories violate a minimal occupancy-map geometry check?

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
- [`docs/stage3/`](docs/stage3): Stage 3 navigation, benchmark notes, and archive material
- [`变更记录.md`](变更记录.md): repository-level Chinese modification record
- [`outputs/prior/train/`](outputs/prior/train): training snapshots organized by variant and seed
- [`outputs/prior/sample/`](outputs/prior/sample): reverse-sampling artifacts organized to mirror train
- [`outputs/prior/eval/`](outputs/prior/eval): evaluation artifacts organized to mirror train
- `outputs/stage3/`: generated Stage 3 benchmark artifacts kept as local runtime outputs
- [`outputs/prior/variants/`](outputs/prior/variants): browsing entry point for the four official variants
- [`outputs/prior/archive/`](outputs/prior/archive): legacy reference outputs and folded historical material
- [`tools/prior/`](tools/prior): training, sampling, and evaluation entry points
- [`tools/stage3/`](tools/stage3): minimal data, baseline, and evaluation scripts for Stage 3 Phase 1
- [`tools/legacy/`](tools/legacy): older exploratory scripts retained for traceability
- [`utils/prior/`](utils/prior): registry and shared semantic helpers
- [`utils/stage3/`](utils/stage3): minimal Stage 3 path and IO helpers

## Stage 3 Phase 1 Boundary

Stage 3 Phase 1 is intentionally small and benchmark-first.

- input: degraded canonical room3 trajectory with one contiguous missing span
- output: completed canonical room3 trajectory
- baselines: linear interpolation, Savitzky-Golay, constant-velocity Kalman filter
- metrics: ADE, FDE, RMSE, masked_ADE, masked_RMSE, wall-crossing count, off-map ratio

The current canonical room3 data layer lives at:

- `outputs/stage3/phase1/canonical_room3/`

The room3 clean data is generated from `datasets/processed/data_eth_ucy_20.npy`
by separate linear scaling into `[0, 3] x [0, 3]`. The full Stage 3 Phase 1
room3 run currently contains 6 experiments and 18 method-result rows. The
machine-readable summary is:

- `outputs/stage3/phase1/canonical_room3/eval/summary_metrics.csv`
- `outputs/stage3/phase1/canonical_room3/eval/summary_report.md`

Metric interpretation:

- This benchmark reports two complementary metric views.
- Full-trajectory metrics, including `ADE`, `FDE`, and `RMSE`, measure overall trajectory consistency over the full window.
- Masked metrics, including `masked_ADE` and `masked_RMSE`, measure reconstruction quality on the removed segment itself.
- Since the task is missing-segment reconstruction, masked metrics are emphasized when discussing reconstruction quality on the missing span.
- When the two views rank methods differently, both rankings are reported explicitly rather than collapsed into a single overall winner.
- Geometry metrics remain boundary and geometry consistency checks: `off_map_ratio` and `wall_crossing_count`.

Data notes:

- the fixed-span sweep covers 10%, 20%, and 30% missing ratios
- the random-position control uses span20 with seeds `42`, `43`, and `44`
- masked metrics are computed only on the missing span
- under the clean missing-span setting, Linear interpolation can be strongest under full-trajectory metrics, while Savitzky-Golay can be slightly stronger under masked metrics
- this difference is expected because full-trajectory metrics include observed time steps and may dilute the error on the removed segment
- the empty room3 geometry check records wall crossings and off-map ratios
- the pushed artifact set is a benchmark data snapshot, not a raw data release

Random-span statistical reliability:

- the standard random-span sweep uses `span_ratio = 0.2`, `span_mode = random`, and seeds `0..19`
- per-seed metrics live in `outputs/stage3/phase1/canonical_room3/random_span_statistics/metrics_by_seed.csv`
- mean and std summaries live in `outputs/stage3/phase1/canonical_room3/random_span_statistics/metrics_summary_mean_std.csv`
- the report lives in `outputs/stage3/phase1/canonical_room3/random_span_statistics/random_span_statistics_report.md`
- the standard figures are:
  - `outputs/stage3/phase1/canonical_room3/random_span_statistics/figures/ADE_mean_std_bar.png`
  - `outputs/stage3/phase1/canonical_room3/random_span_statistics/figures/RMSE_mean_std_bar.png`
  - `outputs/stage3/phase1/canonical_room3/random_span_statistics/figures/masked_ADE_mean_std_bar.png`
  - `outputs/stage3/phase1/canonical_room3/random_span_statistics/figures/full_vs_masked_comparison.png`

The full-vs-masked comparison figure is the current standard visual summary for
showing whether Linear and Savitzky-Golay rank differently under overall
trajectory consistency and missing-segment reconstruction quality.

This phase does not yet include:

- prior integration
- q20 or multi-prior comparison
- learning-based backbones
- complex geometry conditioning
- room graph or door semantics
- multi-dataset comparison

For a more formal Stage 3 statement, use [`docs/stage3/planning/stage3_phase1_formal_spec.md`](docs/stage3/planning/stage3_phase1_formal_spec.md).

## Stage 3 Controlled Benchmark

In addition to the canonical room3 Phase 1 benchmark snapshot, the repository
now contains a controlled coarse-to-refined reconstruction pipeline under the
same room3 coordinate system.

The controlled benchmark starts from:

- `datasets/processed/data_eth_ucy_20.npy`

It normalizes the clean absolute trajectories into canonical room3 coordinates
and then applies four synthetic degradation settings:

- `missing_only`
- `missing_noise`
- `missing_drift`
- `missing_noise_drift`

The current default configuration is:

- `seed = 42`
- `span_ratio = 0.2`
- `span_mode = fixed`
- `noise_std = 0.03`
- `drift_amp = 0.05`

The generated outputs are organized as:

- `outputs/stage3/controlled_benchmark/degradation/`
- `outputs/stage3/controlled_benchmark/reconstruction/`
- `outputs/stage3/controlled_benchmark/eval/`
- `outputs/stage3/controlled_benchmark/figures/`

The machine-readable summary for the controlled benchmark is:

- `outputs/stage3/controlled_benchmark/eval/metrics_summary.csv`
- `outputs/stage3/controlled_benchmark/eval/metrics_summary.json`

The figure set currently includes:

- four representative trajectory comparison figures, one for each degradation
- bar charts for `ADE`, `RMSE`, and `masked_ADE` across degradation settings

The standard command entry points are:

- `.venv/bin/python -m tools.stage3.controlled.build_controlled_degradation`
- `.venv/bin/python -m tools.stage3.controlled.run_coarse_reconstruction_baselines`
- `.venv/bin/python -m tools.stage3.controlled.evaluate_coarse_reconstruction`

Compatibility wrappers are retained at the older `tools/stage3/*.py` locations
so older commands can still dispatch to the moved modules.

The current controlled benchmark is still intentionally simple:

- it reuses the same three baseline families: Linear, Savitzky-Golay, and Kalman
- it keeps the same full-trajectory, masked-segment, and geometry metric views
- it is meant as a reproducible engineering benchmark layer, not a new model family

## Change Record Rule

This repository now maintains a required Chinese modification record at [`变更记录.md`](变更记录.md).

For every future modification:

- record the time of the change
- record what was changed
- record what the previous state was
- if the change is code, it is sufficient to record the modified code file path

This file should be updated together with each future repository change, rather than filled retrospectively.

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

- `backup/full-snapshot-2026-04-22` recommended current backup target
- `backup/full-snapshot-2026-04-16` historical backup
- `backup/full-snapshot-2026-04-12` historical backup
- `backup/full-snapshot-2026-04-11` historical backup
- `backup/full-snapshot-2026-04-07` historical backup

If you do not see the backup in GitHub, switch branches in the repository view. The backup is not the default branch, so it will not appear unless you select it explicitly. New uploads should go to `backup/full-snapshot-2026-04-22`.

Use [`docs/archive/reference/BACKUP.md`](docs/archive/reference/BACKUP.md) for the shortest restore command and backup notes.
