# Documentation and Archive Catalog

This repository preserves both the current Stage 2 mainline evidence and the earlier files produced during project development. To keep the record interpretable, the documentation is organized into four categories:

1. Current mainline documents
2. Historical reference documents
3. Phase-specific working documents
4. Output archives and figure stores

## Current Mainline Documents

These are the documents that define the present Stage 2 interpretation.

- [`stage2_phaseA_multiseed_100epoch_report.md`](stage2_phaseA_multiseed_100epoch_report.md): current mainline result page for the completed 15-seed, multi-seed, 100-epoch screening result.
- [`prior_stage2.md`](prior_stage2.md): repository-level Stage 2 interpretation, registry context, and reading order.
- [`README.md`](README.md): documentation map for the repository.
- [`archive/README.md`](archive/README.md): archive layout and reading rules for folded documentation.

## Historical Reference Documents

These files remain in the repository for traceability and comparison, but they do not define the current interpretation layer.

- `single-seed` or `50`-epoch Stage 2 narrative fragments retained in `prior_stage2.md`
- legacy prior figures under `docs/assets/prior/`
- earlier Stage 2 summary figures and notes that predate the completed 15-seed sweep

## Phase-Specific Working Documents

These files were created during project progression and are useful as planning notes or transfer records. They should be read as staging material rather than current conclusions.

- [`archive/phase_working/stage2_gpt_briefing.md`](archive/phase_working/stage2_gpt_briefing.md): factual handoff note for documentation generation and traceability.
- [`archive/phase_working/multi_seed_stage2_plan.md`](archive/phase_working/multi_seed_stage2_plan.md): archived seed schedule and folder convention for the Phase A sweep.

## Output Archives

The archived outputs are separated by role:

- `outputs/prior/train/`: training snapshots and run notes
- `outputs/prior/sample/`: reverse-sampling artifacts and manifests
- `outputs/prior/eval/`: evaluation diagnostics and summary metrics
- `outputs/prior/archive/`: folded historical Stage 2 material, including the Phase A multi-seed archive

## Practical Reading Order

For a new reader, the recommended order is:

1. [`README.md`](README.md)
2. [`prior_stage2.md`](prior_stage2.md)
3. [`stage2_phaseA_multiseed_100epoch_report.md`](stage2_phaseA_multiseed_100epoch_report.md)
4. [`DOCUMENT_CATALOG.md`](DOCUMENT_CATALOG.md)

This order prioritizes the current mainline interpretation first, then the formal archive structure, and only then the project history.
