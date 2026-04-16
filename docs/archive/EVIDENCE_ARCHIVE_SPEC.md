# Evidence Archive Specification

This specification defines the repository evidence archive and the reading permissions for each layer.

## Scope

The repository is organized into the following documentation and evidence layers:

- current mainline
- current evidence
- legacy reference
- phase-working provenance
- compatibility aliases

## Current Mainline

The current mainline is fixed to the completed 15-seed, multi-seed, 100-epoch Phase A sweep.

Permitted current mainline documents:

- [`../stage2_phaseA_multiseed_100epoch_report.md`](../stage2_phaseA_multiseed_100epoch_report.md)
- [`../prior_stage2.md`](../prior_stage2.md)
- [`../README.md`](../README.md)
- [`../DOCUMENT_CATALOG.md`](../DOCUMENT_CATALOG.md)

## Current Evidence

Current evidence consists of files that directly support the current mainline interpretation.

Permitted current evidence files:

- current mainline result pages
- current training, sampling, and evaluation outputs under `outputs/prior/train/`, `outputs/prior/sample/`, and `outputs/prior/eval/`
- readme files that describe the current evidence layout

## Legacy Reference

Legacy reference consists of files retained for historical comparison, traceability, or recovery.

Legacy reference files include:

- `outputs/prior/archive/`
- `docs/archive/reference/`
- legacy figures and historical narrative fragments

Legacy reference files may be cited only to explain historical context. They do not define the current evidence layer.

## Phase-Working Provenance

Phase-working provenance consists of planning notes, handoff briefs, and development-time documents produced during the project.

Phase-working provenance files include:

- `docs/archive/phase_working/`
- planning briefs
- seed schedule notes

Phase-working provenance may be cited for process context only. It must not be used as the basis for new planning or current conclusions.

## Compatibility Aliases

The following aliases are preserved for historical compatibility:

- `optimization_best -> none`
- `motion_balanced -> q20`

These aliases may be mentioned only as compatibility references. They do not override the current mainline or current evidence layers.

## Citation Rule

The current mainline may cite:

- current mainline documents
- current evidence outputs
- legacy reference material only when explicitly describing historical comparison

The current mainline must not cite:

- phase-working provenance as evidence
- archive paths as current mainline evidence
- compatibility aliases as current planning terms

## Reading Rule

For the present Stage 2 interpretation, read in this order:

1. `README.md`
2. `docs/README.md`
3. `docs/prior_stage2.md`
4. `docs/stage2_phaseA_multiseed_100epoch_report.md`
5. `docs/DOCUMENT_CATALOG.md`
6. `docs/archive/README.md`
7. `docs/archive/EVIDENCE_ARCHIVE_SPEC.md`

