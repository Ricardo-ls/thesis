# 2026-05-16 Stage 3 SDE Mainline Backup

This directory records the May 16, 2026 backup snapshot for the thesis repository.

## Backup Scope

This snapshot backs up the current Stage 3 SDE mainline state after the project cleanup:

- Stage 3 unconditional SDEdit / prior-guided refinement is the current mainline.
- Confidence-aware stabilization is the current stable SDE component.
- Multi-t candidate-source ablation and DPS work are kept as supplementary or negative diagnostics, not as mainline.
- Physical output cleanup has moved non-mainline outputs into curated archive directories.

## Main Navigation

Use these files first:

- `docs/stage3_sde/mainline_manifest.md`
- `docs/stage3_sde/narrative_alignment.md`
- `docs/stage3_sde/output_inventory.md`
- `docs/stage3_sde/path_mapping.md`
- `outputs/README.md`
- `变更记录.md`

## Curated Output Roots

- Mainline: `outputs/curated/stage3_sde/mainline/`
- Supplementary ablations: `outputs/curated/stage3_sde/supplementary_ablation/`
- Negative diagnostics: `outputs/curated/stage3_sde/negative_diagnostic/`
- Legacy/deprecated outputs: `outputs/curated/stage3_sde/legacy_or_deprecated/`

## Backup Branch

Intended backup branch:

- `backup/stage3-sde-mainline-2026-05-16`

Intended tag:

- `stage3-sde-mainline-2026-05-16`

## Notes

The seed `13000-13999` E3 hold-out results are treated as frozen analysis/validation evidence. Future adaptive SDE or sensor-interface changes should use a new independent hold-out and should not tune on this frozen set.
