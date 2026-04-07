# Backup Snapshot

This repository has a full backup branch for rollback and recovery:

- `backup/full-snapshot-2026-04-07`

What it contains:

- code
- datasets
- processed data
- training outputs
- checkpoints
- public Stage 2 reference assets

When to use it:

- recover a file that was removed or changed
- restore a previous experiment output
- inspect the repository exactly as it was at the backup snapshot

## Shortest Restore Command

Restore a single file from the backup branch:

```bash
git checkout backup/full-snapshot-2026-04-07 -- path/to/file
```

Examples:

```bash
git checkout backup/full-snapshot-2026-04-07 -- docs/prior_stage2.md
git checkout backup/full-snapshot-2026-04-07 -- outputs/prior/eval/ddpm_eth_ucy_q20_h128/reference_seed42/summary_metrics.csv
```

## Notes

- The backup branch is for rollback safety.
- The main research line remains on `main`.
- The backup branch is not the default working branch for Stage 2 development.
