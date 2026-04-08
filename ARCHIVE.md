# Archive Branches

This repository keeps separate branches for clean rollback and experiment archiving.

## Current Branch Roles

- `main`: active thesis codebase and documentation
- `backup/full-snapshot-2026-04-07`: full repository snapshot backup
- `archive/stage2-none-ep100-seed42`: archived none variant trained for 100 epochs

## Archived Snapshot Notes

The `archive/stage2-none-ep100-seed42` branch is intended to preserve the 100-epoch none training snapshot without modifying the older backup snapshot.

It should be used when you need to inspect or restore:

- `outputs/prior/train/ddpm_eth_ucy_none_h128/`
- the 100-epoch none training artifacts

