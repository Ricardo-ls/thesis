# none variant — 150-epoch training snapshot

## Purpose
This file marks the formal 150-epoch training update for the `none` variant on the backup snapshot branch.

## Branch
- backup/full-snapshot-2026-04-07

## Training configuration
- variant: none
- epochs: 150
- batch_size: 128
- timesteps: 100
- hidden_dim: 128
- random_seed: 42
- train_ratio: 0.8
- lr: 1e-3

## Final result
- best_epoch: 116
- best_val_loss: 0.086884

## Files included in this commit
- best_model.pt
- last_model.pt
- loss_history.csv
- RUN_NOTE_none_ep150_seed42.md

## Excluded from this commit
- sample outputs under `outputs/prior/sample/ddpm_eth_ucy_none_h128/formal_ep150_seed42`
- evaluation outputs under `outputs/prior/eval/ddpm_eth_ucy_none_h128/formal_ep150_seed42`

## Note
This upload is intended as a labeled backup snapshot entry rather than a public lightweight reference artifact.
