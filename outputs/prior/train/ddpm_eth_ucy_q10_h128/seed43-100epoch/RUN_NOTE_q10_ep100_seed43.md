# q10 variant - 100-epoch training snapshot

## Purpose

This file records the 100-epoch training snapshot for the `q10` variant under the unified Stage 2 protocol.

## Branch

- backup/full-snapshot-2026-04-11

## Training configuration

- variant: `q10`
- epochs: `100`
- batch_size: `128`
- timesteps: `100`
- hidden_dim: `128`
- random_seed: `43`
- train_ratio: `0.8`
- lr: `1e-3`

## Final result

- best_model: `best_model.pt`
- last_model: `last_model.pt`
- loss_history: `loss_history.csv`

## Notes

- The training run converged stably and did not diverge.
- This snapshot is kept in a seed-labeled folder so it stays aligned with the multi-seed layout.
