# q30 variant - 100-epoch training snapshot

## Purpose

This file records the 100-epoch training snapshot for the `q30` variant under the unified Stage 2 protocol.

## Branch

- archive/stage2-none-ep100-seed42

## Training configuration

- variant: `q30`
- epochs: `100`
- batch_size: `128`
- timesteps: `100`
- hidden_dim: `128`
- random_seed: `42`
- train_ratio: `0.8`
- lr: `1e-3`

## Final result

- best_epoch: `72`
- best_val_loss: `0.102625`
- best_model: `best_model.pt`
- last_model: `last_model.pt`
- loss_history: `loss_history.csv`

## Notes

- The training run converged stably and did not diverge.
- This snapshot is kept in a seed-labeled folder so it stays aligned with the `none`, `q10`, and `q20` directory layout.
