# q30 variant - 100-epoch training snapshot

## Purpose

This file records the 100-epoch training snapshot for the `q30` variant under the unified Stage 2 protocol.

## Training configuration

- variant: `q30`
- epochs: `100`
- batch_size: `128`
- timesteps: `100`
- hidden_dim: `128`
- random_seed: `3`
- train_ratio: `0.8`
- lr: `0.001`

## Final result

- best_epoch: `96`
- best_val_loss: `0.101880`
- best_model: `best_model.pt`
- last_model: `last_model.pt`
- loss_history: `loss_history.csv`
- loss_curve_png: `loss_curve_epoch10plus.png`
- loss_curve_svg: `loss_curve_epoch10plus.svg`

## Notes

- The training run is stored in a seed-and-epoch-labeled directory.
- The loss curve starts from epoch 10 to match the repository plotting convention.
