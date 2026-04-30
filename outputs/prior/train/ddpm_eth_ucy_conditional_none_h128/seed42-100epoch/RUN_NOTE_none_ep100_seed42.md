# conditional none variant - 100-epoch training snapshot

## Purpose

This file records the 100-epoch conditional DDPM training snapshot for the `none` variant.

## Training configuration

- variant: `none`
- epochs: `100`
- batch_size: `128`
- timesteps: `100`
- hidden_dim: `128`
- random_seed: `42`
- train_ratio: `0.8`
- lr: `0.001`
- span_start: `8`
- span_end_exclusive: `12`

## Final result

- best_epoch: `96`
- best_val_loss: `0.033284`
- best_model: `best_model.pt`
- last_model: `last_model.pt`
- loss_history: `loss_history.csv`
- loss_curve_png: `loss_curve_epoch10plus.png`
- loss_curve_svg: `loss_curve_epoch10plus.svg`
