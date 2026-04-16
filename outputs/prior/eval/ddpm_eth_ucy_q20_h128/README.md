# `q20` Eval Archive

This directory stores evaluation artifacts for the `q20` variant and follows the same seed-labeled layout as training:

- `seed{train_seed}-{epoch_tag}/`
- `reference_seed{sample_seed}/`

Important:

- `train_seed` identifies the checkpoint source.
- `reference_seed42` is the fixed sampling / visualization protocol seed, not the training seed.
- The evaluation summary always corresponds to the matching sample run.
