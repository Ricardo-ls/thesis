# `none` Eval Archive

This directory stores evaluation artifacts for the `none` variant and is aligned with the training archive structure:

- `seed{train_seed}-{epoch_tag}/`
- `reference_seed{sample_seed}/`

Important:

- `train_seed` identifies the checkpoint source.
- `reference_seed42` is the fixed sampling / visualization protocol seed, not the training seed.
- The evaluation summary reflects the exact sample output under the matching `sample/` run directory.
