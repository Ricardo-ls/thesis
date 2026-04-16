# `none` Sample Archive

This directory stores sampled artifacts for the `none` variant and is aligned with the training archive structure:

- `seed{train_seed}-{epoch_tag}/`
- `reference_seed{sample_seed}/`

Important:

- `train_seed` identifies the checkpoint source.
- `reference_seed42` is the fixed sampling / visualization protocol seed, not the training seed.
- Each run writes into its own seed-labeled folder so sample output stays aligned with the corresponding checkpoint.
