# Stage 3 Phase 1 Figures

## Source

The figures are generated from the existing Stage 3 Phase 1 canonical room3
summary table:

- `outputs/stage3/phase1/canonical_room3/eval/summary_metrics.csv`

The plotting script reads the CSV only. It does not rerun experiments or modify
benchmark logic.

## Figures

Figure 1 compares Experiment 0 baseline errors for:

- `span20_fixed_seed42`
- `ADE`
- `masked_ADE`

Figure 2 shows the Experiment 1 missing-span sweep for:

- `span10_fixed_seed42`
- `span20_fixed_seed42`
- `span30_fixed_seed42`
- `masked_ADE`

Interpretation note:

- `ADE` in Figure 1 is the full-trajectory view.
- `masked_ADE` is the missing-segment view.
- Since the task is missing-segment reconstruction, `masked_ADE` should be
  emphasized when discussing recovery quality on the removed span.
- If `ADE` and `masked_ADE` rank methods differently, both views should be
  reported.

## Command

Run from the `SingularTrajectory/` repository root:

```bash
python tools/stage3/eval/plot_phase1_figures.py
```

## Outputs

The output directory is:

- `outputs/stage3/phase1/canonical_room3/figures/`

Generated files:

- `outputs/stage3/phase1/canonical_room3/figures/exp0_baseline_ade_maskedade_bar.png`
- `outputs/stage3/phase1/canonical_room3/figures/exp0_baseline_ade_maskedade_bar.pdf`
- `outputs/stage3/phase1/canonical_room3/figures/exp1_span_sweep_maskedade_line.png`
- `outputs/stage3/phase1/canonical_room3/figures/exp1_span_sweep_maskedade_line.pdf`
