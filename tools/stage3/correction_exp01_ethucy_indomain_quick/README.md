# exp01_ethucy_indomain_quick

Isolated Stage 3 correction experiment scaffold for a quick in-domain ETH+UCY sanity check in natural coordinate scale.

Run from repository root:

```bash
python -m tools.stage3.correction_exp01_ethucy_indomain_quick.run_exp01
```

This directory only adds the experiment runner and helper utilities. Generated outputs are written to:

`outputs/stage3/correction_exp01_ethucy_indomain_quick/`

Scope:

- use existing ETH+UCY processed public trajectories
- use existing Stage 2 prior checkpoint
- evaluate `missing_only` only
- do not retrain
- do not modify DDPM architecture, v3 inpainting logic, alpha rule, or baseline methods
- do not use Room3
- do not apply global coordinate scaling
