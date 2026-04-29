# CHANGELOG

## 2026-04-29

- created isolated experiment scaffold `tools/stage3/correction_exp01_ethucy_indomain_quick/`
- added `run_exp01.py` as the runnable entry point
- added `utils.py` for isolated natural-scale Stage 3 evaluation helpers
- added local experiment `README.md`
- purpose: quick in-domain sanity check for Stage 3 missing reconstruction on ETH+UCY public trajectories using an existing Stage 2 prior checkpoint
- planned methods: `linear_interp`, `savgol_w5_p2` when available, `kalman_cv_dt1.0_q1e-3_r1e-2`, `ddpm_v3_inpainting`, `ddpm_v3_inpainting_anchored`
- outputs are generated only under `outputs/stage3/correction_exp01_ethucy_indomain_quick/`
- if a method dependency or an exact intermediate output is unavailable at runtime, the runner records that explicitly in generated output documentation instead of inventing results
