# Stage 3 Inpainting Experiment — Hypothesis Validation Report

*Generated automatically. Experiment completed in 35 s.*

---

## §1 Hypothesis

**One-sentence hypothesis:**
> The DDPM prior fails for trajectory reconstruction because it is trained *unconditionally*
> and is applied *outside* the reverse sampling loop (v1/v2), preventing the missing segment
> from being conditioned on the observed context during generation.

**Why inpainting-style sampling should fix this:**
RePaint (v3) clamps the observed relative displacements back to their forward-diffused level
at every reverse step (t → t−1). This forces the reverse chain to remain consistent with the
observed frames throughout the entire denoising trajectory, rather than generating a
context-free sample and only applying the mask post-hoc (v1) or blending it weakly (v2).

---

## §2 Method Rationale

| Method | Category | Description |
| --- | --- | --- |
| Linear | Classical baseline | Linear interpolation across the missing span |
| Savitzky-Golay | Classical baseline | SG filter with w=5, p=2, then gap-fill |
| Kalman | Classical baseline | Constant-velocity Kalman filter |
| DDPM v1 | Learned (post-hoc) | v0 single-shot projection → hard-replace missing |
| DDPM v2 | Learned (post-hoc) | v0 projection → blend α=0.1 on missing |
| DDPM v3 | Learned (inpainting) | RePaint-style: clamp known frames at every reverse step |

**v3 known limitations:**
- The prior was trained on ETH+UCY relative displacements; room3 coordinates are a
  linearly rescaled version of the same data. Any distributional shift between training
  and test coordinate scales will affect all DDPM variants equally.
- The prior is *unconditional* — it models the marginal p(trajectory), not
  p(trajectory | start, end). v3 constrains the observed steps but does not inject
  endpoint information in a structured way.
- `num_samples_per_traj=5` seeds are reported; spread estimates may be noisy.

---

## §3 Experiment Setup

- **Data source:** `datasets/processed/data_eth_ucy_20.npy` normalized to
  canonical room3 [0,3]×[0,3] via `build_imputation_dataset.py`.
- **N trajectories used:** 1024 (first 1024 of 36073;
  chosen to keep wall-clock time under 10 min on CPU).
- **Degradation pipeline:** 4 synthetic settings (missing_only, missing_noise,
  missing_drift, missing_noise_drift) with fixed span [8,11] (4 frames / 20%
  of T=20), seed=42.
- **Metrics:** ADE, RMSE (full trajectory); masked_ADE, masked_RMSE (missing
  segment only); wall_crossing_count, off_map_ratio (geometry, empty-room).
- **Averaging:** For deterministic methods (baselines, v1, v2), one value per
  trajectory → statistics across N trajectories. For v3, each trajectory has
  5 independent reverse-process samples → per-trajectory mean reported
  in the primary table; raw (N, S) values used for variance decomposition.
- **Geometry declaration:** `wall_crossing_count` and `off_map_ratio` are
  *evaluation-only* metrics. No geometry conditioning or loss term is used in
  any method. Empty room3 has no internal walls, so wall_crossing_count = 0
  for all methods by construction.

---

## §4 Full Results

### masked_ADE by method and degradation (primary metric)

| Method | Missing Only | Missing + Noise | Missing + Drift | Missing + Noise + Drift |
| --- | --- | --- | --- | --- |
| Linear | 0.01293 | 0.03382 | 0.03102 | 0.04279 |
| Savitzky-Golay | 0.01257 | 0.03548 | 0.03086 | 0.04411 |
| Kalman | 0.04218 | 0.06488 | 0.05229 | 0.06970 |
| DDPM v1 (masked-replace) | 0.12035 | 0.11956 | 0.12453 | 0.12375 |
| DDPM v2 (blend α=0.1) | 0.01965 | 0.03553 | 0.03432 | 0.04443 |
| DDPM v3 (RePaint inpainting) | 0.12383 | 0.15219 | 0.12902 | 0.15506 |

### Best method per degradation (masked_ADE)

| Degradation | Best Method | masked_ADE |
| --- | --- | --- |
| Missing Only | Savitzky-Golay | 0.01257 |
| Missing + Noise | Linear | 0.03382 |
| Missing + Drift | Savitzky-Golay | 0.03086 |
| Missing + Noise + Drift | Linear | 0.04279 |

*(See `full_results.csv` for complete statistics including ADE, RMSE, masked_RMSE,*
 *wall_crossing_count, off_map_ratio, n_trajectories, n_seeds, n_total for every*
 *method × degradation × metric combination.)*

---

## §5 Figures

Trajectory visualisations are in `trajectory_plots/`. Five cases are shown,
selected at the p10, p25, p50, p75, p90 percentiles of v3 masked_ADE on
`missing_only` degradation (to avoid cherry-picking).

Each figure shows 5 columns:
- **Col 1** – Clean target (ground truth, with missing segment dotted)
- **Col 2** – Degraded input (observed frames only)
- **Col 3** – Coarse reconstruction (Linear interpolation)
- **Col 4** – v3 single sample (seed 0)
- **Col 5** – v3 mean over 5 seeds (thin lines = individual samples,
  thick line = mean)

Figure annotation: `sample_idx`, `span_start:span_end`, `masked_ADE_coarse`,
`masked_ADE_v3`, `improvement_pct` (negative = worse).

**Figure 3 / Figure 5 spread note:**
The `std_across_seeds` column in `variance_decomposition.csv` directly quantifies
within-trajectory spread across the 5 reverse-diffusion seeds. This answers the
question of whether observed spread in trajectory figures reflects population
heterogeneity (std_across_trajectories) or DDPM stochasticity (std_across_seeds).

---

## §6 Discussion

### Verdict: hypothesis is **NOT SUPPORTED**

*v3 outperforms v1 in 0/4 degradations and v2 in 0/4 degradations on masked_ADE.*

**Interpretation (hypothesis not supported):**
The inpainting-style conditioning did not consistently improve over the post-hoc
projection baselines (v1/v2). Two plausible next-level explanations:

1. **Coordinate-scale domain mismatch.** The prior was trained on ETH+UCY
   relative displacements. Room3 data is the same dataset rescaled to [0,3]×[0,3].
   The scale of relative steps in room3 differs from the training distribution.
   Even with perfect inpainting conditioning, the prior's generative mode may not
   produce room3-scale displacements. Fix: retrain on room3-scale data, or
   normalize inputs to the prior's training distribution before inpainting.

2. **Short sequence / large missing fraction.** With T=20 and span_len=4 (20%),
   the observed context is limited (8 frames on one side, 8 on the other). A
   simple interpolation (Linear) that directly uses the boundary endpoints will
   dominate. The DDPM prior adds unnecessary distributional noise on top of what
   is essentially a short-range interpolation problem.

---

## §7 Next Hypothesis

**Next hypothesis:** The primary failure mode is coordinate-scale domain mismatch.
Normalizing room3 relative displacements to the ETH+UCY training distribution
(mean / std standardization using training statistics) before calling the DDPM
prior, and denormalizing afterward, should substantially improve inpainting v3.

*Proposed validation:* Compute mean and std of relative steps from
`data_eth_ucy_20_rel.npy`; apply z-score normalization to room3 inputs before
the DDPM reverse chain; apply inverse normalization to outputs. Re-run this
experiment and compare masked_ADE before and after normalization.

---

## §8 Variance Decomposition (masked_ADE for DDPM methods)

| Method | Degradation | std_across_trajectories | std_across_seeds | std_total |
| --- | --- | --- | --- | --- |
| DDPM v1 (masked-replace) | Missing Only | 0.06437 | nan | 0.06437 |
| DDPM v2 (blend α=0.1) | Missing Only | 0.01527 | nan | 0.01527 |
| DDPM v3 (RePaint inpainting) | Missing Only | 0.11935 | 0.08360 | 0.11952 |
| DDPM v1 (masked-replace) | Missing + Noise | 0.06597 | nan | 0.06597 |
| DDPM v2 (blend α=0.1) | Missing + Noise | 0.01789 | nan | 0.01789 |
| DDPM v3 (RePaint inpainting) | Missing + Noise | 0.11997 | 0.08396 | 0.12014 |
| DDPM v1 (masked-replace) | Missing + Drift | 0.06517 | nan | 0.06517 |
| DDPM v2 (blend α=0.1) | Missing + Drift | 0.01801 | nan | 0.01801 |
| DDPM v3 (RePaint inpainting) | Missing + Drift | 0.11798 | 0.08245 | 0.11816 |
| DDPM v1 (masked-replace) | Missing + Noise + Drift | 0.06676 | nan | 0.06676 |
| DDPM v2 (blend α=0.1) | Missing + Noise + Drift | 0.02192 | nan | 0.02192 |
| DDPM v3 (RePaint inpainting) | Missing + Noise + Drift | 0.11993 | 0.08359 | 0.12011 |

*(std_across_seeds = nan for deterministic methods v1 and v2)*
