# Stage 3 Phase 1 Random-Span Statistics

## Purpose

Random-span statistics test whether the current canonical_room3 benchmark behavior remains stable across many missing-span positions instead of depending on one fixed gap placement.

## Protocol

- `span_ratio = 0.2`
- `span_mode = random`
- `seeds = 0..19`
- baselines: `linear_interp`, `savgol_w5_p2`, `kalman_cv_dt1.0_q1e-3_r1e-2`

## Metric Interpretation

This benchmark reports two complementary evaluation views. Full-trajectory metrics measure overall trajectory consistency, while masked metrics measure reconstruction quality on the missing segment. When the two views rank methods differently, both rankings are reported explicitly.

Full-trajectory metrics, including ADE, FDE, and RMSE, measure overall trajectory consistency over the full window. Masked metrics, including masked_ADE and masked_RMSE, measure reconstruction quality on the removed segment itself. Since the task is missing-segment reconstruction, masked metrics are emphasized when discussing reconstruction quality on the missing span. When the two views rank methods differently, both rankings are reported explicitly rather than collapsed into a single overall ranking.

## Mean +- Std Results

| method | ADE | RMSE | masked_ADE | masked_RMSE |
| --- | --- | --- | --- | --- |
| linear_interp | 0.001724 +- 0.000012 | 0.004099 +- 0.000015 | 0.008622 +- 0.000062 | 0.009167 +- 0.000034 |
| savgol_w5_p2 | 0.002509 +- 0.000011 | 0.004155 +- 0.000014 | 0.008321 +- 0.000061 | 0.008931 +- 0.000035 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | 0.009298 +- 0.000019 | 0.014709 +- 0.000051 | 0.027723 +- 0.000122 | 0.031112 +- 0.000125 |

## Ranking

- Full-trajectory view by mean ADE: `linear_interp, savgol_w5_p2, kalman_cv_dt1.0_q1e-3_r1e-2`
- Missing-segment view by mean masked_ADE: `savgol_w5_p2, linear_interp, kalman_cv_dt1.0_q1e-3_r1e-2`

- The full-trajectory view and the missing-segment view rank Linear and Savitzky-Golay differently, so both rankings should be retained in the interpretation.

## Conclusion

This random-span sweep strengthens statistical reliability relative to a single fixed missing-span position. It should be read as a stability check for the current Phase 1 benchmark, not as a reason to collapse the two metric views into one overall winner.
