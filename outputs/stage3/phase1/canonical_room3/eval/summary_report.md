# Stage 3 Phase 1 Canonical Room3 Summary

Protocol: canonical room3, separate scaling to `[0, 3] x [0, 3]`.

Summary CSV: `outputs/stage3/phase1/canonical_room3/eval/summary_metrics.csv`

## Experiment 0: Main Table

| experiment_id | method_tag | ADE | RMSE | masked_ADE | masked_RMSE | off_map_ratio | wall_crossing_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| span20_fixed_seed42 | linear_interp | 0.001717 | 0.004086 | 0.008584 | 0.009136 | 0.000000 | 0 |
| span20_fixed_seed42 | savgol_w5_p2 | 0.002510 | 0.004137 | 0.008262 | 0.008884 | 0.000003 | 0 |
| span20_fixed_seed42 | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.009910 | 0.015801 | 0.031486 | 0.033783 | 0.000064 | 0 |

## Experiment 1: Missing-Length Sweep

| experiment_id | method_tag | ADE | RMSE | masked_ADE | masked_RMSE | off_map_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| span20_fixed_seed42 | linear_interp | 0.001717 | 0.004086 | 0.008584 | 0.009136 | 0.000000 |
| span20_fixed_seed42 | savgol_w5_p2 | 0.002510 | 0.004137 | 0.008262 | 0.008884 | 0.000003 |
| span20_fixed_seed42 | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.009910 | 0.015801 | 0.031486 | 0.033783 | 0.000064 |
| span10_fixed_seed42 | linear_interp | 0.000391 | 0.001475 | 0.003905 | 0.004666 | 0.000000 |
| span10_fixed_seed42 | savgol_w5_p2 | 0.001248 | 0.001762 | 0.003538 | 0.004218 | 0.000003 |
| span10_fixed_seed42 | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.006320 | 0.008219 | 0.019599 | 0.020179 | 0.000011 |
| span30_fixed_seed42 | linear_interp | 0.004196 | 0.008001 | 0.013988 | 0.014608 | 0.000000 |
| span30_fixed_seed42 | savgol_w5_p2 | 0.004927 | 0.007996 | 0.013719 | 0.014446 | 0.000003 |
| span30_fixed_seed42 | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.016366 | 0.027287 | 0.044822 | 0.049256 | 0.000155 |

## Experiment 2: Missing-Position Control

| experiment_id | method_tag | ADE | RMSE | masked_ADE | masked_RMSE | off_map_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| span20_fixed_seed42 | linear_interp | 0.001717 | 0.004086 | 0.008584 | 0.009136 | 0.000000 |
| span20_fixed_seed42 | savgol_w5_p2 | 0.002510 | 0.004137 | 0.008262 | 0.008884 | 0.000003 |
| span20_fixed_seed42 | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.009910 | 0.015801 | 0.031486 | 0.033783 | 0.000064 |
| span20_random_seed42 | linear_interp | 0.001720 | 0.004092 | 0.008602 | 0.009150 | 0.000000 |
| span20_random_seed42 | savgol_w5_p2 | 0.002504 | 0.004148 | 0.008302 | 0.008917 | 0.000001 |
| span20_random_seed42 | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.009310 | 0.014717 | 0.027780 | 0.031130 | 0.000039 |
| span20_random_seed43 | linear_interp | 0.001714 | 0.004104 | 0.008572 | 0.009177 | 0.000000 |
| span20_random_seed43 | savgol_w5_p2 | 0.002500 | 0.004160 | 0.008273 | 0.008944 | 0.000003 |
| span20_random_seed43 | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.009340 | 0.014759 | 0.027950 | 0.031234 | 0.000076 |
| span20_random_seed44 | linear_interp | 0.001714 | 0.004071 | 0.008569 | 0.009104 | 0.000000 |
| span20_random_seed44 | savgol_w5_p2 | 0.002498 | 0.004128 | 0.008268 | 0.008869 | 0.000003 |
| span20_random_seed44 | kalman_cv_dt1.0_q1e-3_r1e-2 | 0.009303 | 0.014678 | 0.027735 | 0.031032 | 0.000050 |

## Random Position Mean

| method_tag | fixed_masked_ADE | random_mean_masked_ADE | fixed_masked_RMSE | random_mean_masked_RMSE |
| --- | --- | --- | --- | --- |
| linear_interp | 0.008584 | 0.008581 | 0.009136 | 0.009144 |
| savgol_w5_p2 | 0.008262 | 0.008281 | 0.008884 | 0.008910 |
| kalman_cv_dt1.0_q1e-3_r1e-2 | 0.031486 | 0.027822 | 0.033783 | 0.031132 |

## Masked ADE Reference

| experiment_id | best_by_masked_ADE | masked_ADE | masked_RMSE |
| --- | --- | --- | --- |
| span20_fixed_seed42 | savgol_w5_p2 | 0.008262 | 0.008884 |
| span10_fixed_seed42 | savgol_w5_p2 | 0.003538 | 0.004218 |
| span30_fixed_seed42 | savgol_w5_p2 | 0.013719 | 0.014446 |
| span20_random_seed42 | savgol_w5_p2 | 0.008302 | 0.008917 |
| span20_random_seed43 | savgol_w5_p2 | 0.008273 | 0.008944 |
| span20_random_seed44 | savgol_w5_p2 | 0.008268 | 0.008869 |

## Data Notes

- The CSV contains 18 rows: 6 experiments multiplied by 3 baseline methods.
- The fixed-span sweep records higher masked reconstruction errors as the missing ratio moves from 10% to 30%.
- The random-position controls for seeds 42, 43, and 44 have close metric values under the current proxy data.
- The empty room3 geometry check reports zero wall crossings for all rows.
- Off-map ratios are zero or very small, which is expected for normalized room3 trajectories with an empty `[0, 3] x [0, 3]` map.
