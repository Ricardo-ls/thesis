# AUDIT_REPORT

Purpose: validate Stage 3 data-pipeline integrity before professor reporting, using file inspection and numpy spot checks only.

## Summary

- `Overall verdict`: **PASS** — No numeric inconsistencies > 1e-4 were found in the checked headline numbers. The main corrections are definitional clarity: span `[8,11]` is 4 frames inclusive, `masked_ADE` averages only missing frames, and `seed_best` must be treated as oracle diagnostic only.

Special points that must be stated explicitly:
- `Span [8:11]`: **PASS** — Actual missing frames are `8, 9, 10, 11` inclusive, so the missing length is 4, not 3.
- `masked_ADE denominator`: **PASS** — Denominator is the number of missing frames only, not all 20 frames.
- `variance_decomposition std definitions`: **PASS** — `std_across_trajectories` = mean over seeds of per-seed std across trajectories; `std_across_seeds` = mean over trajectories of per-trajectory std across seeds; `std_total` = std of all flattened values.
- `in-domain N=128 subset selection`: **PASS** — The quick in-domain experiment uses the first 128 trajectories by array order, not a random subset.

## Audit 1: Raw Data Confirmation

- `A. clean_windows_room3.npz structure`: **PASS** — keys=`['traj_abs', 'traj_rel']`, `traj_abs` shape=`(36073, 20, 2)`, dtype=`float32`, x range=`[0.0000, 3.0000]`, y range=`[0.0000, 3.0000]`.

Sample trajectory 334 full coordinates:
```text
t= 0: x=0.677162, y=1.656486
t= 1: x=0.701622, y=1.675592
t= 2: x=0.751830, y=1.677503
t= 3: x=0.804612, y=1.667950
t= 4: x=0.874131, y=1.662218
t= 5: x=0.923051, y=1.650754
t= 6: x=0.979696, y=1.639291
t= 7: x=1.040203, y=1.627827
t= 8: x=1.098135, y=1.624006
t= 9: x=1.161217, y=1.606811
t=10: x=1.220436, y=1.608721
t=11: x=1.279656, y=1.591526
t=12: x=1.340162, y=1.589615
t=13: x=1.400669, y=1.572420
t=14: x=1.463751, y=1.595347
t=15: x=1.522970, y=1.597258
t=16: x=1.578328, y=1.599168
t=17: x=1.640122, y=1.587705
t=18: x=1.707066, y=1.570509
t=19: x=1.775297, y=1.578152
```
- `B. sample 334 trajectory load`: **PASS** — Trajectory is readable and consistent with expected `(20, 2)` shape.
- `C. span [8:11] meaning`: **PASS** — For `span20_fixed_seed42`, stored `span_start=8`, `span_end=11` and `obs_mask` zeros at `[8, 9, 10, 11]`. The implementation uses `start : end + 1`, so the missing span length is `4`.

## Audit 2: Degraded Data Confirmation

- Experiment file: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/phase1/canonical_room3/data/experiments/span20_fixed_seed42/missing_span_windows.npz`
- Keys: `['traj_abs', 'traj_obs', 'obs_mask', 'span_start', 'span_end']`
- sample 334 obs_mask: `[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]`

Missing frames clean vs degraded values for sample 334:
```text
t=8: clean=(1.098135,1.624006), degraded=(nan,nan)
t=9: clean=(1.161217,1.606811), degraded=(nan,nan)
t=10: clean=(1.220436,1.608721), degraded=(nan,nan)
t=11: clean=(1.279656,1.591526), degraded=(nan,nan)
```
- `D. missing_only degradation semantics`: **PASS** — Observed frames are preserved and missing frames are set to `NaN` in `traj_obs`; `obs_mask` marks frames 8-11 as 0.

## Audit 3: Manual Linear Interpolation Recalculation

Manual interpolation for sample 334 on missing frames:
```text
t=8: pred=(1.100195,1.620185), clean=(1.098135,1.624006), L2 error=0.004341
t=9: pred=(1.160187,1.612543), clean=(1.161217,1.606811), L2 error=0.005824
t=10: pred=(1.220179,1.604900), clean=(1.220436,1.608721), L2 error=0.003830
t=11: pred=(1.280171,1.597258), clean=(1.279656,1.591526), L2 error=0.005755
```
- manual sample 334 masked_ADE = `0.004937347141`
- saved baseline sample 334 masked_ADE = `0.004937347025`
- max abs difference between manual pred and saved pred = `0.000000000000`
- global manual masked_ADE = `0.008583666756749`
- saved reconstruction_metrics.json masked_ADE = `0.008583666756749`
- `E. manual linear interpolation check`: **PASS** — Sample-level and global-level manual recomputation match saved baseline outputs exactly within machine precision.

## Audit 4: ADE vs masked_ADE Definition

Code source: `tools/stage3/eval/eval_reconstruction_metrics.py`

Relevant logic summary:
- `point_error = np.linalg.norm(diff, axis=-1)`
- `ADE = point_error.mean()`
- `FDE = point_error[:, -1].mean()`
- `masked_point_error = point_error[missing_mask]`
- `masked_ADE = masked_point_error.mean()`

- `F. ADE denominator`: **PASS** — ADE averages over all `N × T` point errors, so the per-trajectory denominator is `T=20`.
- `F. masked_ADE denominator`: **PASS** — masked_ADE averages only missing-frame errors. For `span20_fixed_seed42` the per-trajectory missing denominator is `4`.

## Audit 5: variance_decomposition Definition

Code source: `tools/stage3/refinement/run_inpainting_experiment.py`, function `build_var_rows`.

Definition confirmed from code:
- `std_total`: std of all flattened values across trajectories × seeds
- `std_across_trajectories`: for each seed, std over trajectories; then average those seed-wise std values
- `std_across_seeds`: for each trajectory, std over seeds; then average those trajectory-wise std values

Historical CSV row checked: `ddpm_v3_inpainting / missing_only / masked_ADE` = std_across_trajectories `0.1193481012701`, std_across_seeds `0.0836010182586`, std_total `0.1195199837572`.
- `G. variance_decomposition definition audit`: **PASS** — The exact definitions are unambiguous in code. Exact replay from saved historical per-seed raw arrays is not possible because those arrays were not written out, but the reported CSV and code definitions are internally consistent.

## Audit 6: Scale Mismatch Ratio

- ETH+UCY rel mean step = `0.241194099188`
- room3 clean mean step = `0.034470792860`
- ratio r = `6.997056904635`
- diagnosis scale_comparison.csv ratio = `6.997056904635`
- `H. scale mismatch ratio`: **PASS** — Manual recomputation reproduces the reported `r ≈ 6.997057`.

## Audit 7: In-domain Quick Sanity Check Data

- ETH+UCY abs shape = `(36073, 20, 2)`
- x range = `-7.690` to `15.613`
- y range = `-1.810` to `13.892`
- exp01 config max_trajectories = `128`
- exp01 config natural_scale = `True`
- exp01 config room3_used = `False`
- `I. natural-scale confirmation`: **PASS** — ETH+UCY absolute coordinates span far beyond `[0, 3]`, so the quick in-domain experiment uses natural scale, not room3-normalized scale.
- `I. N=128 subset selection`: **PASS** — Code path `load_absolute_trajectories(..., max_trajectories)` slices `traj_abs[:max_trajectories]`, so the subset is the first 128 trajectories in file order.

## Audit 8: Key Number Cross-Validation

| check | expected | recomputed | diff | status |
| --- | ---: | ---: | ---: | --- |
| linear_interp / missing_only / masked_ADE / mean | 0.012930 | 0.012931005445 | 0.000001005445 | PASS |
| linear_interp / missing_only / masked_ADE / std | 0.012990 | 0.012993169963 | 0.000003169963 | PASS |
| ddpm_v3 / missing_only / masked_ADE / mean | 0.123830 | 0.123826772878 | -0.000003227122 | PASS |
| ddpm_v3 / missing_only / std_across_seeds | 0.083600 | 0.083601018259 | 0.000001018259 | PASS |
| scale_ratio_r | 6.997000 | 6.997056904635 | 0.000056904635 | PASS |
| ddpm_v3_scaled / missing_only / masked_ADE | 0.013796 | 0.013796065934 | 0.000000065934 | PASS |
| ddpm_v3_anchored / missing_only / masked_ADE | 0.031920 | 0.031920347363 | 0.000000347363 | PASS |
| in_domain_linear / masked_ADE | 0.117882 | 0.117881974191 | -0.000000025809 | PASS |
| in_domain_anchored_v3 / masked_ADE | 0.126066 | 0.126066352149 | 0.000000352149 | PASS |

- `J. key-number checklist`: **PASS** — 9 PASS, 0 FAIL under tolerance `1e-4`.

## PASS/FAIL Summary

- `Raw data integrity`: **PASS**
- `Missing-span definition`: **PASS** — Inclusive `[8,11]`, 4 frames.
- `Degraded data semantics`: **PASS**
- `Manual linear interpolation recomputation`: **PASS**
- `ADE and masked_ADE denominator definitions`: **PASS**
- `variance_decomposition formula definitions`: **PASS**
- `Scale mismatch ratio`: **PASS**
- `In-domain natural-scale dataset claim`: **PASS**
- `Headline numeric cross-checks`: **PASS**

## Notes for Professor Report

- Do not describe `span20_fixed_seed42` as python slice `[8:11]` in the strict slicing sense. The implemented missing region is frames `8,9,10,11` inclusive.
- State explicitly that `masked_ADE` averages only missing-frame errors; its denominator is the missing-frame count, not 20.
- State explicitly that `seed_best` is an oracle diagnostic and not a fair deployment-level comparison row.
- For `variance_decomposition`, define the two std values exactly as in code; avoid paraphrases like “error bars over seeds” without the averaging order.
- For the quick in-domain ETH+UCY sanity check, say that `N=128` is the first 128 trajectories in file order.
