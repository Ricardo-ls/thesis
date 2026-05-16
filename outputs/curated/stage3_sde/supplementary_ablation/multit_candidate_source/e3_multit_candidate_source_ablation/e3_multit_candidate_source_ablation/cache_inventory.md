# E3 Multi-t Candidate Source Cache Inventory

Inventory-only pass. No SDEdit candidates were generated, no evaluation was run, and no existing E3 outputs were modified.

## A. Summary

- Python executable used: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/.venv_exp01/bin/python`
- Expected dataset: `data/stage4/e3_holdout_1000/clean_trajs.npy`
- Expected cache root: `outputs/stage4/e3_holdout1000_confidence_aware_sdedit`
- Raw .npy/.npz files scanned: `48`
- Complete 6 conditions x 3 t_start candidates: `True`
- All candidates shape [1000,20,2]: `True`
- Any missing candidate: `False`
- Any ambiguous candidate match: `False`
- Any invalid candidate: `False`
- Any candidate NaN/Inf: `False`

## B. Completeness Matrix

| condition | t1 | t2 | t3 | notes |
| --- | --- | --- | --- | --- |
| gaussian_medium | OK | OK | OK |  |
| drift_medium | OK | OK | OK |  |
| burst_medium | OK | OK | OK |  |
| bias_medium | OK | OK | OK |  |
| jump_medium | OK | OK | OK |  |
| combined_medium | OK | OK | OK |  |

## C. Detailed File Table

| path | inferred_condition | inferred_t_start | shape | dtype | finite_ok | numeric_min | numeric_max | numeric_mean | file_size_bytes | modified_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/bias_medium_sdedit_t1.npy | bias_medium | 1 | (1000, 20, 2) | float32 | True | -0.34429 | 3.22579 | 1.50593 | 160128 | 2026-05-15T16:37:41 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/bias_medium_sdedit_t2.npy | bias_medium | 2 | (1000, 20, 2) | float32 | True | -0.34429 | 3.23287 | 1.50601 | 160128 | 2026-05-15T16:37:49 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/bias_medium_sdedit_t3.npy | bias_medium | 3 | (1000, 20, 2) | float32 | True | -0.34429 | 3.24391 | 1.50609 | 160128 | 2026-05-15T16:38:00 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/burst_medium_sdedit_t1.npy | burst_medium | 1 | (1000, 20, 2) | float32 | True | -0.658906 | 3.41561 | 1.50906 | 160128 | 2026-05-15T16:37:17 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/burst_medium_sdedit_t2.npy | burst_medium | 2 | (1000, 20, 2) | float32 | True | -0.620524 | 3.3976 | 1.50857 | 160128 | 2026-05-15T16:37:25 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/burst_medium_sdedit_t3.npy | burst_medium | 3 | (1000, 20, 2) | float32 | True | -0.581098 | 3.388 | 1.50804 | 160128 | 2026-05-15T16:37:36 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/combined_medium_sdedit_t1.npy | combined_medium | 1 | (1000, 20, 2) | float32 | True | -0.295841 | 3.31138 | 1.50806 | 160128 | 2026-05-15T16:38:29 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/combined_medium_sdedit_t2.npy | combined_medium | 2 | (1000, 20, 2) | float32 | True | -0.28451 | 3.31427 | 1.50798 | 160128 | 2026-05-15T16:38:37 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/combined_medium_sdedit_t3.npy | combined_medium | 3 | (1000, 20, 2) | float32 | True | -0.273173 | 3.30966 | 1.50789 | 160128 | 2026-05-15T16:38:48 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/drift_medium_sdedit_t1.npy | drift_medium | 1 | (1000, 20, 2) | float32 | True | 0.0445807 | 2.99412 | 1.50936 | 160128 | 2026-05-15T16:36:53 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/drift_medium_sdedit_t2.npy | drift_medium | 2 | (1000, 20, 2) | float32 | True | 0.0552523 | 2.99362 | 1.50942 | 160128 | 2026-05-15T16:37:01 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/drift_medium_sdedit_t3.npy | drift_medium | 3 | (1000, 20, 2) | float32 | True | 0.0649761 | 2.983 | 1.50947 | 160128 | 2026-05-15T16:37:12 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/gaussian_medium_sdedit_t1.npy | gaussian_medium | 1 | (1000, 20, 2) | float32 | True | -0.0122278 | 2.98482 | 1.50952 | 160128 | 2026-05-15T16:36:30 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/gaussian_medium_sdedit_t2.npy | gaussian_medium | 2 | (1000, 20, 2) | float32 | True | -0.0144355 | 2.98608 | 1.50943 | 160128 | 2026-05-15T16:36:38 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/gaussian_medium_sdedit_t3.npy | gaussian_medium | 3 | (1000, 20, 2) | float32 | True | -0.021573 | 2.98601 | 1.50934 | 160128 | 2026-05-15T16:36:48 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/jump_medium_sdedit_t1.npy | jump_medium | 1 | (1000, 20, 2) | float32 | True | -0.307802 | 3.33984 | 1.50913 | 160128 | 2026-05-15T16:38:06 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/jump_medium_sdedit_t2.npy | jump_medium | 2 | (1000, 20, 2) | float32 | True | -0.310546 | 3.33938 | 1.5091 | 160128 | 2026-05-15T16:38:13 |
| outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/jump_medium_sdedit_t3.npy | jump_medium | 3 | (1000, 20, 2) | float32 | True | -0.30953 | 3.33354 | 1.50907 | 160128 | 2026-05-15T16:38:24 |

## Companion Array Inventory

| condition | kind | found | path | shape_ok | finite_ok | ambiguous | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | clean | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/gaussian_medium_clean.npy | True | True | False |  |
| gaussian_medium | degraded | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/gaussian_medium_degraded.npy | True | True | False |  |
| gaussian_medium | confidence | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/gaussian_medium_confidence.npy | True | True | False |  |
| gaussian_medium | fused | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/gaussian_medium_fused_best_tau07_gamma2.npy; outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/gaussian_medium_fused_t1_tau07_gamma2.npy | False | False | True | ambiguous companion matches |
| drift_medium | clean | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/drift_medium_clean.npy | True | True | False |  |
| drift_medium | degraded | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/drift_medium_degraded.npy | True | True | False |  |
| drift_medium | confidence | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/drift_medium_confidence.npy | True | True | False |  |
| drift_medium | fused | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/drift_medium_fused_best_tau07_gamma2.npy; outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/drift_medium_fused_t1_tau07_gamma2.npy | False | False | True | ambiguous companion matches |
| burst_medium | clean | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/burst_medium_clean.npy | True | True | False |  |
| burst_medium | degraded | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/burst_medium_degraded.npy | True | True | False |  |
| burst_medium | confidence | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/burst_medium_confidence.npy | True | True | False |  |
| burst_medium | fused | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/burst_medium_fused_best_tau07_gamma2.npy; outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/burst_medium_fused_t1_tau07_gamma2.npy | False | False | True | ambiguous companion matches |
| bias_medium | clean | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/bias_medium_clean.npy | True | True | False |  |
| bias_medium | degraded | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/bias_medium_degraded.npy | True | True | False |  |
| bias_medium | confidence | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/bias_medium_confidence.npy | True | True | False |  |
| bias_medium | fused | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/bias_medium_fused_best_tau07_gamma2.npy; outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/bias_medium_fused_t1_tau07_gamma2.npy | False | False | True | ambiguous companion matches |
| jump_medium | clean | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/jump_medium_clean.npy | True | True | False |  |
| jump_medium | degraded | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/jump_medium_degraded.npy | True | True | False |  |
| jump_medium | confidence | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/jump_medium_confidence.npy | True | True | False |  |
| jump_medium | fused | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/jump_medium_fused_best_tau07_gamma2.npy; outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/jump_medium_fused_t1_tau07_gamma2.npy | False | False | True | ambiguous companion matches |
| combined_medium | clean | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/combined_medium_clean.npy | True | True | False |  |
| combined_medium | degraded | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/combined_medium_degraded.npy | True | True | False |  |
| combined_medium | confidence | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/combined_medium_confidence.npy | True | True | False |  |
| combined_medium | fused | True | outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/combined_medium_fused_best_tau07_gamma2.npy; outputs/stage4/e3_holdout1000_confidence_aware_sdedit/arrays/combined_medium_fused_t1_tau07_gamma2.npy | False | False | True | ambiguous companion matches |

## D. Required Next Action

READY_FOR_EVAL

All t1/t2/t3 candidates are present and valid.
