# E3 Multi-t Per-Condition Consistency Extraction

## A. Purpose And Constraints

This is a read-only extraction from existing Stage 4 E3 multi-t candidate-source ablation outputs. It does not rerun SDEdit, generate candidates, train, use DPS, modify checkpoints, or change tau/gamma/t_start/confidence thresholds.

## Variant Mapping

- `t1` -> `V1_single_t1`
- `t2` -> `V2_single_t2`
- `t3` -> `V3_single_t3`
- `multit` -> `V4_multit_confidence_bin`

## B. 4 x 6 ADE Table

| condition | ADE_t1 | ADE_t2 | ADE_t3 | ADE_multit_bin | best_variant |
| --- | --- | --- | --- | --- | --- |
| gaussian_medium | 0.061417311 | 0.060643539 | 0.059841648 | 0.060201008 | t3 |
| drift_medium | 0.035146751 | 0.034655422 | 0.034238301 | 0.034325946 | t3 |
| burst_medium | 0.071492597 | 0.071739152 | 0.072575316 | 0.071192235 | multit_bin |
| bias_medium | 0.187059984 | 0.187033728 | 0.187039986 | 0.187039763 | t2 |
| jump_medium | 0.257850826 | 0.255911469 | 0.253745109 | 0.254901350 | t3 |
| combined_medium | 0.205996528 | 0.205701306 | 0.205332279 | 0.205512941 | t3 |

## C. Difference Table

| condition | t3_minus_t2 | t3_minus_t1 | multit_minus_t3 | best_variant | t3_improves_over_t1 | t3_improves_over_t2 | multit_improves_over_all_single_t |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | -0.000801891 | -0.001575664 | 0.000359360 | t3 | True | True | False |
| drift_medium | -0.000417121 | -0.000908449 | 0.000087645 | t3 | True | True | False |
| burst_medium | 0.000836164 | 0.001082718 | -0.001383081 | multit_bin | False | False | True |
| bias_medium | 0.000006258 | -0.000019997 | -0.000000224 | t2 | True | False | False |
| jump_medium | -0.002166361 | -0.004105717 | 0.001156241 | t3 | True | True | False |
| combined_medium | -0.000369027 | -0.000664249 | 0.000180662 | t3 | True | True | False |

## D. High-Confidence No-Harm Table

| condition | multit | t1 | t2 | t3 |
| --- | --- | --- | --- | --- |
| gaussian_medium | PASS | PASS | PASS | PASS |
| drift_medium | PASS | PASS | PASS | PASS |
| burst_medium | PASS | PASS | PASS | PASS |
| bias_medium | PASS | PASS | PASS | PASS |
| jump_medium | UNINTERPRETABLE | UNINTERPRETABLE | UNINTERPRETABLE | UNINTERPRETABLE |
| combined_medium | PASS | PASS | PASS | PASS |

## E. Low-Confidence Improvement Table

| condition | multit | t1 | t2 | t3 |
| --- | --- | --- | --- | --- |
| gaussian_medium | PASS | PASS | PASS | PASS |
| drift_medium | PASS | PASS | PASS | PASS |
| burst_medium | PASS | PASS | PASS | PASS |
| bias_medium | PASS | PASS | PASS | PASS |
| jump_medium | PASS | PASS | PASS | PASS |
| combined_medium | PASS | PASS | PASS | PASS |

## F. Motion Usage Sanity Table

| condition | multit | t1 | t2 | t3 |
| --- | --- | --- | --- | --- |
| gaussian_medium | 0.498083704 | 0.408194921 | 0.409048703 | 0.410133683 |
| drift_medium | 0.567404642 | 0.476310473 | 0.478261796 | 0.478776526 |
| burst_medium | 0.617212419 | 0.481558740 | 0.482946716 | 0.484280135 |
| bias_medium | 0.485090446 | 0.403487361 | 0.403988951 | 0.402688271 |
| jump_medium | 0.500326372 | 0.436639631 | 0.439062565 | 0.440602239 |
| combined_medium | 0.488318429 | 0.399820987 | 0.399618176 | 0.398342340 |

All motion_usage_ratio entries are also written with `[0.2, 0.7]` sanity flags in the CSV output.

## G. Pattern Classification

- Classification: `Pattern 2`
- Evidence: gaussian_medium has t3 clearly better than t2 (t3_minus_t2=-0.000801891); t3 is best in 4/6 conditions.

## H. Final Caution

Protocol reconciliation is needed before final lock because this conflicts with the earlier Stage 3 vanilla gaussian t2 sweet spot.

This extraction does not make the final scientific decision alone; it supplies the per-condition evidence needed before locking the final t_start/source rule.

## Output Files

- `outputs/stage4/e3_multit_candidate_source_ablation/per_condition_consistency/per_condition_ade_table.csv`
- `outputs/stage4/e3_multit_candidate_source_ablation/per_condition_consistency/per_condition_high_conf_table.csv`
- `outputs/stage4/e3_multit_candidate_source_ablation/per_condition_consistency/per_condition_low_conf_table.csv`
- `outputs/stage4/e3_multit_candidate_source_ablation/per_condition_consistency/per_condition_motion_usage_table.csv`
- `outputs/stage4/e3_multit_candidate_source_ablation/per_condition_consistency/per_condition_consistency_summary.md`
