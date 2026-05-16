# E3 Multi-t Candidate-Source Ablation Decision Summary

This evaluation reused existing per-frame SDEdit t1/t2/t3 candidates only. It did not train, generate candidates, use DPS, or modify existing E3 hold-out outputs.

## Baseline Reproduction

- Known six-condition E3 fused t1 ADE anchor: `0.136567`
- Recomputed V1_single_t1 six-condition mean ADE: `0.136494000`
- Absolute difference: `0.000073000`
- Result: `BASELINE_REPRODUCTION_PASS`

## Six-Condition Mean ADE Table

| variant | six_condition_mean_ADE | mean_motion_usage_ratio | high_conf_pass_count | low_conf_pass_count |
| --- | --- | --- | --- | --- |
| V1_single_t1 | 0.136494000 | 0.434335352 | 5 | 6 |
| V2_single_t2 | 0.135947436 | 0.435487818 | 5 | 6 |
| V3_single_t3 | 0.135462107 | 0.435803866 | 5 | 6 |
| V4_multit_confidence_bin | 0.135528874 | 0.526072669 | 5 | 6 |

## Per-Condition ADE Table

| condition | ADE_t1 | ADE_t2 | ADE_t3 | ADE_multit_bin | best_variant |
| --- | --- | --- | --- | --- | --- |
| gaussian_medium | 0.061417311 | 0.060643539 | 0.059841648 | 0.060201008 | V3_single_t3 |
| drift_medium | 0.035146751 | 0.034655422 | 0.034238301 | 0.034325946 | V3_single_t3 |
| burst_medium | 0.071492597 | 0.071739152 | 0.072575316 | 0.071192235 | V4_multit_confidence_bin |
| bias_medium | 0.187059984 | 0.187033728 | 0.187039986 | 0.187039763 | V2_single_t2 |
| jump_medium | 0.257850826 | 0.255911469 | 0.253745109 | 0.254901350 | V3_single_t3 |
| combined_medium | 0.205996528 | 0.205701306 | 0.205332279 | 0.205512941 | V3_single_t3 |

## High-Confidence No-Harm Table

| condition | high_conf_pass_t1 | high_conf_pass_t2 | high_conf_pass_t3 | high_conf_pass_multit_bin |
| --- | --- | --- | --- | --- |
| gaussian_medium | PASS | PASS | PASS | PASS |
| drift_medium | PASS | PASS | PASS | PASS |
| burst_medium | PASS | PASS | PASS | PASS |
| bias_medium | PASS | PASS | PASS | PASS |
| jump_medium | UNINTERPRETABLE | UNINTERPRETABLE | UNINTERPRETABLE | UNINTERPRETABLE |
| combined_medium | PASS | PASS | PASS | PASS |

## Low-Confidence Improvement Table

| condition | low_conf_pass_t1 | low_conf_pass_t2 | low_conf_pass_t3 | low_conf_pass_multit_bin |
| --- | --- | --- | --- | --- |
| gaussian_medium | PASS | PASS | PASS | PASS |
| drift_medium | PASS | PASS | PASS | PASS |
| burst_medium | PASS | PASS | PASS | PASS |
| bias_medium | PASS | PASS | PASS | PASS |
| jump_medium | PASS | PASS | PASS | PASS |
| combined_medium | PASS | PASS | PASS | PASS |

## Motion Usage Ratio Table

| condition | motion_usage_ratio_t1 | motion_usage_ratio_t2 | motion_usage_ratio_t3 | motion_usage_ratio_multit_bin |
| --- | --- | --- | --- | --- |
| gaussian_medium | 0.408195 | 0.409049 | 0.410134 | 0.498084 |
| drift_medium | 0.476310 | 0.478262 | 0.478777 | 0.567405 |
| burst_medium | 0.481559 | 0.482947 | 0.484280 | 0.617212 |
| bias_medium | 0.403487 | 0.403989 | 0.402688 | 0.485090 |
| jump_medium | 0.436640 | 0.439063 | 0.440602 | 0.500326 |
| combined_medium | 0.399821 | 0.399618 | 0.398342 | 0.488318 |

## Case A/B/C Decision

| case | variant | six_condition_mean_ADE | six_condition_mean_ADE_t1 | mean_ADE_lower_than_t1 | conditions_lower_than_t1 | high_conf_no_harm_pass_count | high_conf_interpretable_count | low_conf_pass_count | mean_motion_usage_ratio | motion_usage_in_range_0p2_0p7 | passed | mean_ADE_lower_than_all_single_t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Case A | V2_single_t2 | 0.135947436 | 0.136494000 | True | 5 | 5 | 5 | 6 | 0.435487818 | True | True |  |
| Case A | V3_single_t3 | 0.135462107 | 0.136494000 | True | 5 | 5 | 5 | 6 | 0.435803866 | True | True |  |
| Case B | V4_multit_confidence_bin | 0.135528874 | 0.136494000 |  | 6 | 5 | 5 | 6 | 0.526072669 | True | False | False |

- Final decision: `Case A`
- Final recommended SDE candidate source: `V3_single_t3`

## Required Answers

1. Are t2 or t3 better than t1 under the same fusion rule? `t2=True`, `t3=True`.
2. Is confidence-binned multi-t source better than all single-t variants? `False`.
3. Candidate-source rule to use as final SDE component: `V3_single_t3`.
4. E3-t1 is replaced by `V3_single_t3` under the pre-specified decision rules.

## Representative Figures

- `outputs/stage4/e3_multit_candidate_source_ablation/figures/gaussian_medium_median_t1_candidate_source_comparison.png`
- `outputs/stage4/e3_multit_candidate_source_ablation/figures/drift_medium_median_t1_candidate_source_comparison.png`
- `outputs/stage4/e3_multit_candidate_source_ablation/figures/burst_medium_median_t1_candidate_source_comparison.png`
- `outputs/stage4/e3_multit_candidate_source_ablation/figures/bias_medium_median_t1_candidate_source_comparison.png`
- `outputs/stage4/e3_multit_candidate_source_ablation/figures/jump_medium_median_t1_candidate_source_comparison.png`
- `outputs/stage4/e3_multit_candidate_source_ablation/figures/combined_medium_median_t1_candidate_source_comparison.png`

## Output Files

- `outputs/stage4/e3_multit_candidate_source_ablation/e3_multit_candidate_source_full_metrics.csv`
- `outputs/stage4/e3_multit_candidate_source_ablation/e3_multit_candidate_source_condition_summary.csv`
- `outputs/stage4/e3_multit_candidate_source_ablation/e3_multit_candidate_source_bin_metrics.csv`
- `outputs/stage4/e3_multit_candidate_source_ablation/e3_multit_candidate_source_pass_fail_summary.csv`
- `outputs/stage4/e3_multit_candidate_source_ablation/e3_multit_candidate_source_decision_summary.md`
