# E3 Gain Bottleneck Minimal Diagnosis

Current 1000 hold-out results were frozen: `yes`.

This diagnosis reads only existing arrays. It does not run new SDEdit sampling, train models, modify checkpoints, tune tau/gamma/t_start, or create a new formal method.

## Oracle Upper Bound

- Frame-wise best oracle mean gap ratio vs fused_t1: `0.072665`
- Confidence-constrained oracle mean gap ratio vs fused_t1: `0.042794`
- Confidence-constrained oracle mean gap ratio vs fused_best: `0.038288`
- Trajectory-level best oracle mean gap ratio vs fused_t1: `0.036709`

Interpretation:

- Confidence-constrained oracle is close to current fused_t1; current fusion is near the SDEdit-candidate ceiling.

## Correction Direction And Magnitude

- Best overall low-confidence t_start by cosine: `t3`
- Best low-confidence cosine mean: `0.250379`
- Best low-confidence median magnitude ratio: `0.143189`
- Best overall mid-confidence t_start by cosine: `t3`
- Best mid-confidence cosine mean: `0.148766`

- Low-confidence SDEdit correction direction is weak; amplifying SDEdit would be risky.
- Low-confidence correction magnitude is generally too weak relative to oracle correction.

## Decision Cases

| condition | confidence_oracle_gap_ratio_t1 | best_low_t_start_by_cosine | low_cosine_mean | low_magnitude_ratio_median | decision_case |
| --- | --- | --- | --- | --- | --- |
| gaussian_medium | 0.088955 | 3 | 0.337390 | 0.208938 | Case E |
| drift_medium | 0.090760 | 3 | 0.214109 | 0.288352 | Case E |
| burst_medium | -0.008794 | 3 | 0.258511 | 0.269797 | Case D |
| bias_medium | 0.014422 | 2 | 0.038131 | 0.030033 | Case D |
| jump_medium | 0.047414 | 3 | 0.529522 | 0.077440 | Case D |
| combined_medium | 0.024006 | 3 | 0.125034 | 0.059837 | Case D |

Overall modal case: `Case D`.
Recommended next step: current fusion is near the SDEdit-candidate ceiling; stop tuning fusion and consider reporting stable limited improvement or a separate SDEdit-initialized DPS audit.

If any adaptive gamma or adaptive t_start rule is tested next, it must be validated on a new independent hold-out set, not seed 13000-13999.

## Output Files

- `outputs/stage4/e3_gain_bottleneck_minimal_diagnosis/e3_oracle_upper_bound_minimal.csv`
- `outputs/stage4/e3_gain_bottleneck_minimal_diagnosis/e3_correction_direction_minimal.csv`
- `outputs/stage4/e3_gain_bottleneck_minimal_diagnosis/e3_gain_bottleneck_decision_matrix.md`
