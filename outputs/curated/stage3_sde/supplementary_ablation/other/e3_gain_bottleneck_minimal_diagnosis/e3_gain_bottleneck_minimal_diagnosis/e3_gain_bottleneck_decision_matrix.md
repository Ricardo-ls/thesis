# E3 Gain Bottleneck Decision Matrix

Oracle diagnostics are analysis-only and are not method performance.

Decision uses the confidence-constrained frame-wise oracle gap vs `fused_t1_tau07_gamma2`, combined with the best low-confidence direction alignment across t_start 1/2/3.

| condition | confidence_oracle_gap_ratio_t1 | best_low_t_start_by_cosine | low_cosine_mean | low_magnitude_ratio_median | low_fraction_cosine_lt_0 | decision_case | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 0.088955 | 3 | 0.337390 | 0.208938 | 0.271841 | Case E | boundary case; only a small adaptive test is justified if direction diagnostics are strong, and it must use a new independent hold-out |
| drift_medium | 0.090760 | 3 | 0.214109 | 0.288352 | 0.370309 | Case E | boundary case; only a small adaptive test is justified if direction diagnostics are strong, and it must use a new independent hold-out |
| burst_medium | -0.008794 | 3 | 0.258511 | 0.269797 | 0.325540 | Case D | current fusion is near the SDEdit-candidate ceiling; stop tuning fusion and consider reporting stable limited improvement or a separate SDEdit-initialized DPS audit |
| bias_medium | 0.014422 | 2 | 0.038131 | 0.030033 | 0.450531 | Case D | current fusion is near the SDEdit-candidate ceiling; stop tuning fusion and consider reporting stable limited improvement or a separate SDEdit-initialized DPS audit |
| jump_medium | 0.047414 | 3 | 0.529522 | 0.077440 | 0.180224 | Case D | current fusion is near the SDEdit-candidate ceiling; stop tuning fusion and consider reporting stable limited improvement or a separate SDEdit-initialized DPS audit |
| combined_medium | 0.024006 | 3 | 0.125034 | 0.059837 | 0.401719 | Case D | current fusion is near the SDEdit-candidate ceiling; stop tuning fusion and consider reporting stable limited improvement or a separate SDEdit-initialized DPS audit |

## Overall

- modal case: `Case D`
- recommendation: current fusion is near the SDEdit-candidate ceiling; stop tuning fusion and consider reporting stable limited improvement or a separate SDEdit-initialized DPS audit

Case thresholds:

- Case A: oracle gap > 10%, low-conf cosine > 0.5, median magnitude ratio < 0.75
- Case B: oracle gap > 10%, low-conf cosine > 0.5, median magnitude ratio in [0.75, 1.25]
- Case C: oracle gap > 10%, low-conf cosine < 0.3 or negative-direction fraction > 0.35
- Case D: oracle gap < 5%
- Case E: oracle gap 5-10% or mixed direction/magnitude evidence
