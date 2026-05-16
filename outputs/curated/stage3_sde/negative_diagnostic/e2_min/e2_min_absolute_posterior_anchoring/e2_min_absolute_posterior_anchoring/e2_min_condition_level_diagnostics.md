# E2-Min Condition-Level Diagnostics

## Protocol
All Stage 4 formal E1/E2 evaluation is interpreted under the unified six-condition protocol: gaussian_medium, drift_medium, burst_medium, bias_medium, jump_medium, and combined_medium. Inputs are Stage 3 protocol-validated per-frame conditional outputs; provenance labels are retained, but all ADE means and decisions use all six conditions.

## Six-Condition Mean ADE
| method | six_condition_mean_ADE |
| --- | --- |
| Formal E1 | 0.128928 |
| E2-Min V1 | 0.126577 |
| E2-Min V2 selected | 0.124478 |
| E2-Min V3 | 0.125509 |

## Layered PASS / NO-PASS
| layer | criterion | description | pass | value |
| --- | --- | --- | --- | --- |
| Layer 1 | G1 | ADE improves over Formal E1 in at least 4/6 degradation conditions | True | 4/6 |
| Layer 1 | G2 | No severe regression: ADE_E2 / ADE_E1 < 1.10 for all six | True | max ratio 1.025085 |
| Layer 1 | G3 | Six-condition mean ADE improves over Formal E1 | True | 0.124478 vs 0.128928 |
| Layer 2 | H1 | drift_medium high-confidence no-harm passes | False | V2 high 0.006796; noisy high 0.002408 |
| Layer 2 | H2 | burst_medium high-confidence no-harm passes | False | V2 high 0.010821; noisy high 0.003384 |
| Layer 2 | H3 | drift_medium overall ADE improves over Formal E1 | True | 0.021083 vs 0.023272 |
| Layer 2 | H4 | burst_medium overall ADE improves over Formal E1 | True | 0.054175 vs 0.073922 |
| Layer 2 | H5 | bias_medium ADE improves by at least 5%, or bias offset error decreases clearly | False | ADE 0.189430 vs 0.187644; offset 0.188996 vs 0.187441 |
| Layer 2 | H6 | Low-confidence correction preservation passes | True | all six |
| Overall | Decision | PASS if Layer 1 and Layer 2 all pass; Partial-PASS if Layer 1 all pass and at least 4/6 Layer 2 pass | False | NO-PASS; Layer1 all=True; Layer2 3/6 |

Final decision: **NO-PASS**. E2-Min is partial-positive under six-condition global ADE, but its hypothesis-level success is limited because it does not fully repair high-confidence no-harm and bias anchoring.

## Why V2 Was Selected
V2_uniform_cond_motion was selected by the existing E2-Min six-ADE rule: maximize the number of conditions where ADE improves over Formal E1, then break ties by mean ADE gain and mean ADE. The best setting was alpha0=0.5, beta=0.05, sigma0=1.0.

| variant | alpha0 | beta | sigma0 | six_ADE_improved_count | mean_ADE_gain_vs_E1 | mean_ADE_E2 |
| --- | --- | --- | --- | --- | --- | --- |
| V2_uniform_cond_motion | 0.500000 | 0.050000 | 1.000000 | 4.000000 | 0.004450 | 0.124478 |
| V2_uniform_cond_motion | 0.500000 | 0.010000 | 1.000000 | 4.000000 | 0.003902 | 0.125025 |
| V2_uniform_cond_motion | 0.500000 | 0.100000 | 0.500000 | 4.000000 | 0.003535 | 0.125393 |
| V3_conf_mod_cond_motion | 0.500000 | 0.010000 | 1.000000 | 4.000000 | 0.003500 | 0.125428 |
| V3_conf_mod_cond_motion | 0.500000 | 0.050000 | 1.000000 | 4.000000 | 0.003419 | 0.125509 |
| V2_uniform_cond_motion | 0.200000 | 0.010000 | 1.000000 | 4.000000 | 0.003318 | 0.125610 |
| V2_uniform_cond_motion | 0.500000 | 0.050000 | 0.500000 | 4.000000 | 0.003144 | 0.125784 |
| V2_uniform_cond_motion | 0.100000 | 0.010000 | 1.000000 | 4.000000 | 0.002836 | 0.126092 |
| V3_conf_mod_cond_motion | 0.500000 | 0.100000 | 0.500000 | 4.000000 | 0.002676 | 0.126252 |
| V3_conf_mod_cond_motion | 0.200000 | 0.010000 | 1.000000 | 4.000000 | 0.002644 | 0.126284 |

## ADE by Condition
| degradation | Formal_E1_ADE | V1_ADE | V1_rel_change_vs_E1 | V2_selected_ADE | V2_rel_change_vs_E1 | V3_ADE | V3_rel_change_vs_E1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 0.048260 | 0.049931 | 0.034606 | 0.039722 | -0.176919 | 0.041733 | -0.135248 |
| drift_medium | 0.023272 | 0.031611 | 0.358305 | 0.021083 | -0.094083 | 0.022299 | -0.041817 |
| burst_medium | 0.073922 | 0.039268 | -0.468796 | 0.054175 | -0.267130 | 0.055555 | -0.248461 |
| bias_medium | 0.187644 | 0.191176 | 0.018823 | 0.189430 | 0.009519 | 0.189756 | 0.011251 |
| jump_medium | 0.246264 | 0.254094 | 0.031797 | 0.252441 | 0.025085 | 0.252810 | 0.026582 |
| combined_medium | 0.194204 | 0.193383 | -0.004228 | 0.190018 | -0.021555 | 0.190899 | -0.017016 |

## C2 High-Confidence No-Harm
| degradation | noisy_high_ADE | Formal_E1_high_ADE | V1_high_ADE | V1_C2_pass_1p05 | V2_high_ADE | V2_C2_pass_1p05 | V3_high_ADE | V3_C2_pass_1p05 | C2_threshold_1p05_noisy_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | 0.014083 | 0.015238 | 0.019758 | False | 0.015872 | False | 0.017743 | False | 0.014787 |
| drift_medium | 0.002408 | 0.003635 | 0.007599 | False | 0.006796 | False | 0.007092 | False | 0.002528 |
| burst_medium | 0.003384 | 0.011275 | 0.014307 | False | 0.010821 | False | 0.013207 | False | 0.003554 |
| bias_medium | 0.042336 | 0.041958 | 0.049337 | False | 0.044288 | True | 0.047246 | False | 0.044453 |
| jump_medium | 0.000000 | 0.000000 | 0.009281 | False | 0.010637 | False | 0.009908 | False | 0.000000 |
| combined_medium | 0.044971 | 0.046620 | 0.049710 | False | 0.049146 | False | 0.049470 | False | 0.047220 |

C2 failures for selected V2: gaussian_medium, drift_medium, burst_medium, jump_medium, combined_medium. The substantive failures are drift and burst; gaussian and combined are mild bin-level mismatches, while jump is affected by the sparse-outlier/zero-noisy-high-confidence edge case.

## Drift Diagnostic
V2 improves drift overall ADE versus Formal E1, but still fails high-confidence no-harm. That means the absolute-space objective helps the average trajectory yet still introduces too much error in frames that the oracle confidence marks as reliable.

## Burst Diagnostic
| degradation | V1_ADE | V1_high_ADE | V1_low_ADE | V1_smooth_acc_rms | V1_motion_usage_ratio | V1_noisy_reversion_gap | V2_selected_ADE | V2_high_ADE | V2_low_ADE | V2_smooth_acc_rms | V2_motion_usage_ratio | V2_noisy_reversion_gap | V3_ADE | V3_high_ADE | V3_low_ADE | V3_smooth_acc_rms | V3_motion_usage_ratio | V3_noisy_reversion_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burst_medium | 0.039268 | 0.014307 | 0.065639 | 0.088699 | 2.584914 | -0.035488 | 0.054175 | 0.010821 | 0.115016 | 0.265571 | 1.285020 | -0.020580 | 0.055555 | 0.013207 | 0.115386 | 0.252633 | 1.373067 | -0.019200 |

V1's burst improvement is real at the condition ADE level: it is far below noisy and Formal E1, and its noisy_reversion_gap is strongly negative. However, the mechanism is mostly absolute anchoring plus aggressive smoothness over localized burst corruption, not a complete high-confidence repair: V1 still fails C2 and has very low acceleration RMS.

## Bias and Jump Regression Diagnostic
| degradation | Formal_E1_ADE | V1_ADE | V2_selected_ADE | V3_ADE | V1_high_ADE | V2_high_ADE | V3_high_ADE | V1_low_ADE | V2_low_ADE | V3_low_ADE | V1_motion_usage_ratio | V2_motion_usage_ratio | V3_motion_usage_ratio | V1_noisy_reversion_gap | V2_noisy_reversion_gap | V3_noisy_reversion_gap | V1_smooth_acc_rms | V2_smooth_acc_rms | V3_smooth_acc_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bias_medium | 0.187644 | 0.191176 | 0.189430 | 0.189756 | 0.049337 | 0.044288 | 0.047246 | 0.289761 | 0.288265 | 0.288320 | 3.050695 | 1.480129 | 1.720026 | 0.002181 | 0.000435 | 0.000760 | 0.087255 | 0.140433 | 0.131027 |
| jump_medium | 0.246264 | 0.254094 | 0.252441 | 0.252810 | 0.009281 | 0.010637 | 0.009908 | 0.405701 | 0.403742 | 0.404390 | 1.777168 | 1.142333 | 1.223394 | -0.009274 | -0.010927 | -0.010558 | 0.154520 | 0.218085 | 0.207704 |

Bias regression is mainly caused by L2 anchoring to a biased observation: the selected V2 keeps the biased absolute offset instead of reducing it, while conditional motion cannot correct a global absolute shift. Jump regression is a known sparse-outlier limitation: the L2/smoothness objective mildly worsens Formal E1 despite remaining better than noisy, and jump is not one of the H1-H6 hypothesis targets.

## Bias Offset
| method | ADE_mean | bias_offset_error | smooth_acc_rms_mean |
| --- | --- | --- | --- |
| noisy_input | 0.188996 | 0.188996 | 0.176980 |
| stage3_cond_residual_t20 | 0.188868 | 0.188100 | 0.172079 |
| formal_e1_linear_gate | 0.187644 | 0.187441 | 0.173481 |
| V1_abs_anchor_smooth | 0.191176 | 0.188996 | 0.087255 |
| V2_uniform_cond_motion | 0.189430 | 0.188996 | 0.140433 |
| V3_conf_mod_cond_motion | 0.189756 | 0.188996 | 0.131027 |

## Layer 2 Criteria
- H1 drift high-confidence no-harm: False
- H2 burst high-confidence no-harm: False
- H3 drift overall ADE improves: True
- H4 burst overall ADE improves: True
- H5 bias ADE or offset improves: False
- H6 low-confidence preservation: True

## E2-DPS Decision
E2-DPS should be pursued. It must specifically fix high-confidence anchoring without damaging reliable frames, handle global bias as an absolute-position posterior problem rather than motion-only correction, and avoid L2 smoothing failure around sparse jump outliers.
