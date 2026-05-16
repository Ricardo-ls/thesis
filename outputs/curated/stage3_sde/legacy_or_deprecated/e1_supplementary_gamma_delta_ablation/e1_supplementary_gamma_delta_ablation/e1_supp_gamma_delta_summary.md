# E1 Supplementary Gamma/Delta Ablation

## Objective
This is a supplementary diagnostic ablation, not a new main method.

## Motivation
Formal E1 passed C1 and C3 but failed C2 under drift/burst. This ablation checks whether scalar gates can solve C2 or merely suppress residual usage.

## Data source
Uses the same six Stage 3 protocol-validated per-frame conditional outputs as the 6-condition Formal E1.
| degradation | conditional_source | protocol_validation_pass |
| --- | --- | --- |
| gaussian_medium | archived_original | True |
| drift_medium | archived_original | True |
| jump_medium | matched_protocol_recomputed | True |
| burst_medium | archived_original | True |
| bias_medium | archived_original | True |
| combined_medium | matched_protocol_recomputed | True |

## Formula
- lambda_t = (1 - c_t)^gamma
- x_gamma,t = y_t + lambda_t * (x_cond,t - y_t)

## Full gamma/delta table
| degradation | conditional_source | delta0_label | delta0 | gamma | ADE_noisy | ADE_cond | ADE_gamma | ADE_gamma_high | ADE_gamma_low | C2_high_no_harm_pass | C3_low_preservation_pass | residual_usage_ratio | noisy_reversion_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | archived_original | default | 0.058828 | 1 | 0.062567 | 0.055695 | 0.048260 | 0.015238 | 0.069237 | True | True | 0.629625 | -0.014306 |
| gaussian_medium | archived_original | default | 0.058828 | 2 | 0.062567 | 0.055695 | 0.049949 | 0.013816 | 0.072336 | True | True | 0.432365 | -0.012617 |
| gaussian_medium | archived_original | default | 0.058828 | 3 | 0.062567 | 0.055695 | 0.052502 | 0.013986 | 0.075874 | True | False | 0.312922 | -0.010064 |
| gaussian_medium | archived_original | default | 0.058828 | 4 | 0.062567 | 0.055695 | 0.054584 | 0.014055 | 0.079123 | True | False | 0.234772 | -0.007983 |
| gaussian_medium | archived_original | fixed_0.02 | 0.020000 | 1 | 0.062567 | 0.055695 | 0.051854 | 0.009418 | 0.055064 | False | True | 0.901658 | -0.010713 |
| gaussian_medium | archived_original | fixed_0.02 | 0.020000 | 2 | 0.062567 | 0.055695 | 0.049737 | 0.005270 | 0.053598 | True | True | 0.832208 | -0.012829 |
| gaussian_medium | archived_original | fixed_0.02 | 0.020000 | 3 | 0.062567 | 0.055695 | 0.048584 | 0.005085 | 0.052603 | True | True | 0.778782 | -0.013983 |
| gaussian_medium | archived_original | fixed_0.02 | 0.020000 | 4 | 0.062567 | 0.055695 | 0.047989 | 0.005122 | 0.051988 | True | True | 0.735500 | -0.014577 |
| gaussian_medium | archived_original | fixed_0.05 | 0.050000 | 1 | 0.062567 | 0.055695 | 0.048304 | 0.014069 | 0.064037 | False | True | 0.680897 | -0.014262 |
| gaussian_medium | archived_original | fixed_0.05 | 0.050000 | 2 | 0.062567 | 0.055695 | 0.048815 | 0.011972 | 0.065905 | True | True | 0.499822 | -0.013751 |
| gaussian_medium | archived_original | fixed_0.05 | 0.050000 | 3 | 0.062567 | 0.055695 | 0.050793 | 0.011998 | 0.068524 | True | True | 0.383859 | -0.011774 |
| gaussian_medium | archived_original | fixed_0.05 | 0.050000 | 4 | 0.062567 | 0.055695 | 0.052652 | 0.012029 | 0.071143 | True | False | 0.304004 | -0.009914 |
| gaussian_medium | archived_original | fixed_0.10 | 0.100000 | 1 | 0.062567 | 0.055695 | 0.049931 | 0.021634 | 0.101530 | True | True | 0.460430 | -0.012636 |
| gaussian_medium | archived_original | fixed_0.10 | 0.100000 | 2 | 0.062567 | 0.055695 | 0.054670 | 0.022499 | 0.109549 | True | False | 0.239963 | -0.007897 |
| gaussian_medium | archived_original | fixed_0.10 | 0.100000 | 3 | 0.062567 | 0.055695 | 0.057783 | 0.022963 | 0.116028 | True | False | 0.135064 | -0.004784 |
| gaussian_medium | archived_original | fixed_0.10 | 0.100000 | 4 | 0.062567 | 0.055695 | 0.059600 | 0.023090 | 0.121069 | True | False | 0.080225 | -0.002966 |
| gaussian_medium | archived_original | fixed_0.20 | 0.200000 | 1 | 0.062567 | 0.055695 | 0.054086 | 0.038435 | nan | True | True | 0.274583 | -0.008480 |
| gaussian_medium | archived_original | fixed_0.20 | 0.200000 | 2 | 0.062567 | 0.055695 | 0.059436 | 0.041602 | nan | True | True | 0.088735 | -0.003131 |
| gaussian_medium | archived_original | fixed_0.20 | 0.200000 | 3 | 0.062567 | 0.055695 | 0.061379 | 0.042436 | nan | True | True | 0.031817 | -0.001188 |
| gaussian_medium | archived_original | fixed_0.20 | 0.200000 | 4 | 0.062567 | 0.055695 | 0.062092 | 0.042641 | nan | True | True | 0.012291 | -0.000475 |
| drift_medium | archived_original | default | 0.015678 | 1 | 0.017648 | 0.031591 | 0.023272 | 0.003635 | 0.036123 | False | True | 0.650923 | 0.005624 |
| drift_medium | archived_original | default | 0.015678 | 2 | 0.017648 | 0.031591 | 0.020115 | 0.002378 | 0.033505 | True | True | 0.469359 | 0.002467 |
| drift_medium | archived_original | default | 0.015678 | 3 | 0.017648 | 0.031591 | 0.018865 | 0.002384 | 0.031833 | True | True | 0.359529 | 0.001217 |
| drift_medium | archived_original | default | 0.015678 | 4 | 0.017648 | 0.031591 | 0.018284 | 0.002401 | 0.030800 | True | True | 0.286749 | 0.000636 |
| drift_medium | archived_original | fixed_0.02 | 0.020000 | 1 | 0.017648 | 0.031591 | 0.021851 | 0.004500 | 0.038323 | False | True | 0.576620 | 0.004203 |
| drift_medium | archived_original | fixed_0.02 | 0.020000 | 2 | 0.017648 | 0.031591 | 0.018956 | 0.003356 | 0.035821 | True | True | 0.377127 | 0.001308 |
| drift_medium | archived_original | fixed_0.02 | 0.020000 | 3 | 0.017648 | 0.031591 | 0.018071 | 0.003348 | 0.034389 | True | True | 0.266048 | 0.000423 |
| drift_medium | archived_original | fixed_0.02 | 0.020000 | 4 | 0.017648 | 0.031591 | 0.017724 | 0.003355 | 0.033574 | True | True | 0.197528 | 0.000076 |
| drift_medium | archived_original | fixed_0.05 | 0.050000 | 1 | 0.017648 | 0.031591 | 0.018329 | 0.009868 | 0.028606 | True | True | 0.315470 | 0.000681 |
| drift_medium | archived_original | fixed_0.05 | 0.050000 | 2 | 0.017648 | 0.031591 | 0.017468 | 0.009283 | 0.033302 | True | True | 0.122906 | -0.000180 |
| drift_medium | archived_original | fixed_0.05 | 0.050000 | 3 | 0.017648 | 0.031591 | 0.017464 | 0.009275 | 0.039952 | True | False | 0.054794 | -0.000184 |
| drift_medium | archived_original | fixed_0.05 | 0.050000 | 4 | 0.017648 | 0.031591 | 0.017517 | 0.009277 | 0.045954 | True | False | 0.026844 | -0.000131 |
| drift_medium | archived_original | fixed_0.10 | 0.100000 | 1 | 0.017648 | 0.031591 | 0.017600 | 0.015493 | nan | True | True | 0.178265 | -0.000048 |
| drift_medium | archived_original | fixed_0.10 | 0.100000 | 2 | 0.017648 | 0.031591 | 0.017515 | 0.015323 | nan | True | True | 0.041060 | -0.000133 |
| drift_medium | archived_original | fixed_0.10 | 0.100000 | 3 | 0.017648 | 0.031591 | 0.017592 | 0.015343 | nan | True | True | 0.011193 | -0.000056 |
| drift_medium | archived_original | fixed_0.10 | 0.100000 | 4 | 0.017648 | 0.031591 | 0.017625 | 0.015349 | nan | True | True | 0.003438 | -0.000023 |
| drift_medium | archived_original | fixed_0.20 | 0.200000 | 1 | 0.017648 | 0.031591 | 0.017495 | 0.017495 | nan | True | True | 0.095143 | -0.000153 |
| drift_medium | archived_original | fixed_0.20 | 0.200000 | 2 | 0.017648 | 0.031591 | 0.017600 | 0.017600 | nan | True | True | 0.012020 | -0.000048 |
| drift_medium | archived_original | fixed_0.20 | 0.200000 | 3 | 0.017648 | 0.031591 | 0.017637 | 0.017637 | nan | True | True | 0.001835 | -0.000011 |
| drift_medium | archived_original | fixed_0.20 | 0.200000 | 4 | 0.017648 | 0.031591 | 0.017646 | 0.017646 | nan | True | True | 0.000321 | -0.000002 |
| burst_medium | archived_original | default | 0.014054 | 1 | 0.074755 | 0.091885 | 0.073922 | 0.011275 | 0.148815 | False | True | 0.703691 | -0.000833 |
| burst_medium | archived_original | default | 0.014054 | 2 | 0.074755 | 0.091885 | 0.066331 | 0.004138 | 0.145235 | False | True | 0.564005 | -0.008424 |
| burst_medium | archived_original | default | 0.014054 | 3 | 0.074755 | 0.091885 | 0.063066 | 0.003409 | 0.142784 | True | True | 0.488090 | -0.011689 |
| burst_medium | archived_original | default | 0.014054 | 4 | 0.074755 | 0.091885 | 0.061648 | 0.003381 | 0.141201 | True | True | 0.442999 | -0.013107 |
| burst_medium | archived_original | fixed_0.02 | 0.020000 | 1 | 0.074755 | 0.091885 | 0.069872 | 0.011309 | 0.215154 | False | True | 0.631371 | -0.004883 |
| burst_medium | archived_original | fixed_0.02 | 0.020000 | 2 | 0.074755 | 0.091885 | 0.062968 | 0.005158 | 0.213822 | True | True | 0.487842 | -0.011788 |
| burst_medium | archived_original | fixed_0.02 | 0.020000 | 3 | 0.074755 | 0.091885 | 0.061035 | 0.004734 | 0.213021 | True | True | 0.422125 | -0.013721 |
| burst_medium | archived_original | fixed_0.02 | 0.020000 | 4 | 0.074755 | 0.091885 | 0.060591 | 0.004738 | 0.212552 | True | True | 0.388693 | -0.014164 |
| burst_medium | archived_original | fixed_0.05 | 0.050000 | 1 | 0.074755 | 0.091885 | 0.062785 | 0.012556 | 0.254290 | False | True | 0.478617 | -0.011971 |
| burst_medium | archived_original | fixed_0.05 | 0.050000 | 2 | 0.074755 | 0.091885 | 0.060393 | 0.009842 | 0.254176 | True | True | 0.368410 | -0.014363 |
| burst_medium | archived_original | fixed_0.05 | 0.050000 | 3 | 0.074755 | 0.091885 | 0.060535 | 0.009965 | 0.254216 | True | True | 0.337561 | -0.014221 |
| burst_medium | archived_original | fixed_0.05 | 0.050000 | 4 | 0.074755 | 0.091885 | 0.060657 | 0.010021 | 0.254350 | True | True | 0.325891 | -0.014099 |
| burst_medium | archived_original | fixed_0.10 | 0.100000 | 1 | 0.074755 | 0.091885 | 0.061098 | 0.012693 | 0.269521 | True | True | 0.392272 | -0.013658 |
| burst_medium | archived_original | fixed_0.10 | 0.100000 | 2 | 0.074755 | 0.091885 | 0.061257 | 0.012366 | 0.271819 | True | True | 0.305927 | -0.013499 |
| burst_medium | archived_original | fixed_0.10 | 0.100000 | 3 | 0.074755 | 0.091885 | 0.061832 | 0.012508 | 0.274098 | True | True | 0.279982 | -0.012924 |
| burst_medium | archived_original | fixed_0.10 | 0.100000 | 4 | 0.074755 | 0.091885 | 0.062286 | 0.012537 | 0.276260 | True | True | 0.264629 | -0.012469 |
| burst_medium | archived_original | fixed_0.20 | 0.200000 | 1 | 0.074755 | 0.091885 | 0.062245 | 0.012484 | 0.333567 | True | True | 0.305661 | -0.012510 |
| burst_medium | archived_original | fixed_0.20 | 0.200000 | 2 | 0.074755 | 0.091885 | 0.064239 | 0.012770 | 0.342560 | True | True | 0.219051 | -0.010516 |
| burst_medium | archived_original | fixed_0.20 | 0.200000 | 3 | 0.074755 | 0.091885 | 0.065693 | 0.012826 | 0.350358 | True | True | 0.181422 | -0.009062 |
| burst_medium | archived_original | fixed_0.20 | 0.200000 | 4 | 0.074755 | 0.091885 | 0.066814 | 0.012833 | 0.357071 | True | False | 0.155413 | -0.007941 |
| bias_medium | archived_original | default | 0.172925 | 1 | 0.188996 | 0.188868 | 0.187644 | 0.041958 | 0.284625 | True | True | 0.617825 | -0.001351 |
| bias_medium | archived_original | default | 0.172925 | 2 | 0.188996 | 0.188868 | 0.187645 | 0.042210 | 0.285081 | True | True | 0.414391 | -0.001350 |
| bias_medium | archived_original | default | 0.172925 | 3 | 0.188996 | 0.188868 | 0.187827 | 0.042306 | 0.285498 | True | True | 0.294348 | -0.001169 |
| bias_medium | archived_original | default | 0.172925 | 4 | 0.188996 | 0.188868 | 0.188020 | 0.042329 | 0.285865 | True | True | 0.218175 | -0.000976 |
| bias_medium | archived_original | fixed_0.02 | 0.020000 | 1 | 0.188996 | 0.188868 | 0.188739 | nan | 0.190493 | False | True | 0.990230 | -0.000257 |
| bias_medium | archived_original | fixed_0.02 | 0.020000 | 2 | 0.188996 | 0.188868 | 0.188659 | nan | 0.190404 | False | True | 0.982065 | -0.000337 |
| bias_medium | archived_original | fixed_0.02 | 0.020000 | 3 | 0.188996 | 0.188868 | 0.188597 | nan | 0.190330 | False | True | 0.975041 | -0.000398 |
| bias_medium | archived_original | fixed_0.02 | 0.020000 | 4 | 0.188996 | 0.188868 | 0.188545 | nan | 0.190269 | False | True | 0.968853 | -0.000450 |
| bias_medium | archived_original | fixed_0.05 | 0.050000 | 1 | 0.188996 | 0.188868 | 0.188248 | nan | 0.196029 | False | True | 0.926139 | -0.000748 |
| bias_medium | archived_original | fixed_0.05 | 0.050000 | 2 | 0.188996 | 0.188868 | 0.187933 | nan | 0.195728 | False | True | 0.869043 | -0.001062 |
| bias_medium | archived_original | fixed_0.05 | 0.050000 | 3 | 0.188996 | 0.188868 | 0.187718 | nan | 0.195498 | False | True | 0.822511 | -0.001278 |
| bias_medium | archived_original | fixed_0.05 | 0.050000 | 4 | 0.188996 | 0.188868 | 0.187558 | nan | 0.195322 | False | True | 0.783294 | -0.001438 |
| bias_medium | archived_original | fixed_0.10 | 0.100000 | 1 | 0.188996 | 0.188868 | 0.187770 | 0.023088 | 0.224212 | True | True | 0.783734 | -0.001226 |
| bias_medium | archived_original | fixed_0.10 | 0.100000 | 2 | 0.188996 | 0.188868 | 0.187460 | 0.023278 | 0.224298 | True | True | 0.641329 | -0.001536 |
| bias_medium | archived_original | fixed_0.10 | 0.100000 | 3 | 0.188996 | 0.188868 | 0.187374 | 0.023289 | 0.224430 | True | True | 0.540397 | -0.001621 |
| bias_medium | archived_original | fixed_0.10 | 0.100000 | 4 | 0.188996 | 0.188868 | 0.187385 | 0.023278 | 0.224577 | True | True | 0.465318 | -0.001611 |
| bias_medium | archived_original | fixed_0.20 | 0.200000 | 1 | 0.188996 | 0.188868 | 0.187661 | 0.048542 | 0.304162 | True | True | 0.570696 | -0.001334 |
| bias_medium | archived_original | fixed_0.20 | 0.200000 | 2 | 0.188996 | 0.188868 | 0.187761 | 0.048481 | 0.305206 | True | True | 0.357657 | -0.001235 |
| bias_medium | archived_original | fixed_0.20 | 0.200000 | 3 | 0.188996 | 0.188868 | 0.187995 | 0.048470 | 0.306031 | True | True | 0.239393 | -0.001001 |
| bias_medium | archived_original | fixed_0.20 | 0.200000 | 4 | 0.188996 | 0.188868 | 0.188207 | 0.048462 | 0.306678 | True | True | 0.168269 | -0.000789 |
| jump_medium | matched_protocol_recomputed | default | 0.302825 | 1 | 0.263368 | 0.246523 | 0.246264 | 0.000000 | 0.399224 | True | True | 0.601119 | -0.017105 |
| jump_medium | matched_protocol_recomputed | default | 0.302825 | 2 | 0.263368 | 0.246523 | 0.251001 | 0.000000 | 0.405950 | True | True | 0.415141 | -0.012368 |
| jump_medium | matched_protocol_recomputed | default | 0.302825 | 3 | 0.263368 | 0.246523 | 0.254381 | 0.000000 | 0.411255 | True | True | 0.291319 | -0.008988 |
| jump_medium | matched_protocol_recomputed | default | 0.302825 | 4 | 0.263368 | 0.246523 | 0.256783 | 0.000000 | 0.415404 | True | True | 0.207364 | -0.006586 |
| jump_medium | matched_protocol_recomputed | fixed_0.02 | 0.020000 | 1 | 0.263368 | 0.246523 | 0.239945 | 0.000000 | 0.318440 | True | True | 0.885891 | -0.023424 |
| jump_medium | matched_protocol_recomputed | fixed_0.02 | 0.020000 | 2 | 0.263368 | 0.246523 | 0.239945 | 0.000000 | 0.318440 | True | True | 0.885889 | -0.023424 |
| jump_medium | matched_protocol_recomputed | fixed_0.02 | 0.020000 | 3 | 0.263368 | 0.246523 | 0.239945 | 0.000000 | 0.318440 | True | True | 0.885888 | -0.023424 |
| jump_medium | matched_protocol_recomputed | fixed_0.02 | 0.020000 | 4 | 0.263368 | 0.246523 | 0.239945 | 0.000000 | 0.318440 | True | True | 0.885886 | -0.023424 |
| jump_medium | matched_protocol_recomputed | fixed_0.05 | 0.050000 | 1 | 0.263368 | 0.246523 | 0.239976 | 0.000000 | 0.318481 | True | True | 0.883529 | -0.023393 |
| jump_medium | matched_protocol_recomputed | fixed_0.05 | 0.050000 | 2 | 0.263368 | 0.246523 | 0.240007 | 0.000000 | 0.318523 | True | True | 0.881183 | -0.023361 |
| jump_medium | matched_protocol_recomputed | fixed_0.05 | 0.050000 | 3 | 0.263368 | 0.246523 | 0.240039 | 0.000000 | 0.318565 | True | True | 0.878857 | -0.023330 |
| jump_medium | matched_protocol_recomputed | fixed_0.05 | 0.050000 | 4 | 0.263368 | 0.246523 | 0.240071 | 0.000000 | 0.318608 | True | True | 0.876549 | -0.023297 |
| jump_medium | matched_protocol_recomputed | fixed_0.10 | 0.100000 | 1 | 0.263368 | 0.246523 | 0.240555 | 0.000000 | 0.319250 | True | True | 0.850235 | -0.022814 |
| jump_medium | matched_protocol_recomputed | fixed_0.10 | 0.100000 | 2 | 0.263368 | 0.246523 | 0.241176 | 0.000000 | 0.320074 | True | True | 0.816941 | -0.022192 |
| jump_medium | matched_protocol_recomputed | fixed_0.10 | 0.100000 | 3 | 0.263368 | 0.246523 | 0.241791 | 0.000000 | 0.320890 | True | True | 0.785812 | -0.021577 |
| jump_medium | matched_protocol_recomputed | fixed_0.10 | 0.100000 | 4 | 0.263368 | 0.246523 | 0.242389 | 0.000000 | 0.321685 | True | True | 0.756667 | -0.020979 |
| jump_medium | matched_protocol_recomputed | fixed_0.20 | 0.200000 | 1 | 0.263368 | 0.246523 | 0.243320 | 0.000000 | 0.338172 | True | True | 0.722133 | -0.020048 |
| jump_medium | matched_protocol_recomputed | fixed_0.20 | 0.200000 | 2 | 0.263368 | 0.246523 | 0.246300 | 0.000000 | 0.342044 | True | True | 0.594032 | -0.017068 |
| jump_medium | matched_protocol_recomputed | fixed_0.20 | 0.200000 | 3 | 0.263368 | 0.246523 | 0.248812 | 0.000000 | 0.345370 | True | True | 0.492793 | -0.014557 |
| jump_medium | matched_protocol_recomputed | fixed_0.20 | 0.200000 | 4 | 0.263368 | 0.246523 | 0.250909 | 0.000000 | 0.348205 | True | True | 0.411988 | -0.012460 |
| combined_medium | matched_protocol_recomputed | default | 0.188138 | 1 | 0.199064 | 0.197429 | 0.194204 | 0.046620 | 0.295732 | True | True | 0.606001 | -0.004860 |
| combined_medium | matched_protocol_recomputed | default | 0.188138 | 2 | 0.199064 | 0.197429 | 0.194805 | 0.045084 | 0.297256 | True | True | 0.403306 | -0.004259 |
| combined_medium | matched_protocol_recomputed | default | 0.188138 | 3 | 0.199064 | 0.197429 | 0.195633 | 0.044980 | 0.298598 | True | True | 0.284288 | -0.003431 |
| combined_medium | matched_protocol_recomputed | default | 0.188138 | 4 | 0.199064 | 0.197429 | 0.196329 | 0.044971 | 0.299741 | True | True | 0.208521 | -0.002734 |
| combined_medium | matched_protocol_recomputed | fixed_0.02 | 0.020000 | 1 | 0.199064 | 0.197429 | 0.196858 | 0.008928 | 0.198328 | False | True | 0.986683 | -0.002206 |
| combined_medium | matched_protocol_recomputed | fixed_0.02 | 0.020000 | 2 | 0.199064 | 0.197429 | 0.196462 | 0.004806 | 0.198018 | True | True | 0.976511 | -0.002601 |
| combined_medium | matched_protocol_recomputed | fixed_0.02 | 0.020000 | 3 | 0.199064 | 0.197429 | 0.196165 | 0.005094 | 0.197760 | True | True | 0.968141 | -0.002899 |
| combined_medium | matched_protocol_recomputed | fixed_0.02 | 0.020000 | 4 | 0.199064 | 0.197429 | 0.195937 | 0.005279 | 0.197549 | True | True | 0.960975 | -0.003127 |
| combined_medium | matched_protocol_recomputed | fixed_0.05 | 0.050000 | 1 | 0.199064 | 0.197429 | 0.195396 | 0.015483 | 0.204360 | False | True | 0.925433 | -0.003667 |
| combined_medium | matched_protocol_recomputed | fixed_0.05 | 0.050000 | 2 | 0.199064 | 0.197429 | 0.194458 | 0.011649 | 0.203806 | True | True | 0.871663 | -0.004606 |
| combined_medium | matched_protocol_recomputed | fixed_0.05 | 0.050000 | 3 | 0.199064 | 0.197429 | 0.193971 | 0.010987 | 0.203444 | True | True | 0.829517 | -0.005092 |
| combined_medium | matched_protocol_recomputed | fixed_0.05 | 0.050000 | 4 | 0.199064 | 0.197429 | 0.193685 | 0.010810 | 0.203198 | True | True | 0.794856 | -0.005378 |
| combined_medium | matched_protocol_recomputed | fixed_0.10 | 0.100000 | 1 | 0.199064 | 0.197429 | 0.194247 | 0.027909 | 0.228129 | False | True | 0.793388 | -0.004817 |
| combined_medium | matched_protocol_recomputed | fixed_0.10 | 0.100000 | 2 | 0.199064 | 0.197429 | 0.193686 | 0.024790 | 0.228228 | True | True | 0.661343 | -0.005378 |
| combined_medium | matched_protocol_recomputed | fixed_0.10 | 0.100000 | 3 | 0.199064 | 0.197429 | 0.193721 | 0.024507 | 0.228495 | True | True | 0.567701 | -0.005343 |
| combined_medium | matched_protocol_recomputed | fixed_0.10 | 0.100000 | 4 | 0.199064 | 0.197429 | 0.193929 | 0.024476 | 0.228836 | True | True | 0.497095 | -0.005135 |
| combined_medium | matched_protocol_recomputed | fixed_0.20 | 0.200000 | 1 | 0.199064 | 0.197429 | 0.194260 | 0.049172 | 0.306263 | True | True | 0.586334 | -0.004803 |
| combined_medium | matched_protocol_recomputed | fixed_0.20 | 0.200000 | 2 | 0.199064 | 0.197429 | 0.194981 | 0.047654 | 0.308073 | True | True | 0.379279 | -0.004083 |
| combined_medium | matched_protocol_recomputed | fixed_0.20 | 0.200000 | 3 | 0.199064 | 0.197429 | 0.195862 | 0.047517 | 0.309622 | True | True | 0.260637 | -0.003202 |
| combined_medium | matched_protocol_recomputed | fixed_0.20 | 0.200000 | 4 | 0.199064 | 0.197429 | 0.196572 | 0.047495 | 0.310913 | True | True | 0.186774 | -0.002491 |

## Diagnostic conclusion
| degradation | conditional_source | formal_default_C2_pass | formal_default_C3_pass | formal_default_ADE_gamma | formal_default_residual_usage_ratio | best_C2_delta0_label | best_C2_delta0 | best_C2_gamma | best_C2_ADE_gamma | best_C2_C3_pass | best_C2_residual_usage_ratio | best_C2_noisy_reversion_gap | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_medium | archived_original | True | True | 0.048260 | 0.629625 | fixed_0.02 | 0.020000 | 4 | 0.047989 | True | 0.735500 | -0.014577 | C2 can pass while retaining nontrivial residual usage |
| drift_medium | archived_original | False | True | 0.023272 | 0.650923 | fixed_0.05 | 0.050000 | 2 | 0.017468 | True | 0.122906 | -0.000180 | C2 can pass mainly through residual shutdown / noisy reversion |
| burst_medium | archived_original | False | True | 0.073922 | 0.703691 | fixed_0.05 | 0.050000 | 2 | 0.060393 | True | 0.368410 | -0.014363 | C2 can pass while retaining nontrivial residual usage |
| bias_medium | archived_original | True | True | 0.187644 | 0.617825 | fixed_0.10 | 0.100000 | 3 | 0.187374 | True | 0.540397 | -0.001621 | C2 can pass mainly through residual shutdown / noisy reversion |
| jump_medium | matched_protocol_recomputed | True | True | 0.246264 | 0.601119 | fixed_0.02 | 0.020000 | 2 | 0.239945 | True | 0.885889 | -0.023424 | C2 can pass while retaining nontrivial residual usage |
| combined_medium | matched_protocol_recomputed | True | True | 0.194204 | 0.606001 | fixed_0.05 | 0.050000 | 4 | 0.193685 | True | 0.794856 | -0.005378 | C2 can pass while retaining nontrivial residual usage |

Answers:
- Does a larger gamma fix C2 for drift/burst? Yes, for drift and burst there are gamma/delta settings that pass C2.
- Does it preserve C3? In the reported best C2 settings, C3 remains true.
- Does it reduce residual_usage_ratio strongly? The table reports the residual usage for each best C2 setting; low values indicate residual shutdown.
- Does it revert toward noisy input? noisy_reversion_gap near zero indicates reversion; negative values indicate improvement over noisy.
- Does this support moving to E2 rather than extending E1? Yes. Scalar gates can only scale r_hat, not rotate or correct residual direction.

## Figures
- overall_ADE_vs_gamma: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_supplementary_gamma_delta_ablation/figures/overall_ADE_vs_gamma.png`
- high_conf_ADE_vs_gamma: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_supplementary_gamma_delta_ablation/figures/drift_burst_high_conf_ADE_vs_gamma.png`
- low_conf_ADE_vs_gamma: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_supplementary_gamma_delta_ablation/figures/drift_burst_low_conf_ADE_vs_gamma.png`
- residual_usage_ratio_vs_gamma: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_supplementary_gamma_delta_ablation/figures/residual_usage_ratio_vs_gamma.png`
- noisy_reversion_gap_vs_gamma: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_supplementary_gamma_delta_ablation/figures/noisy_reversion_gap_vs_gamma.png`
- case_c2_pass_residual_shutdown: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_supplementary_gamma_delta_ablation/figures/case_c2_pass_residual_shutdown.png`
- case_low_conf_residual_preserved: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage4/e1_supplementary_gamma_delta_ablation/figures/case_low_conf_residual_preserved.png`

## Boundary statement
Even if conservative scalar gates improve C2, they cannot rotate or correct residual direction. They only scale r_hat. Therefore this ablation is used to motivate E2 absolute-space posterior anchoring.
