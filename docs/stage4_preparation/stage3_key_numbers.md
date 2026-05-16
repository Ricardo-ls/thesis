# Stage 3 Key Numbers

All numbers were extracted from existing Stage 3 outputs only.

## A. Matched Gaussian Diagnostic

Source: `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/cond_residual_gaussian_summary.csv`

| Quantity | ADE |
|---|---:|
| noisy_input | 0.0626 |
| Kalman CV | 0.0850 |
| Unconditional SDEdit t=2 | 0.0608 |
| Conditional residual DDPM t=20 | 0.0557 |

## B. Cross-Degradation Diagnostic

Source: `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_summary.csv`

| Degradation | noisy_input ADE | Uncond. SDEdit ADE | Cond. residual DDPM ADE | Conditional p-value |
|---|---:|---:|---:|---:|
| bias_medium | 0.1890 | 0.1893 | 0.1889 | 0.440 |
| burst_medium | 0.0748 | 0.0828 | 0.0919 | 1.000 |
| combined_medium | 0.1991 | 0.1990 | 0.1974 | 0.232 |
| drift_medium | 0.0176 | 0.0195 | 0.0316 | 1.000 |
| gaussian_medium | 0.0626 | 0.0608 | 0.0557 | 5.335e-11 |
| jump_medium | 0.2634 | 0.2572 | 0.2465 | 2.405e-14 |

## C. Receptive-Field Expansion

Sources: `outputs/stage3_indoor/receptive_field_expansion/param_count.txt`, `outputs/stage3_indoor/receptive_field_expansion/ablation_summary.md`

| Quantity | Value |
|---|---:|
| current 2-block parameter count | 212354 |
| expanded 4-block parameter count | 409474 |
| current RF including projection | 13 |
| expanded RF including projection | 21 |
| median relative ADE improvement (%) | 0.73 |
| degradations where 4-block beats 2-block | 4/6 |
| degradations where 4-block beats noisy_input | 1/6 |
