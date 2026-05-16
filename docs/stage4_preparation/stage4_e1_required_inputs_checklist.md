# Stage 4 E1 Required Inputs Checklist

## Required for E1

| Item | Status | Path or source | Notes | Blocking issue |
|---|---|---|---|---|
| clean evaluation trajectories | FOUND | `data/stage3_indoor/clean_trajs.npy` | Stage 3 reports use paired N=200 subsets from clean trajectories; val_trajs.npy also exists for held-out pool. | no |
| degraded trajectories or degradation generator | FOUND | `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_bias.npy, outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_burst.npy, outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_combined.npy, outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_drift.npy, outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_gaussian.npy, outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_degraded_jump.npy; generator=tools/stage3_indoor/degrade.py` | N=200 generalization degraded arrays are present; data/stage3_indoor also has N=1000 degraded_*_medium.npy arrays. | no |
| degradation names | FOUND | `bias_medium, burst_medium, combined_medium, drift_medium, gaussian_medium, jump_medium` | Names inferred from degraded array filenames. | no |
| noisy_input baseline | FOUND | `outputs/stage3_indoor/conditional_residual_ddpm_gaussian/seed42/generalization_summary.csv` | Rows with method=noisy_input provide baseline ADE/RMSE/smooth metrics by degradation. | no |
| Stage 3 indoor prior checkpoint | FOUND | `outputs/stage3_indoor/ddpm_indoor_v2/seed42/best_ema_model.pt` | Preferred retrained indoor v2 EMA checkpoint for SDEdit diagnostics. | no |
| normalization statistics | FOUND | `outputs/stage3_indoor/ddpm_indoor_v2/seed42/rel_norm_params_v2.npz` | rel_norm_params_v2.npz is colocated with indoor v2 prior and conditional model outputs. | no |
| diffusion schedule | FOUND | `outputs/stage3_indoor/ddpm_indoor_v2/seed42/prior_check_v2.json` | timesteps=100; beta schedule details may require reading training code. | no |
| SDEdit start step tau_star | FOUND | `outputs/stage3_indoor/ddpm_indoor_v2/seed42/sdedit_gaussian_full_conclusion.json` | best t_start=2; known Stage 3 best t_start is 2. | no |
| evaluation seeds | FOUND | `outputs/stage3_indoor/report/captions.txt` | DDPM inference seeds=[42, 43, 44, 45, 46]; degradation seed conventions also exist in degradation_config.json. | no |
| trajectory length T | FOUND | `data/stage3_indoor/clean_trajs.npy` | T=20 frames. | no |
| coordinate representation | FOUND | `data/stage3_indoor/*.npy and rel_norm_params_v2.npz` | absolute coordinates in 3m x 3m room for clean/degraded arrays; relative displacement normalization for DDPM internals | no |

## E1 Go/No-Go Decision

E1_READY = TRUE

No blocking required input is missing. Ambiguities should still be reviewed before implementation.
