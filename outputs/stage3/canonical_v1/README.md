# stage3_canonical_v1

This is the fixed Stage 3 canonical protocol.
Old Stage 3 directories are historical and not primary evidence.
All report tables and figures should be regenerated from `outputs/stage3/canonical_v1`.

Protocol summary:
- dataset: `canonical_room3` from `outputs/stage3/phase1/canonical_room3/data/clean_windows_room3.npz`
- fixed conditions: span10_fixed_seed42, span20_fixed_seed42, span30_fixed_seed42, span20_random_seed42, span20_random_seed43, span20_random_seed44
- methods: linear_interp, savgol_w5_p2, kalman_cv_dt1.0_q1e-3_r1e-2, ddpm_v3_inpainting, ddpm_v3_inpainting_anchored
- metrics: ADE, FDE, RMSE, masked_ADE, masked_RMSE, endpoint_error, path_length_error, acceleration_error, off_map_ratio, wall_crossing_count
- deterministic baselines and DDPM seed-level results have different N definitions
- trajectory-level aggregation is the fair comparison table
- `seed_best` is an oracle diagnostic only and must not be used as the main comparison conclusion

Execution note:
- current fixed runnable subset uses the first `1024` canonical_room3 trajectories so the DDPM seed-level evidence layer is reproducible in the local CPU environment

DDPM prior:
- objective: `optimization_best`
- variant: `none`
- checkpoint: `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-100epoch/best_model.pt`

Generated outputs:
- seed-level rows: `79872`
- trajectory-level rows: `67584`
- missing-cell count: `0`

