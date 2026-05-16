# FDE Zero Audit

Question: is FDE equal to zero because endpoint clamping or endpoint preservation forces the last frame to match ground truth?

Short answer:
- `linear_interp`: YES, FDE is exactly zero because the last frame is observed and linear interpolation preserves observed boundary points.
- `ddpm_v3_inpainting_anchored`: YES, FDE is exactly zero because the anchoring step restores all observed points, including the final frame.
- `savgol_w5_p2`: NO, FDE is not forced to zero because smoothing perturbs observed points after gap filling.
- `kalman_cv_dt1.0_q1e-3_r1e-2`: NO, FDE is not forced to zero because the Kalman rollout estimates all frames and does not hard-clamp the final observation.
- `ddpm_v3_inpainting`: NO, FDE is not forced to zero because the raw DDPM output is not post-clamped in absolute endpoint space.

Evidence from code:
- `build_obs_mask` only removes an interior contiguous span, so the final frame remains observed.
- `run_linear_interp` and `run_savgol.validate_and_interp` require the sequence to start and end with observations.
- `anchor_missing_spans` writes `traj[observed_mask] = observed_abs[...]`, which restores every observed point exactly.
- `ddpm_prior_inpainting_v3` clamps known relative displacements during reverse diffusion, but it does not directly hard-clamp the final absolute endpoint after absolute reconstruction.

Observed FDE behavior in canonical_v1 seed-level outputs:

## span10_fixed_seed42
- `linear_interp`: N=1024, zero_count=1024, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000
- `savgol_w5_p2`: N=1024, zero_count=0, zero_fraction=0.000000, mean=0.001646, max_abs=0.014926
- `kalman_cv_dt1.0_q1e-3_r1e-2`: N=1024, zero_count=3, zero_fraction=0.002930, mean=0.010179, max_abs=0.102239
- `ddpm_v3_inpainting`: N=5120, zero_count=0, zero_fraction=0.000000, mean=0.090156, max_abs=0.876818
- `ddpm_v3_inpainting_anchored`: N=5120, zero_count=5120, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000

## span20_fixed_seed42
- `linear_interp`: N=1024, zero_count=1024, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000
- `savgol_w5_p2`: N=1024, zero_count=0, zero_fraction=0.000000, mean=0.001646, max_abs=0.014926
- `kalman_cv_dt1.0_q1e-3_r1e-2`: N=1024, zero_count=2, zero_fraction=0.001953, mean=0.010047, max_abs=0.101288
- `ddpm_v3_inpainting`: N=5120, zero_count=0, zero_fraction=0.000000, mean=0.238888, max_abs=2.302333
- `ddpm_v3_inpainting_anchored`: N=5120, zero_count=5120, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000

## span30_fixed_seed42
- `linear_interp`: N=1024, zero_count=1024, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000
- `savgol_w5_p2`: N=1024, zero_count=0, zero_fraction=0.000000, mean=0.001646, max_abs=0.014926
- `kalman_cv_dt1.0_q1e-3_r1e-2`: N=1024, zero_count=2, zero_fraction=0.001953, mean=0.009846, max_abs=0.098954
- `ddpm_v3_inpainting`: N=5120, zero_count=0, zero_fraction=0.000000, mean=0.506844, max_abs=4.372342
- `ddpm_v3_inpainting_anchored`: N=5120, zero_count=5120, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000

## span20_random_seed42
- `linear_interp`: N=1024, zero_count=1024, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000
- `savgol_w5_p2`: N=1024, zero_count=3, zero_fraction=0.002930, mean=0.001533, max_abs=0.014753
- `kalman_cv_dt1.0_q1e-3_r1e-2`: N=1024, zero_count=2, zero_fraction=0.001953, mean=0.009995, max_abs=0.102888
- `ddpm_v3_inpainting`: N=5120, zero_count=0, zero_fraction=0.000000, mean=0.309035, max_abs=2.719554
- `ddpm_v3_inpainting_anchored`: N=5120, zero_count=5120, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000

## span20_random_seed43
- `linear_interp`: N=1024, zero_count=1024, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000
- `savgol_w5_p2`: N=1024, zero_count=3, zero_fraction=0.002930, mean=0.001515, max_abs=0.014926
- `kalman_cv_dt1.0_q1e-3_r1e-2`: N=1024, zero_count=2, zero_fraction=0.001953, mean=0.009907, max_abs=0.102888
- `ddpm_v3_inpainting`: N=5120, zero_count=0, zero_fraction=0.000000, mean=0.295447, max_abs=2.659517
- `ddpm_v3_inpainting_anchored`: N=5120, zero_count=5120, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000

## span20_random_seed44
- `linear_interp`: N=1024, zero_count=1024, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000
- `savgol_w5_p2`: N=1024, zero_count=1, zero_fraction=0.000977, mean=0.001524, max_abs=0.014926
- `kalman_cv_dt1.0_q1e-3_r1e-2`: N=1024, zero_count=3, zero_fraction=0.002930, mean=0.009915, max_abs=0.101075
- `ddpm_v3_inpainting`: N=5120, zero_count=0, zero_fraction=0.000000, mean=0.306116, max_abs=3.164696
- `ddpm_v3_inpainting_anchored`: N=5120, zero_count=5120, zero_fraction=1.000000, mean=0.000000, max_abs=0.000000

Verdict:
- FDE is exactly zero for `linear_interp` and `ddpm_v3_inpainting_anchored` because endpoint preservation restores the final observed frame.
- FDE is not globally zero for the other methods, so there is no universal endpoint-clamping artifact across the whole protocol.
