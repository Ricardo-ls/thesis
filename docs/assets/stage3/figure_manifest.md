# Figure Manifest

## Figures for first full review

### 1. Problem and protocol

- `overall_stage3_objective.png`
- `missing_reconstruction_task.png`
- `refinement_interface_v0_v1_v2.png`
- `wall_door_v1_layout.png`

### 2. Benchmark evidence

- `full_vs_masked_comparison.png`
- `random_span_masked_ADE_mean_std.png`

### 3. Controlled coarse reconstruction

- `controlled_degradation_examples.png`
- `controlled_benchmark_metric_summary.png`

### 4. DDPM prior refinement

- `refinement_v0_v1_v2_comparison.png`
- `alpha_sweep_masked_ADE.png`
- `alpha_sweep_improvement_masked_ADE.png`

### 5. Geometry extension wall_door_v1

- `geometry_violation_summary.png`

| Figure filename | Type | Source files used | Purpose | Status | Short interpretation |
| --- | --- | --- | --- | --- | --- |
| overall_stage3_objective.png | conceptual | programmatic schematic | Show the Stage 3 reconstruction/refinement objective. | generated | Stage 3 is missing indoor trajectory reconstruction, not generic forecasting. |
| missing_reconstruction_task.png | conceptual | programmatic schematic | Show the one contiguous missing-segment task definition. | generated | Missing-segment reconstruction quality should be read against the clean target. |
| refinement_interface_v0_v1_v2.png | conceptual | programmatic schematic | Explain v0/v1/v2 DDPM refinement interfaces. | generated | v0 changes the whole trajectory; v1/v2 protect observed points. |
| wall_door_v1_layout.png | conceptual | programmatic schematic | Explain the wall-door geometry extension layout. | generated | canonical_room3 stays fixed while wall_door_v1 adds feasibility constraints. |
| full_vs_masked_comparison.png | data-result | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/phase1/canonical_room3/random_span_statistics/figures/full_vs_masked_comparison.png | Show full vs masked metric ranking differences. | copied | Full-trajectory consistency and missing-segment reconstruction quality can rank methods differently. |
| random_span_masked_ADE_mean_std.png | data-result | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/phase1/canonical_room3/random_span_statistics/metrics_summary_mean_std.csv | Show mean ± std masked_ADE over random span positions. | generated | Masked metrics are the direct view of missing-segment reconstruction quality. |
| controlled_degradation_examples.png | data-result | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/controlled_benchmark/degradation/clean.npy, /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/controlled_benchmark/degradation/mask_span20_fixed_seed42.npy, /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/controlled_benchmark/degradation/degraded_missing_only_span20_fixed_seed42.npy, /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/controlled_benchmark/degradation/degraded_missing_noise_span20_fixed_seed42.npy, /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/controlled_benchmark/degradation/degraded_missing_drift_span20_fixed_seed42.npy, /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/controlled_benchmark/degradation/degraded_missing_noise_drift_span20_fixed_seed42.npy | Show the four controlled degradation settings. | generated | The controlled benchmark stresses reconstruction under missingness, noise, drift, and combined degradation. |
| controlled_benchmark_metric_summary.png | data-result | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/controlled_benchmark/eval/metrics_summary.csv | Summarize controlled coarse reconstruction metrics. | generated | Baseline behavior changes across degradation types; masked_ADE emphasizes the missing segment. |
| refinement_v0_v1_v2_comparison.png | data-result | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/refinement/eval/refinement_metrics.csv | Compare identity, Light SG, DDPM v0, DDPM v1, and DDPM v2 alpha=0.25. | generated | v0 proves integration, v1 protects observed points, and v2 blends the DDPM candidate into the missing span. |
| alpha_sweep_masked_ADE.png | data-result | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv | Show alpha sensitivity by coarse method. | generated | Linear and Savitzky-Golay prefer alpha=0.00, while Kalman benefits only from very small alpha. |
| alpha_sweep_improvement_masked_ADE.png | data-result | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv | Show masked_ADE improvement relative to alpha=0.00. | generated | Large alpha usually hurts, which indicates the unconditional DDPM prior is not a reliable direct refiner. |
| geometry_violation_summary.png | data-result | /Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/geometry_extension/wall_door_v1/geometry_summary.csv | Summarize geometry feasibility violations under wall_door_v1. | generated | wall_door_v1 reveals how often existing outputs rely on infeasible indoor transitions. |

## Missing data sources

- None

## TODO

- input_degraded is not present in metrics_summary.csv, so it is omitted from this plot.
