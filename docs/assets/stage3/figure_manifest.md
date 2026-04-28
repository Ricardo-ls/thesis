# Figure Manifest

## Figures for first full review

### 1. Problem and protocol

- `overall_stage3_objective.png`
- `missing_reconstruction_task.png`
- `refinement_interface_v0_v1_v2.png`
- `obstacle_v1_layout.png`
- `two_room_v1_layout.png`

### 2. Benchmark evidence

- `full_vs_masked_comparison.png`
- `random_span_masked_ADE_mean_std.png`

### 3. Controlled coarse reconstruction

- `controlled_degradation_examples.png`
- `controlled_benchmark_metric_summary.png`

### 4. DDPM prior refinement

- `refinement_v0_v1_v2_comparison.png`
- `alpha_sweep_masked_ADE.png`
- `alpha_sweep_mean_masked_ADE.png`
- `alpha_sweep_improvement_masked_ADE.png`

### 5. Geometry extension

- `obstacle_v1_geometry_violation_summary.png`
- `two_room_v1_geometry_violation_summary.png`
- `geometry_profiles_comparison.png`

| Figure filename | Type | Source files used | Purpose | Status | Short interpretation |
| --- | --- | --- | --- | --- | --- |
| overall_stage3_objective.png | conceptual | programmatic schematic | Show the Stage 3 reconstruction/refinement objective. | generated | Stage 3 is missing indoor trajectory reconstruction, not generic forecasting. |
| missing_reconstruction_task.png | conceptual | programmatic schematic | Show the one contiguous missing-segment task definition. | generated | Missing-segment reconstruction quality should be read against the clean target. |
| refinement_interface_v0_v1_v2.png | conceptual | programmatic schematic | Explain v0/v1/v2 DDPM refinement interfaces. | generated | v0 changes the whole trajectory; v1/v2 protect observed points. |
| obstacle_v1_layout.png | conceptual | `tools/stage3/geometry_extension/run_geometry_extension.py` | Explain the obstacle geometry extension layout. | generated | `obstacle_v1` adds a central blocked region as a feasibility stress test. |
| two_room_v1_layout.png | conceptual | `tools/stage3/geometry_extension/run_geometry_extension.py` | Explain the narrow-opening room-transition layout. | generated | `two_room_v1` tightens room-transition feasibility without changing the benchmark itself. |
| full_vs_masked_comparison.png | data-result | `outputs/stage3/phase1/canonical_room3/random_span_statistics/figures/full_vs_masked_comparison.png` | Show full vs masked metric ranking differences. | copied | Full-trajectory consistency and missing-segment reconstruction quality can rank methods differently. |
| random_span_masked_ADE_mean_std.png | data-result | `outputs/stage3/phase1/canonical_room3/random_span_statistics/metrics_summary_mean_std.csv` | Show mean +- std masked_ADE over random span positions. | generated | Masked metrics are the direct view of missing-segment reconstruction quality. |
| controlled_degradation_examples.png | data-result | `outputs/stage3/controlled_benchmark/degradation/clean.npy`<br>`outputs/stage3/controlled_benchmark/degradation/mask_span20_fixed_seed42.npy`<br>`outputs/stage3/controlled_benchmark/degradation/degraded_missing_only_span20_fixed_seed42.npy`<br>`outputs/stage3/controlled_benchmark/degradation/degraded_missing_noise_span20_fixed_seed42.npy`<br>`outputs/stage3/controlled_benchmark/degradation/degraded_missing_drift_span20_fixed_seed42.npy`<br>`outputs/stage3/controlled_benchmark/degradation/degraded_missing_noise_drift_span20_fixed_seed42.npy` | Show the four controlled degradation settings. | generated | The controlled benchmark stresses reconstruction under missingness, noise, drift, and combined degradation. |
| controlled_benchmark_metric_summary.png | data-result | `outputs/stage3/controlled_benchmark/eval/metrics_summary.csv` | Summarize controlled coarse reconstruction metrics. | generated | Baseline behavior changes across degradation types; masked_ADE emphasizes the missing segment. |
| refinement_v0_v1_v2_comparison.png | data-result | `outputs/stage3/refinement/eval/refinement_metrics.csv` | Compare identity, Light SG, DDPM v0, DDPM v1, and DDPM v2 alpha=0.25. | generated | v0 proves integration, v1 protects observed points, and v2 blends the DDPM candidate into the missing span. |
| alpha_sweep_masked_ADE.png | data-result | `outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv` | Show the simplified alpha sensitivity summary by coarse method. | generated | Linear and Savitzky-Golay are best at alpha=0.00, while Kalman is best near alpha=0.10. |
| alpha_sweep_mean_masked_ADE.png | data-result | `outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv` | Show mean `masked_ADE` by alpha for the three coarse methods. | generated | The summary plot makes the main alpha trend visible at a glance, with lower values indicating better missing-segment reconstruction. |
| alpha_sweep_improvement_masked_ADE.png | data-result | `outputs/stage3/refinement/alpha_sweep/alpha_sweep_summary.csv` | Show masked_ADE improvement relative to alpha=0.00. | generated | Large alpha usually hurts, which indicates the unconditional DDPM prior is not a reliable direct refiner. |
| obstacle_v1_geometry_violation_summary.png | data-result | `outputs/stage3/geometry_extension/obstacle_v1/geometry_summary.csv` | Summarize geometry feasibility violations under `obstacle_v1`. | generated | `obstacle_v1` checks whether trajectories remain feasible around a blocked central region. |
| two_room_v1_geometry_violation_summary.png | data-result | `outputs/stage3/geometry_extension/two_room_v1/geometry_summary.csv` | Summarize geometry feasibility violations under `two_room_v1`. | generated | `two_room_v1` tests whether outputs can pass through a narrower valid transition opening. |
| geometry_profiles_comparison.png | data-result | `outputs/stage3/geometry_extension/geometry_profiles_summary.csv` | Compare profile-level window violation rates by source family. | generated | Cross-profile normalized rates show how geometry strictness changes feasibility diagnostics. |

## Missing data sources

- None

## TODO

- `input_degraded` is not present in `metrics_summary.csv`, so it remains omitted from plots that depend on that table.

## Geometry Extension Limitation

These profiles are synthetic feasibility stress tests applied after scaling ETH+UCY trajectories into `canonical_room3`.
Their violation counts should be interpreted as feasibility diagnostics under artificial layouts, not as direct evidence of real-room navigation failure.
The main geometry evaluation is computed only on the feasible clean-target subset, so normalized violation rates should be read as conditional diagnostics on that retained subset.
