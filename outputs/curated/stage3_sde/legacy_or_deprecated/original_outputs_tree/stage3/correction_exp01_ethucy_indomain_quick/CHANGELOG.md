# CHANGELOG

## 2026-04-29

Files created:
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/tools/stage3/correction_exp01_ethucy_indomain_quick/run_exp01.py`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/tools/stage3/correction_exp01_ethucy_indomain_quick/utils.py`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/tools/stage3/correction_exp01_ethucy_indomain_quick/README.md`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/tools/stage3/correction_exp01_ethucy_indomain_quick/CHANGELOG.md`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/config/exp01_config.json`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/raw/per_case_results.csv`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/raw/selected_cases.json`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/tables/full_stats_matrix.csv`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/tables/full_stats_matrix.md`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/tables/summary_key_metrics.md`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/median_case_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/best_ddpm_improvement_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/worst_ddpm_degradation_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/logs/run_log.txt`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/README.md`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/CHANGELOG.md`

Purpose of the experiment:
- quick in-domain sanity check for Stage 3 missing reconstruction on ETH+UCY public trajectories

Methods included:
- `linear_interp`
- `savgol_w5_p2`
- `kalman_cv_dt1.0_q1e-3_r1e-2`
- `ddpm_v3_inpainting`
- `ddpm_v3_inpainting_anchored`

Outputs generated:
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/config/exp01_config.json`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/raw/per_case_results.csv`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/raw/selected_cases.json`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/tables/full_stats_matrix.csv`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/tables/full_stats_matrix.md`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/tables/summary_key_metrics.md`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/median_case_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/best_ddpm_improvement_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/figures/worst_ddpm_degradation_five_column.png`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/logs/run_log.txt`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/README.md`
- `/Users/shangshanchong/Desktop/pytorch_env_check/trajectory_ddpm_mvp/SingularTrajectory/outputs/stage3/correction_exp01_ethucy_indomain_quick/CHANGELOG.md`

Missing methods or missing intermediate outputs:
- no method implementation was missing at runtime
- five-column figures use `linear_interp` as the reference coarse column because the current v3 interface does not expose a coarse-dependent internal stage
