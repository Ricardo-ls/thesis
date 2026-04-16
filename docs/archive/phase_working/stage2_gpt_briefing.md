# Stage 2 GPT Briefing

This file is a factual handoff for writing Stage 2 documentation. It only records paths, public assets, and verified run metadata.

## Repository Scope

- Project: trajectory DDPM thesis codebase
- Stage 2 focus: trajectory-only diffusion prior pre-training
- Stage 3 and downstream sensor-conditioned work are separate

## Official Registry Semantics

The registry in [`utils/prior/ablation_paths.py`](../utils/prior/ablation_paths.py) is the source of truth.

- `optimization_best -> none`
- `motion_balanced -> q20`

Official Stage 2 roles:

- `none`: optimization-best baseline
- `q10`: weak-filter reference point
- `q20`: most balanced motion-focused prior among filtered variants
- `q30`: strong-filter reference point

## Current Mainline Evidence

The current Stage 2 mainline evidence is the completed multi-seed `100`-epoch screening result.

Archived Phase A seed set:

- `2`
- `3`
- `4`
- `12`
- `13`
- `14`
- `22`
- `23`
- `24`
- `32`
- `33`
- `34`
- `42`
- `43`
- `44`

## Official Seeded Reference Runs

Fixed public figure protocol:

- `sample_seed = 42`
- `vis_seed = 42`
- `num_generate = 512`
- `num_show = 16`
- `denoise_selection = endpoint_quantile`
- `denoise_quantile = 0.5`
- `reference_tag = reference_seed42`
- `device = cpu`

Reference run output layout:

- `outputs/prior/sample/ddpm_eth_ucy_none_h128/reference_seed42/`
- `outputs/prior/sample/ddpm_eth_ucy_q10_h128/reference_seed42/`
- `outputs/prior/sample/ddpm_eth_ucy_q20_h128/reference_seed42/`
- `outputs/prior/sample/ddpm_eth_ucy_q30_h128/reference_seed42/`

Eval output layout:

- `outputs/prior/eval/ddpm_eth_ucy_none_h128/reference_seed42/`
- `outputs/prior/eval/ddpm_eth_ucy_q10_h128/reference_seed42/`
- `outputs/prior/eval/ddpm_eth_ucy_q20_h128/reference_seed42/`
- `outputs/prior/eval/ddpm_eth_ucy_q30_h128/reference_seed42/`

Public assets layout:

- `docs/assets/stage2/none/reference_seed42/`
- `docs/assets/stage2/q10/reference_seed42/`
- `docs/assets/stage2/q20/reference_seed42/`
- `docs/assets/stage2/q30/reference_seed42/`

Each public asset directory contains:

- `sample/real_vs_generated.png`
- `sample/denoise_check.png`
- `sample/manifest.json`
- `eval/summary_metrics.csv`
- `eval/manifest.json`
- `eval/*.png`

## Verified Records

These legacy records remain available for traceability only.

- historical single-seed `50`-epoch references remain in the archive
- public seeded reference runs remain fixed at `sample_seed=42` and `vis_seed=42`
- the current mainline evidence should be read from [`docs/stage2_phaseA_multiseed_100epoch_report.md`](stage2_phaseA_multiseed_100epoch_report.md)

## Selected Public Figure Paths

### Quality checks

- `docs/assets/stage2/none/reference_seed42/sample/real_vs_generated.png`
- `docs/assets/stage2/q20/reference_seed42/sample/real_vs_generated.png`
- `docs/assets/stage2/none/reference_seed42/sample/denoise_check.png`
- `docs/assets/stage2/q20/reference_seed42/sample/denoise_check.png`

### Distribution diagnostics

- `docs/assets/stage2/none/reference_seed42/eval/hist_endpoint_displacement.png`
- `docs/assets/stage2/q20/reference_seed42/eval/hist_endpoint_displacement.png`
- `docs/assets/stage2/q20/reference_seed42/eval/hist_propulsion_ratio.png`
- `docs/assets/stage2/q20/reference_seed42/eval/hist_acc_rms.png`

### Legacy prior figures still present in docs

- `docs/assets/prior/eval_none_hist_endpoint_displacement.png`
- `docs/assets/prior/eval_none_hist_propulsion_ratio.png`
- `docs/assets/prior/eval_q20/hist_endpoint_displacement.png`
- `docs/assets/prior/eval_q20/hist_propulsion_ratio.png`
- `docs/assets/prior/eval_q20/hist_acc_rms.png`

## Reproducibility Notes

- Public reference figures are seeded reference runs, not training reruns.
- Fixed seeds are for sample and visualization reproducibility.
- The public workflow does not change model architecture, training protocol, sampling math, or metric definitions.
- Public repository contains code, selected figures, manifests, and lightweight docs only.
- No raw pedestrian trajectory files, processed corpora, checkpoints, or large outputs are committed.
