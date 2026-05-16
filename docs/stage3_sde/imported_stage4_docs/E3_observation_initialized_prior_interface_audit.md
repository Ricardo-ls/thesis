# E3 Observation-Initialized Prior Interface Audit

Date: 2026-05-15

Scope: path, configuration, and interface audit only. No E3 experiment was run. No model was trained. No checkpoint was modified.

## Executive Answer

Stage 2 unconditional trajectory prior is interface-compatible with a minimal observation-initialized diffusion refinement, but the repository does not currently expose this as a ready-to-run function.

Minimal E3 is feasible as a new implementation if it adds:

1. a reverse loop that starts from a supplied `x_t` instead of pure Gaussian noise;
2. a helper to convert `y_abs -> delta_y -> [B, 2, 19]`;
3. reconstruction helper `delta_x -> anchor y_abs[0] -> x_abs`;
4. explicit scale/domain checks because Stage 2 prior is ETH+UCY relative displacement and is not Stage 3 indoor-normalized.

## 1. Stage 2 Unconditional DDPM Checkpoints

Found.

Primary Stage 2 current mainline checkpoint candidates are under:

- `outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-100epoch/best_model.pt`
- `outputs/prior/train/ddpm_eth_ucy_q10_h128/seed42-100epoch/best_model.pt`
- `outputs/prior/train/ddpm_eth_ucy_q20_h128/seed42-100epoch/best_model.pt`
- `outputs/prior/train/ddpm_eth_ucy_q30_h128/seed42-100epoch/best_model.pt`

The current Stage 2 documentation says the mainline evidence is a 15-seed, 100-epoch screen, with `none` as optimization-best and `q10` as closest secondary candidate. See:

- `docs/prior_stage2.md`
- `docs/stage2_phaseA_multiseed_100epoch_report.md`
- `outputs/prior/train/README.md`

The inspected `none` seed42 checkpoint has:

- path: `outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-100epoch/best_model.pt`
- epoch: `68`
- best_val_loss: `0.08729857769655999`
- config:
  - variant: `none`
  - batch_size: `128`
  - epochs: `100`
  - timesteps: `100`
  - hidden_dim: `128`
  - rel_data_path: `datasets/processed/data_eth_ucy_20_rel.npy`

Legacy / older checkpoint also exists:

- `outputs/ddpm_eth_ucy_q20_h128/best_model.pt`

It is a 20-epoch q20 snapshot and should be treated as older/legacy relative to the curated `outputs/prior/train/` Stage 2 evidence layer.

## 2. Representation

Stage 2 unconditional prior representation is relative displacement.

Evidence:

- `docs/stage2_phaseA_multiseed_100epoch_report.md` states representation is `19 x 2` relative-step trajectory windows.
- `tools/prior/data/build_eth_ucy_dataset.py` builds `rel_data = np.diff(abs_data, axis=1)`.
- `datasets/traj_dataset.py` loads `.npy` arrays with shape `(N, T, 2)`.
- `tools/prior/train/train_ddpm_eth_ucy_h128.py` uses `TrajectoryDataset(rel_data_path)` and trains on `x0 = batch.permute(0, 2, 1)`.

It is not an absolute-coordinate prior.

It is not a normalized-relative-displacement prior in the Stage 3 sense. The Stage 2 training code directly uses the raw processed ETH+UCY relative displacement array.

Observed dataset scale:

- `datasets/processed/data_eth_ucy_20_rel.npy`: shape `(36073, 19, 2)`, dtype `float32`, std approximately `0.2168`
- `datasets/processed/data_eth_ucy_20_rel_q20.npy`: shape `(28858, 19, 2)`, dtype `float32`, std approximately `0.2422`

For comparison, Stage 3 indoor v2 normalization exists separately:

- `data/stage3_indoor/rel_norm_params_v2.npz`
- rel_std approximately `[0.1478053, 0.14835621]`

This Stage 3 normalization must not be silently applied to Stage 2 prior inputs.

## 3. Input Shape

Model input shape is:

```text
[B, 2, 19]
```

Dataset storage shape is:

```text
[B, 19, 2]
```

Training conversion:

```python
x0 = batch.permute(0, 2, 1)
```

Model:

- `models/temporal_denoiser.py`
- class: `TemporalDenoiser1D`
- constructor for Stage 2 h128: `TemporalDenoiser1D(max_timesteps=100, in_channels=2, hidden_dim=128)`
- forward input: `x: [B, 2, 19]`, `t: [B]`
- output: predicted noise with shape `[B, 2, 19]`

## 4. Reverse Sampling Interface

Existing reverse sampling exists.

Relevant files:

- `tools/prior/sample/reverse_sample_ddpm_eth_ucy_h128.py`
- `tools/prior/sample/reverse_sample_ddpm_h128.py`

The current h128 official sampler starts from pure Gaussian noise:

```python
x = torch.randn(num_generate, channels, seq_len, device=device)
for t in reversed(range(timesteps)):
    ...
```

The older `DDPMSampler` helper in `tools/prior/sample/reverse_sample_ddpm_h128.py` exposes:

- `q_sample(x0, t, noise=None)`
- `predict_x0_from_eps(xt, t, eps_pred)`
- `p_sample(model, xt, t_scalar)`
- `sample(model, num_samples, channels=2, seq_len=19, return_history=False)`

However:

- existing public `sample()` starts only from Gaussian noise;
- there is no dedicated function like `sample_from_xt(model, x_t, start_t)` or `reverse_from_xt(...)`;
- a minimal E3 implementation must add this wrapper.

The building blocks are present because `p_sample(model, xt, t_scalar)` accepts an arbitrary `xt`. A reverse loop from supplied `x_t` at timestep `t_start` is therefore straightforward, but not currently packaged.

## 5. Forward Noising / q_sample

Forward noising exists.

Primary implementation:

- `diffusion/ddpm_utils.py`
- class: `DDPMForwardProcess`
- method: `q_sample(x0, t, noise=None)`

Signature:

```python
q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None)
```

Expected shape:

```text
x0: [B, C, L] = [B, 2, 19]
t:  [B]
```

This can directly noise an observation-derived relative trajectory:

```text
y_abs [B,20,2]
-> delta_y [B,19,2]
-> delta_y_ch [B,2,19]
-> q_sample(delta_y_ch, t)
```

The older sampler in `tools/prior/sample/reverse_sample_ddpm_h128.py` also includes a `q_sample` method with the same DDPM formula.

## 6. Normalization / Denormalization

Stage 2 prior training does not use explicit scale normalization or saved mean/std normalization.

Evidence:

- `TrajectoryDataset` loads `.npy` arrays as `float32`.
- `train_ddpm_eth_ucy_h128.py` directly uses `batch.permute(0,2,1)` as `x0`.
- No Stage 2 `rel_norm_params*.npz` file was found under `outputs/prior/` or `datasets/processed/`.
- Checkpoint configs contain data paths and hyperparameters but no normalization statistics.

Therefore, for observation-initialized E3 using Stage 2 prior:

- do not use Stage 3 `rel_norm_params_v2.npz`;
- feed raw relative displacements in the same coordinate units as the Stage 2 processed ETH+UCY data;
- after reverse denoising, transpose `[B,2,19] -> [B,19,2]`;
- reconstruct absolute coordinates with the observation anchor `y_abs[:,0,:]`.

Important caveat:

Stage 4 indoor trajectories and Stage 2 ETH+UCY trajectories may have different scale/domain distributions. Interface feasibility does not imply scientific compatibility.

## 7. Minimal E3 Observation-Initialized Refinement Feasibility

The requested pipeline is feasible with small helper additions:

```text
absolute y_abs
-> delta_y = y_abs[:,1:,:] - y_abs[:,:-1,:]
-> delta_y_ch = delta_y.transpose(0,2,1)
-> q_sample(delta_y_ch, t_start)
-> reverse denoise from supplied x_t and t_start
-> delta_x_hat_ch
-> delta_x_hat = delta_x_hat_ch.transpose(0,2,1)
-> x_abs_hat[:,0,:] = y_abs[:,0,:]
-> x_abs_hat[:,1:,:] = y_abs[:,0:1,:] + cumsum(delta_x_hat)
```

Required helper functions:

1. `to_relative(y_abs) -> [B,19,2]`
2. `to_model_shape(rel) -> [B,2,19]`
3. `reverse_from_xt(model, diffusion_or_sampler, x_t, start_t)`
4. `rel_to_abs_with_anchor(rel, anchor=y_abs[:,0,:])`
5. checkpoint loader for `TemporalDenoiser1D(max_timesteps=100, in_channels=2, hidden_dim=128)`

No new model training is required for this interface test.

## 8. Direct Answers

1. Stage 2 unconditional DDPM checkpoint exists: yes. The most relevant current-mainline path is `outputs/prior/train/ddpm_eth_ucy_none_h128/seed42-100epoch/best_model.pt`; q10/q20/q30 variants also exist.

2. Representation: raw relative displacement, stored as `[N,19,2]` and fed to the model as `[B,2,19]`.

3. Input shape: model input `[B,2,19]`; dataset file shape `[N,19,2]`.

4. Reverse sampling function: yes, but public sampler starts from Gaussian noise. A lower-level `p_sample(model, xt, t_scalar)` exists and can support supplied `x_t`; a dedicated `reverse_from_xt` wrapper is missing.

5. Forward noising `q_sample`: yes. `DDPMForwardProcess.q_sample(x0,t)` supports noising `delta_y` after converting to `[B,2,19]`.

6. Normalization: Stage 2 prior has no explicit saved normalization. Observation-initialized `delta_y` should be raw relative displacement in compatible units; do not use Stage 3 normalization unless a new, explicitly audited adaptation layer is introduced.

7. Minimal E3 observation-initialized refinement: interface-feasible. It needs helper wrappers but not retraining. Main risk is domain/scale mismatch between Stage 2 ETH+UCY prior and Stage 4 indoor trajectories.

## 9. Go / No-Go for Implementing E3

Interface status:

```text
E3_INTERFACE_FEASIBLE = TRUE
```

Blocking implementation gaps:

- missing `reverse_from_xt(...)` helper;
- missing E3-specific reconstruction helper with anchor `y_abs[0]`;
- missing explicit scale/domain check between Stage 2 prior relative-displacement scale and Stage 4 indoor relative-displacement scale.

Non-blocking but important scientific caveat:

- Stage 2 prior is an ETH+UCY motion prior, not the Stage 3 indoor prior. E3 should be framed as an observation-initialized prior-interface test, not as a guaranteed improvement over E1/E2.

