# Stage 3 Run Checklist

## Purpose

This checklist records the minimal runnable order for the current Stage 3 Phase 1 benchmark.

The clean input is fixed to:

- `datasets/processed/data_eth_ucy_20.npy`

## Run Order

### 1. Build the clean Stage 3 dataset

```bash
./.venv/bin/python tools/stage3/data/build_imputation_dataset.py
```

Expected output:

- `outputs/stage3/data/clean_windows.npz`

### 2. Generate one contiguous missing span

```bash
./.venv/bin/python tools/stage3/data/generate_missing_span.py
```

Expected output:

- `outputs/stage3/data/missing_span_windows.npz`

### 3. Build the minimal occupancy map

```bash
./.venv/bin/python tools/stage3/data/build_occupancy_map.py \
  --height 128 \
  --width 128 \
  --x_min -10 \
  --x_max 10 \
  --y_min -10 \
  --y_max 10
```

Expected output:

- `outputs/stage3/data/occupancy_map.npz`

### 4. Run linear interpolation

```bash
./.venv/bin/python tools/stage3/baselines/run_linear_interp.py
```

Expected output:

- `outputs/stage3/baselines/linear_interp_results.npz`

### 5. Run Kalman baseline

```bash
./.venv/bin/python tools/stage3/baselines/run_kalman.py \
  --dt 1.0 \
  --process_var 0.01 \
  --measure_var 0.01
```

Expected output:

- `outputs/stage3/baselines/kalman_results.npz`

### 6. Run Savitzky-Golay baseline

```bash
./.venv/bin/python tools/stage3/baselines/run_savgol.py \
  --window_length 5 \
  --polyorder 2
```

Expected output:

- `outputs/stage3/baselines/savgol_results.npz`

Note:

- this step requires `scipy`
- if `scipy` is unavailable, the script exits with a clear dependency message

### 7. Evaluate reconstruction metrics

Linear interpolation:

```bash
./.venv/bin/python tools/stage3/eval/eval_reconstruction_metrics.py \
  --pred_path outputs/stage3/baselines/linear_interp_results.npz \
  --method_name linear_interp
```

Kalman:

```bash
./.venv/bin/python tools/stage3/eval/eval_reconstruction_metrics.py \
  --pred_path outputs/stage3/baselines/kalman_results.npz \
  --method_name kalman
```

Savitzky-Golay:

```bash
./.venv/bin/python tools/stage3/eval/eval_reconstruction_metrics.py \
  --pred_path outputs/stage3/baselines/savgol_results.npz \
  --method_name savgol
```

Expected output:

- `outputs/stage3/eval/<method_name>_reconstruction_metrics.json`

### 8. Evaluate geometry metrics

Linear interpolation:

```bash
./.venv/bin/python tools/stage3/eval/eval_geometry_metrics.py \
  --pred_path outputs/stage3/baselines/linear_interp_results.npz \
  --method_name linear_interp
```

Kalman:

```bash
./.venv/bin/python tools/stage3/eval/eval_geometry_metrics.py \
  --pred_path outputs/stage3/baselines/kalman_results.npz \
  --method_name kalman
```

Savitzky-Golay:

```bash
./.venv/bin/python tools/stage3/eval/eval_geometry_metrics.py \
  --pred_path outputs/stage3/baselines/savgol_results.npz \
  --method_name savgol
```

Expected output:

- `outputs/stage3/eval/<method_name>_geometry_metrics.json`

## Boundary Reminder

This checklist is only for the current minimal benchmark:

- clean windows from `datasets/processed/data_eth_ucy_20.npy`
- one contiguous missing span
- linear interpolation / Savitzky-Golay / Kalman
- ADE / FDE / RMSE / wall-crossing count / off-map ratio

It does not include:

- prior
- q20
- raw reconstruction
- learning backbones
- multi-dataset comparison
- complex geometry conditioning
