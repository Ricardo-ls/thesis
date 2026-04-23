# Stage 3 Phase 1 Runbook

Run from the `SingularTrajectory/` repository root:

```bash
source ../.venv/bin/activate
```

## 1. Build Canonical Room3 Clean Data

```bash
python tools/stage3/data/build_imputation_dataset.py
```

Expected outputs:

- `outputs/stage3/phase1/canonical_room3/data/clean_windows_room3.npz`
- `outputs/stage3/phase1/canonical_room3/data/clean_windows_room3_meta.json`

## 2. Build Canonical Room3 Empty Map

```bash
python tools/stage3/data/build_occupancy_map.py
```

Expected output:

- `outputs/stage3/phase1/canonical_room3/data/occupancy_map_room3_empty.npz`

## 3. Generate Missing-Span Experiments

```bash
python tools/stage3/data/generate_missing_span.py \
  --experiment_id span20_fixed_seed42 \
  --span_ratio 0.2 \
  --span_mode fixed \
  --seed 42
```

```bash
python tools/stage3/data/generate_missing_span.py \
  --experiment_id span10_fixed_seed42 \
  --span_ratio 0.1 \
  --span_mode fixed \
  --seed 42
```

```bash
python tools/stage3/data/generate_missing_span.py \
  --experiment_id span30_fixed_seed42 \
  --span_ratio 0.3 \
  --span_mode fixed \
  --seed 42
```

```bash
python tools/stage3/data/generate_missing_span.py \
  --experiment_id span20_random_seed42 \
  --span_ratio 0.2 \
  --span_mode random \
  --seed 42
```

```bash
python tools/stage3/data/generate_missing_span.py \
  --experiment_id span20_random_seed43 \
  --span_ratio 0.2 \
  --span_mode random \
  --seed 43
```

```bash
python tools/stage3/data/generate_missing_span.py \
  --experiment_id span20_random_seed44 \
  --span_ratio 0.2 \
  --span_mode random \
  --seed 44
```

Expected output pattern:

- `outputs/stage3/phase1/canonical_room3/data/experiments/<experiment_id>/missing_span_windows.npz`

## 4. Run Baselines

For each `<experiment_id>`, run:

```bash
python tools/stage3/baselines/run_linear_interp.py \
  --experiment_id <experiment_id>
```

```bash
python tools/stage3/baselines/run_savgol.py \
  --experiment_id <experiment_id> \
  --window_length 5 \
  --polyorder 2
```

```bash
python tools/stage3/baselines/run_kalman.py \
  --experiment_id <experiment_id> \
  --dt 1.0 \
  --process_var 1e-3 \
  --measure_var 1e-2
```

Expected output pattern:

- `outputs/stage3/phase1/canonical_room3/baselines/<experiment_id>/<method_tag>/results.npz`

## 5. Evaluate Reconstruction

For each `<experiment_id>` and `<method_tag>`, run:

```bash
python tools/stage3/eval/eval_reconstruction_metrics.py \
  --experiment_id <experiment_id> \
  --method_tag <method_tag>
```

Method tags:

- `linear_interp`
- `savgol_w5_p2`
- `kalman_cv_dt1.0_q1e-3_r1e-2`

Expected output pattern:

- `outputs/stage3/phase1/canonical_room3/eval/<experiment_id>/<method_tag>/reconstruction_metrics.json`

## 6. Evaluate Geometry

For each `<experiment_id>` and `<method_tag>`, run:

```bash
python tools/stage3/eval/eval_geometry_metrics.py \
  --experiment_id <experiment_id> \
  --method_tag <method_tag>
```

Expected output pattern:

- `outputs/stage3/phase1/canonical_room3/eval/<experiment_id>/<method_tag>/geometry_metrics.json`

## Fixed Experiment Groups

Experiment 0, main table:

- `span20_fixed_seed42`

Experiment 1, missing-length sweep:

- `span10_fixed_seed42`
- `span20_fixed_seed42`
- `span30_fixed_seed42`

Experiment 2, missing-position control:

- `span20_fixed_seed42`
- `span20_random_seed42`
- `span20_random_seed43`
- `span20_random_seed44`
