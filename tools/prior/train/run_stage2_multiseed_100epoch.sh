#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
TRAIN_MODULE="tools.prior.train.train_ddpm_eth_ucy_h128"

VARIANTS=(none q10 q20 q30)
SEEDS=(2 3 4 12 13 14 22 23 24 32 33 34 42 43 44)

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "variant=${variant} seed=${seed} epochs=100"
    echo "============================================================"
    PYTHONPYCACHEPREFIX=/tmp MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl "$PYTHON_BIN" -u -m "$TRAIN_MODULE" \
      --variant "$variant" \
      --epochs 100 \
      --batch_size 128 \
      --timesteps 100 \
      --hidden_dim 128 \
      --random_seed "$seed"
  done
done
