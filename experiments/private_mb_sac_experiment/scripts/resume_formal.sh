#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/RL_projects/decentralized-estimator}"
EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/private_mb_sac_experiment}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/private_mb_sac_seed${SEED}}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoints/latest.pkl}"

export PYTHONPATH="$EXP_ROOT:$PROJECT_ROOT/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m private_mb_sac.train_formal   --config configs/default.yaml   --seed "$SEED"   --output-dir "$OUTPUT_DIR"   --resume "$CHECKPOINT"
