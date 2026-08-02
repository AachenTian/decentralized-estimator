#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/RL_projects/decentralized-estimator}"
EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/private_mb_sac_experiment}"

export PYTHONPATH="$EXP_ROOT:$PROJECT_ROOT/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m private_mb_sac.train   --config configs/default.yaml   --sac-smoke-test   --smoke-rounds 1   --smoke-num-envs 2   --dynamics-smoke-updates 10   --dynamics-smoke-batch-size 32   --model-smoke-batch-size 32   --model-smoke-horizon 2   --sac-smoke-updates 10   --sac-smoke-batch-size 32   --calibration-smoke-rounds 1   --output-dir outputs/phase4_independent_sac_smoke
