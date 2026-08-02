#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD:${PROJECT_ROOT:-$PWD/../..}/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m private_mb_sac.train \
  --config configs/default.yaml \
  --smoke-test \
  --smoke-rounds 1 \
  --smoke-num-envs 2 \
  --output-dir outputs/phase1_smoke
