#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/RL_projects/decentralized-estimator}"
EXP_ROOT="${EXP_ROOT:-$PROJECT_ROOT/experiments/private_mb_sac_experiment}"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$EXP_ROOT:$PROJECT_ROOT/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m pytest -q
