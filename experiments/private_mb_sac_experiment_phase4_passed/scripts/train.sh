#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m private_mb_sac.train --config configs/default.yaml
