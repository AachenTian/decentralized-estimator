#!/usr/bin/env bash
set -euo pipefail

# Two-worker launcher for run_sender_baseline_comparison.py.
# Default total: 600 paired runs.

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-run_sender_baseline_comparison.py}"
ROOT="${OUTPUT_ROOT:-results/sender_baseline_comparison_parallel}"
LOG_ROOT="${LOG_ROOT:-logs/sender_baseline_comparison_parallel}"

mkdir -p "$ROOT" "$LOG_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# argparse option is --scenarios and can be repeated.
COMMON_ARGS=(
  --num-trajectories 10
  --scenarios 0 0
  --scenarios 1 0
  --scenarios 1 1
  --error-threshold 0.10
  --periodic-interval 20
  --no-plots
)

echo "Starting baseline worker 0 (seeds 0 1 2)..."
"$PYTHON_BIN" "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --seeds 0 1 2 \
  --output-directory "$ROOT/worker_0" \
  > "$LOG_ROOT/worker_0.log" 2>&1 &
PID0=$!

echo "Starting baseline worker 1 (seeds 3 4)..."
"$PYTHON_BIN" "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --seeds 3 4 \
  --output-directory "$ROOT/worker_1" \
  > "$LOG_ROOT/worker_1.log" 2>&1 &
PID1=$!

echo "worker_0 PID: $PID0"
echo "worker_1 PID: $PID1"
echo "Monitor: watch -n 3 'tail -n 3 $LOG_ROOT/worker_*.log'"

status=0
wait "$PID0" || status=$?
wait "$PID1" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "At least one worker failed. Inspect logs." >&2
  exit "$status"
fi

"$PYTHON_BIN" - "$ROOT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
files = sorted(root.glob("worker_*/raw_runs.csv"))
if not files:
    raise RuntimeError("No worker raw_runs.csv files found.")

data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
keys = [
    "trajectory_index",
    "seed",
    "process_noise_scale",
    "measurement_noise_scale",
    "method",
]
data = (
    data.drop_duplicates(subset=keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
)
merged = root / "merged"
merged.mkdir(parents=True, exist_ok=True)
data.to_csv(merged / "raw_runs.csv", index=False)
print("Merged rows:", len(data), "(expected 600)")
PY

# Re-run in resume mode to aggregate and plot only.
"$PYTHON_BIN" "$SCRIPT" \
  --num-trajectories 10 \
  --seeds 0 1 2 3 4 \
  --scenarios 0 0 \
  --scenarios 1 0 \
  --scenarios 1 1 \
  --error-threshold 0.10 \
  --periodic-interval 20 \
  --output-directory "$ROOT/merged"

echo "Baseline comparison complete: $ROOT/merged"
