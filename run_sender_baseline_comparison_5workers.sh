#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-run_sender_baseline_comparison.py}"
ROOT="${OUTPUT_ROOT:-results/sender_baseline_comparison_5workers}"
LOG_ROOT="${LOG_ROOT:-logs/sender_baseline_comparison_5workers}"

mkdir -p "$ROOT" "$LOG_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

# Important: --scenarios accepts exactly one pair per occurrence,
# so it must be repeated for the three scenarios.
COMMON_ARGS=(
  --num-trajectories 10
  --scenarios 0 0
  --scenarios 1 0
  --scenarios 1 1
  --error-threshold 0.10
  --periodic-interval 20
  --no-plots
)

pids=()
for seed in 0 1 2 3 4; do
  worker="worker_${seed}"
  echo "Starting ${worker} with seed ${seed}..."

  "$PYTHON_BIN" -u "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --seeds "$seed" \
    --output-directory "$ROOT/$worker" \
    > "$LOG_ROOT/$worker.log" 2>&1 &

  pids+=("$!")
done

echo "Worker PIDs: ${pids[*]}"
echo "Waiting for all workers..."

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one worker failed. Inspect logs under $LOG_ROOT." >&2
  exit "$status"
fi

echo "All workers finished. Merging CSV files..."

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

print(f"Merged rows: {len(data)} (expected 600)")
print(f"Output: {merged / 'raw_runs.csv'}")
PY

# Resume from the merged raw CSV. Existing rows are skipped; this invocation
# creates aggregate.csv and plots.
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
