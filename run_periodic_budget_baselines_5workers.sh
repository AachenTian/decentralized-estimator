#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-run_periodic_budget_baselines.py}"
ROOT="${OUTPUT_ROOT:-results/periodic_budget_baselines_5workers}"
LOG_ROOT="${LOG_ROOT:-logs/periodic_budget_baselines_5workers}"
FINAL_ROOT="${FINAL_ROOT:-results/fair_budget_method_comparison}"
ORIGINAL_RAW="${ORIGINAL_RAW:-results/sender_baseline_comparison_5workers/merged/raw_runs.csv}"

mkdir -p "$ROOT" "$LOG_ROOT"

# Prevent two copies of this launcher from writing the same CSV files.
exec 9>"$ROOT/.run.lock"
if ! flock -n 9; then
  echo "Another periodic-budget experiment is already running for $ROOT." >&2
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --num-trajectories 10
  --scenarios 0 0
  --scenarios 1 0
  --scenarios 1 1
  --periodic-intervals 50 100
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
echo "Each worker has 60 runs."
echo "Waiting for all workers..."

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one worker failed. Inspect logs under $LOG_ROOT." >&2
  exit "$status"
fi

echo "All workers finished. Merging periodic results..."

"$PYTHON_BIN" - "$ROOT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
files = sorted(root.glob("worker_*/raw_runs.csv"))
if len(files) != 5:
    raise RuntimeError(f"Expected 5 worker CSVs, found {len(files)}.")

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

print(f"Periodic unique rows: {len(data)} (expected 300)")
if len(data) != 300:
    raise RuntimeError(
        f"Unexpected periodic result count: {len(data)}; expected 300."
    )
PY

echo "Creating final six-method comparison..."

"$PYTHON_BIN" merge_and_plot_fair_budget_comparison.py \
  --original-raw "$ORIGINAL_RAW" \
  --periodic-raw "$ROOT/merged/raw_runs.csv" \
  --output-directory "$FINAL_ROOT"

echo "Done."
echo "Periodic results: $ROOT/merged"
echo "Final comparison: $FINAL_ROOT"
