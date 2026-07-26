#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-datasets/accel_diff_drive_noise005.npz}"
GRADIENT_STEPS="${GRADIENT_STEPS:-5000}"
TRAINING_SEED="${TRAINING_SEED:-0}"
SUBSET_SEED="${SUBSET_SEED:-2026}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-10}"
FORCE="${FORCE:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/model_quality_sanity}"
RESULT_ROOT="${RESULT_ROOT:-results/model_quality_sanity_check}"
METRICS_ROOT="$RESULT_ROOT/training_metrics"
LOG_ROOT="$RESULT_ROOT/training_logs"
mkdir -p "$CHECKPOINT_ROOT" "$METRICS_ROOT" "$LOG_ROOT"

fractions=(1.0 0.5 0.25 0.10)
labels=(100 50 25 10)

checkpoint_args=()
for index in "${!fractions[@]}"; do
  fraction="${fractions[$index]}"
  label="${labels[$index]}"
  checkpoint="$CHECKPOINT_ROOT/model_${label}.pkl"
  metrics="$METRICS_ROOT/model_${label}.json"
  log="$LOG_ROOT/model_${label}.log"

  if [[ -f "$checkpoint" && "$FORCE" != "1" ]]; then
    echo "Skipping existing checkpoint: $checkpoint"
  else
    echo "Training model_${label} with fraction=$fraction"
    "$PYTHON_BIN" -u train_structured_diff_drive_dynamics_variant.py \
      --dataset "$DATASET" \
      --train-data-fraction "$fraction" \
      --train-subset-seed "$SUBSET_SEED" \
      --seed "$TRAINING_SEED" \
      --gradient-steps "$GRADIENT_STEPS" \
      --prediction-modes ensemble_mean \
      --checkpoint-path "$checkpoint" \
      --metrics-json "$metrics" \
      --save-checkpoint \
      > "$log" 2>&1
  fi

  checkpoint_args+=(--checkpoint "model_${label}" "$fraction" "$checkpoint")
done

echo "Evaluating all model-quality variants..."
"$PYTHON_BIN" run_model_quality_sanity_check.py \
  --dataset "$DATASET" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --error-threshold 0.10 \
  --output-directory "$RESULT_ROOT/evaluation" \
  "${checkpoint_args[@]}"

echo "Done."
echo "Results: $RESULT_ROOT/evaluation"
