#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-datasets/accel_diff_drive_noise005.npz}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-10}"
MODEL_SEEDS_TEXT="${MODEL_SEEDS:-0 1 2}"
TRAIN_STEPS_TEXT="${TRAIN_STEPS:-100 500 1000 5000}"
FORCE="${FORCE:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/training_steps_sanity}"
RESULT_ROOT="${RESULT_ROOT:-results/training_steps_sanity_check}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-checkpoints/structured_diff_drive_dynamics_noise005.pkl}"
REUSE_REFERENCE_CHECKPOINT="${REUSE_REFERENCE_CHECKPOINT:-1}"
LOG_ROOT="$RESULT_ROOT/training_logs"
METRICS_ROOT="$RESULT_ROOT/training_metrics"
mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT" "$METRICS_ROOT"

read -r -a MODEL_SEEDS <<< "$MODEL_SEEDS_TEXT"
read -r -a TRAIN_STEPS <<< "$TRAIN_STEPS_TEXT"

checkpoint_args=()

for seed in "${MODEL_SEEDS[@]}"; do
  mkdir -p "$CHECKPOINT_ROOT/seed_${seed}"
  for steps in "${TRAIN_STEPS[@]}"; do
    label="steps_${steps}"
    checkpoint="$CHECKPOINT_ROOT/seed_${seed}/${label}.pkl"
    log="$LOG_ROOT/${label}_seed${seed}.log"
    metrics="$METRICS_ROOT/${label}_seed${seed}.json"

    if [[ "$steps" == "5000" && "$seed" == "0" \
          && "$REUSE_REFERENCE_CHECKPOINT" == "1" \
          && -f "$REFERENCE_CHECKPOINT" \
          && ! -e "$checkpoint" ]]; then
      echo "Reusing reference checkpoint for steps=5000 seed=0"
      ln -s "$(realpath --relative-to="$(dirname "$checkpoint")" "$REFERENCE_CHECKPOINT")" "$checkpoint"
    fi

    if [[ -e "$checkpoint" && "$FORCE" != "1" ]]; then
      echo "Skipping existing checkpoint: $checkpoint"
    else
      rm -f "$checkpoint"
      echo "Training gradient_steps=$steps, model_seed=$seed"
      "$PYTHON_BIN" -u train_structured_diff_drive_dynamics_variant.py \
        --dataset "$DATASET" \
        --train-data-fraction 1.0 \
        --seed "$seed" \
        --gradient-steps "$steps" \
        --eval-interval "$steps" \
        --prediction-modes ensemble_mean \
        --checkpoint-path "$checkpoint" \
        --metrics-json "$metrics" \
        --save-checkpoint \
        > "$log" 2>&1
    fi

    checkpoint_args+=(--checkpoint "$label" "$steps" "$seed" "$checkpoint")
  done
done

echo "Evaluating training-step variants..."
"$PYTHON_BIN" run_training_steps_sanity_check.py \
  --dataset "$DATASET" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --error-threshold 0.10 \
  --output-directory "$RESULT_ROOT/evaluation" \
  "${checkpoint_args[@]}"

echo "Done."
echo "Results: $RESULT_ROOT/evaluation"
