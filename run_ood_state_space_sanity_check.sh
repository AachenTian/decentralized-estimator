#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-datasets/accel_diff_drive_noise005.npz}"
GRADIENT_STEPS="${GRADIENT_STEPS:-5000}"
TRAINING_SEED="${TRAINING_SEED:-0}"
SUBSET_SEED="${SUBSET_SEED:-2026}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-10}"
FORCE="${FORCE:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/ood_state_space_sanity}"
RESULT_ROOT="${RESULT_ROOT:-results/ood_state_space_sanity_check}"
LOG_ROOT="$RESULT_ROOT/training_logs"
mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT"

# High linear velocity is used rather than px/py because the structured model
# learns from [sin(phi), cos(phi), v, omega, action], while position is
# reconstructed analytically and is translationally invariant.
if [[ -z "${FULL_CHECKPOINT:-}" ]]; then
  if [[ -f "checkpoints/model_quality_sanity/model_100.pkl" ]]; then
    FULL_CHECKPOINT="checkpoints/model_quality_sanity/model_100.pkl"
  else
    FULL_CHECKPOINT="$CHECKPOINT_ROOT/full_coverage.pkl"
  fi
fi
HELDOUT_CHECKPOINT="${HELDOUT_CHECKPOINT:-$CHECKPOINT_ROOT/heldout_high_v_q80.pkl}"

if [[ ! -f "$FULL_CHECKPOINT" || "$FORCE" == "1" ]]; then
  echo "Training full-coverage model..."
  "$PYTHON_BIN" -u train_structured_diff_drive_dynamics_variant.py \
    --dataset "$DATASET" \
    --train-data-fraction 1.0 \
    --train-subset-seed "$SUBSET_SEED" \
    --seed "$TRAINING_SEED" \
    --gradient-steps "$GRADIENT_STEPS" \
    --prediction-modes ensemble_mean \
    --checkpoint-path "$FULL_CHECKPOINT" \
    --metrics-json "$RESULT_ROOT/full_coverage_training_metrics.json" \
    --save-checkpoint \
    > "$LOG_ROOT/full_coverage.log" 2>&1
else
  echo "Skipping existing checkpoint: $FULL_CHECKPOINT"
fi

if [[ ! -f "$HELDOUT_CHECKPOINT" || "$FORCE" == "1" ]]; then
  echo "Training model with high-v region held out..."
  "$PYTHON_BIN" -u train_structured_diff_drive_dynamics_variant.py \
    --dataset "$DATASET" \
    --train-data-fraction 1.0 \
    --train-subset-seed "$SUBSET_SEED" \
    --holdout-dimension v \
    --holdout-quantile 0.80 \
    --holdout-side upper \
    --seed "$TRAINING_SEED" \
    --gradient-steps "$GRADIENT_STEPS" \
    --prediction-modes ensemble_mean \
    --checkpoint-path "$HELDOUT_CHECKPOINT" \
    --metrics-json "$RESULT_ROOT/heldout_training_metrics.json" \
    --save-checkpoint \
    > "$LOG_ROOT/heldout_high_v.log" 2>&1
else
  echo "Skipping existing checkpoint: $HELDOUT_CHECKPOINT"
fi

echo "Evaluating held-out state-space behavior..."
"$PYTHON_BIN" run_ood_state_space_sanity_check.py \
  --dataset "$DATASET" \
  --full-checkpoint "$FULL_CHECKPOINT" \
  --heldout-checkpoint "$HELDOUT_CHECKPOINT" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --error-threshold 0.10 \
  --output-directory "$RESULT_ROOT/evaluation"

echo "Done."
echo "Results: $RESULT_ROOT/evaluation"
