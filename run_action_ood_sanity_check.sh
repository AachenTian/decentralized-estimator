#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-datasets/accel_diff_drive_noise005.npz}"
MODEL_SEEDS_TEXT="${MODEL_SEEDS:-0 1 2}"
GRADIENT_STEPS="${GRADIENT_STEPS:-5000}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-10}"
REGION_TYPE="${REGION_TYPE:-linear_upper_saturation}"
STATE_QUANTILE="${STATE_QUANTILE:-0.70}"
ACTION_QUANTILE="${ACTION_QUANTILE:-0.70}"
FORCE="${FORCE:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/action_ood_sanity}"
RESULT_ROOT="${RESULT_ROOT:-results/action_ood_sanity_check}"
TRAINING_STEPS_ROOT="${TRAINING_STEPS_ROOT:-checkpoints/training_steps_sanity}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-checkpoints/structured_diff_drive_dynamics_noise005.pkl}"
LOG_ROOT="$RESULT_ROOT/training_logs"
METRICS_ROOT="$RESULT_ROOT/training_metrics"
mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT" "$METRICS_ROOT"

read -r -a MODEL_SEEDS <<< "$MODEL_SEEDS_TEXT"
checkpoint_args=()

for seed in "${MODEL_SEEDS[@]}"; do
  seed_root="$CHECKPOINT_ROOT/seed_${seed}"
  mkdir -p "$seed_root"

  full_checkpoint="$TRAINING_STEPS_ROOT/seed_${seed}/steps_${GRADIENT_STEPS}.pkl"
  if [[ ! -e "$full_checkpoint" && "$seed" == "0" && -f "$REFERENCE_CHECKPOINT" ]]; then
    full_checkpoint="$REFERENCE_CHECKPOINT"
  fi
  if [[ ! -e "$full_checkpoint" ]]; then
    full_checkpoint="$seed_root/full_coverage.pkl"
    if [[ ! -e "$full_checkpoint" || "$FORCE" == "1" ]]; then
      echo "Training full coverage model, seed=$seed"
      "$PYTHON_BIN" -u train_structured_diff_drive_dynamics_action_ood.py \
        --dataset "$DATASET" \
        --training-mode full_coverage \
        --region-type "$REGION_TYPE" \
        --state-quantile "$STATE_QUANTILE" \
        --action-quantile "$ACTION_QUANTILE" \
        --seed "$seed" \
        --gradient-steps "$GRADIENT_STEPS" \
        --eval-interval "$GRADIENT_STEPS" \
        --checkpoint-path "$full_checkpoint" \
        --metrics-json "$METRICS_ROOT/full_seed${seed}.json" \
        > "$LOG_ROOT/full_seed${seed}.log" 2>&1
    fi
  else
    echo "Reusing full coverage checkpoint: $full_checkpoint"
  fi

  random_checkpoint="$seed_root/random_matched.pkl"
  if [[ ! -e "$random_checkpoint" || "$FORCE" == "1" ]]; then
    rm -f "$random_checkpoint"
    echo "Training random-matched model, seed=$seed"
    "$PYTHON_BIN" -u train_structured_diff_drive_dynamics_action_ood.py \
      --dataset "$DATASET" \
      --training-mode random_matched \
      --region-type "$REGION_TYPE" \
      --state-quantile "$STATE_QUANTILE" \
      --action-quantile "$ACTION_QUANTILE" \
      --random-match-seed 90210 \
      --seed "$seed" \
      --gradient-steps "$GRADIENT_STEPS" \
      --eval-interval "$GRADIENT_STEPS" \
      --checkpoint-path "$random_checkpoint" \
      --metrics-json "$METRICS_ROOT/random_seed${seed}.json" \
      > "$LOG_ROOT/random_seed${seed}.log" 2>&1
  else
    echo "Skipping existing checkpoint: $random_checkpoint"
  fi

  heldout_checkpoint="$seed_root/heldout_region.pkl"
  if [[ ! -e "$heldout_checkpoint" || "$FORCE" == "1" ]]; then
    rm -f "$heldout_checkpoint"
    echo "Training held-out action-region model, seed=$seed"
    "$PYTHON_BIN" -u train_structured_diff_drive_dynamics_action_ood.py \
      --dataset "$DATASET" \
      --training-mode heldout_region \
      --region-type "$REGION_TYPE" \
      --state-quantile "$STATE_QUANTILE" \
      --action-quantile "$ACTION_QUANTILE" \
      --seed "$seed" \
      --gradient-steps "$GRADIENT_STEPS" \
      --eval-interval "$GRADIENT_STEPS" \
      --checkpoint-path "$heldout_checkpoint" \
      --metrics-json "$METRICS_ROOT/heldout_seed${seed}.json" \
      > "$LOG_ROOT/heldout_seed${seed}.log" 2>&1
  else
    echo "Skipping existing checkpoint: $heldout_checkpoint"
  fi

  checkpoint_args+=(--checkpoint full_coverage "$seed" "$full_checkpoint")
  checkpoint_args+=(--checkpoint random_matched "$seed" "$random_checkpoint")
  checkpoint_args+=(--checkpoint heldout_region "$seed" "$heldout_checkpoint")
done

echo "Evaluating controlled action-region OOD experiment..."
"$PYTHON_BIN" run_action_ood_sanity_check.py \
  --dataset "$DATASET" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --error-threshold 0.10 \
  --output-directory "$RESULT_ROOT/evaluation" \
  "${checkpoint_args[@]}"

echo "Done."
echo "Results: $RESULT_ROOT/evaluation"
