# Oracle decentralized SAC animation

The script creates a deterministic Simple-Spread animation for the current
independent oracle decentralized SAC policy.

## Install

```bash
cd ~/RL_projects/decentralized-estimator
mkdir -p src/sender_marl/visualization

cp animate_oracle_decentralized_sac.py \
  src/sender_marl/visualization/animate_oracle_decentralized_sac.py
```

## Run environment seed 30

Use the `best.pkl` from the current experiment:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m sender_marl.visualization.animate_oracle_decentralized_sac \
  --checkpoint \
    results/oracle_normalized_persistent16_budgetmatched_seed2/best.pkl \
  --normalization \
    results/oracle_normalized_persistent16_budgetmatched_seed2/normalization.json \
  --env-seed 30 \
  --max-steps 25 \
  --fps 5 \
  --output-dir \
    results/policy_animations/oracle_sac_training_seed2_env_seed30
```

Replace the result directory when the current checkpoint is stored elsewhere.

## Outputs

```text
oracle_sac_seed30.gif
oracle_sac_seed30.mp4
oracle_sac_seed30_trajectory.npz
oracle_sac_seed30_summary.json
```

The MP4 is generated when `ffmpeg` is installed.
