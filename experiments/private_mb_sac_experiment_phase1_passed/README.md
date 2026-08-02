# Private Model-Based SAC with Actor Snapshot Synchronization

This folder is an isolated implementation skeleton for a three-agent,
fully private training pipeline:

- three independent live actors;
- three independent twin-critic SAC learners;
- three independent real replay buffers;
- three independent model replay buffers;
- three independent five-member probabilistic dynamics ensembles;
- actor snapshot exchange only at the beginning of each synchronization round;
- no actor averaging;
- no trajectory or replay sharing;
- no opponent model;
- no communication fallback.

The initial training schedule mirrors the previous AORPO-style experiment:

- 200 synchronization rounds;
- 300 real joint transitions per owner per round;
- 50 dynamics updates per owner per round;
- 100 SAC updates per owner per round;
- model rollout horizon scheduled from 1 to 6.

See `ARCHITECTURE.md` for the full design and invariants.

## Weights & Biases

The isolated project uses a new W&B project name:

```text
private-mb-sac-snapshot-sync
```

Tracking is configured in `configs/default.yaml`. The logger is implemented in
`private_mb_sac/tracking/wandb_logger.py`, and the full metric plan is in
`WANDB_METRICS.md`.

W&B is initialized only from the training entry point. Importing modules never
calls `wandb.login()` or starts a run.

## Phase-one implementation

This package now implements and tests:

- recursive validated YAML configuration;
- common environment RNG schedules and owner-private policy RNG schedules;
- independent actor initialization and immutable snapshot exchange;
- three strictly isolated private real replay buffers;
- vectorized 25-step Simple Spread real collection;
- correct terminal transition storage before reset;
- terminal filtering for future dynamics training;
- per-owner and system interaction accounting;
- offline W&B logging for the collection smoke test.

Run:

```bash
export PROJECT_ROOT=~/RL_projects/decentralized-estimator
export PYTHONPATH="$PWD:$PROJECT_ROOT/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pytest -q
bash scripts/smoke_test.sh
```

Expected smoke result:

```text
owner 0: env_steps=50, replay=50
owner 1: env_steps=50, replay=50
owner 2: env_steps=50, replay=50
system: env_steps=150
```

The actor observation normalizer is currently identity-only. A calibrated common
frozen normalizer is added before SAC training begins. Dynamics and SAC modules
remain intentionally unimplemented until this data path passes.

## Terminal-count invariant

For `--smoke-num-envs 2` and `rollout_length=max_steps=25`, the expected
terminal count is `[2, 2, 2]`. A zero count is not accepted because dynamics
training must be able to filter time-limit terminal transitions.
