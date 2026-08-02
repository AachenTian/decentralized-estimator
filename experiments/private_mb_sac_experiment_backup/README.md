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
