# Phase 2 Status: Private Five-Member Dynamics

Phase two adds a complete one-step dynamics-training path while preserving all
phase-one invariants.

Implemented:

- separate public random-policy calibration for one common frozen actor
  observation normalizer;
- three owner-private dynamics normalizers;
- normalizers freeze after the first owner has enough nonterminal transitions;
- three independent probabilistic dynamics ensembles;
- five independently initialized members per owner, fifteen members total;
- input `[18D model state, 6D joint action]`;
- output `12D joint agent local-state delta`;
- independent bootstrap resampling for every ensemble member;
- Gaussian negative log-likelihood training;
- terminal/time-limit transitions excluded from dynamics fitting;
- one-step RMSE, uncertainty, coverage, collision-conditioned metrics;
- offline W&B logging under `owner/{i}/dynamics/...`;
- parameter-isolation and shape tests.

Not implemented yet:

- model rollout;
- model replay;
- collision/uncertainty synthetic termination;
- twin critics;
- focal SAC actor update;
- alpha update;
- target critics.

## Smoke-test budget

The default phase-two smoke script uses:

- one synchronization round;
- two environments per owner;
- 25 real steps per environment;
- 50 real transitions per owner;
- 48 nonterminal dynamics transitions per owner;
- 10 dynamics updates per owner;
- batch size 32 with bootstrap replacement;
- one separate public calibration rollout of 50 environment steps.

Expected private training interaction count:

```text
50 transitions/owner
150 transitions/system
```

The 50 public calibration steps are reported separately and are not inserted
into any owner's private replay.
