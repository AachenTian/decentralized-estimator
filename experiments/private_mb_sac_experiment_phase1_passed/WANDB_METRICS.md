# W&B Tracking Plan

The new W&B project is:

```text
private-mb-sac-snapshot-sync
```

The old `AORPO-dynamics model` project name is not reused.

## Logging topology

Use one W&B run for one full three-owner training seed. Metrics are separated by
namespace:

```text
owner/0/...
owner/1/...
owner/2/...
system/...
evaluation/...
diagnostics/...
```

All metrics use `round` as the explicit W&B step.

## Critical metrics

### Real environment interaction

Per owner:

```text
real/env_steps_round
real/env_steps_total
real/episodes_round
real/return_mean
real/collision_rate
real/replay_size
```

System:

```text
system/real_env_steps_round
system/real_env_steps_total
system/owner_env_steps_min
system/owner_env_steps_max
```

The system-wide real interaction counter must include all three private owners.

### Dynamics ensemble

```text
dynamics/nll
dynamics/state_rmse
dynamics/position_rmse
dynamics/velocity_rmse
dynamics/noncollision_velocity_rmse
dynamics/collision_velocity_rmse
dynamics/epistemic_mean
dynamics/aleatoric_mean
dynamics/coverage95
dynamics/nonterminal_fraction
dynamics/grad_norm
```

Collision and non-collision velocity RMSE are kept separate because earlier
diagnostics showed that sparse collisions dominate velocity prediction error.

### Model rollout

```text
model/generated_transitions
model/mean_horizon
model/max_horizon
model/termination_horizon_fraction
model/termination_done_fraction
model/termination_collision_fraction
model/termination_uncertainty_fraction
model/predicted_collision_rate
model/epistemic_mean
model/replay_size
```

Termination fractions must sum approximately to one over terminated synthetic
trajectories. Log raw counts locally as well.

### SAC learner

```text
critic/q1_loss
critic/q2_loss
critic/q1_mean
critic/q2_mean
critic/target_q_mean
critic/td_abs
critic/fixed_td_abs
critic/grad_norm

actor/loss
actor/policy_q
actor/entropy
actor/log_prob_mean
actor/action_saturation_rate
actor/grad_norm

alpha/value
alpha/loss
```

Also log cumulative update counters:

```text
critic/updates_total
actor/updates_total
```

### Evaluation

```text
evaluation/return_mean
evaluation/return_std
evaluation/return_median
evaluation/return_worst10_mean
evaluation/collision_rate_mean
evaluation/completion_rate
evaluation/best_return_so_far
evaluation/best_round
```

Evaluation should use deterministic `tanh(mean)` actions and fixed evaluation
seed schedules that are separate from training.

### Timing and throughput

```text
timing/collection_seconds
timing/dynamics_seconds
timing/model_rollout_seconds
timing/sac_seconds

system/round_seconds
system/real_steps_per_second
system/synthetic_steps_per_second
```

## Histograms

Histograms are useful but should be logged less frequently:

```text
diagnostics/actor_action_hist_owner_0
diagnostics/actor_action_hist_owner_1
diagnostics/actor_action_hist_owner_2
diagnostics/dynamics_epistemic_hist_owner_0
diagnostics/dynamics_epistemic_hist_owner_1
diagnostics/dynamics_epistemic_hist_owner_2
```

Recommended frequency: every 10 or 20 synchronization rounds.

## Artifacts

Upload these as W&B artifacts:

- `best.pkl`;
- `latest.pkl`;
- frozen actor-observation normalization JSON;
- final resolved YAML configuration;
- selected evaluation animations;
- final metrics JSONL file.

Do not upload replay buffers by default because they are large and private to
each owner.
