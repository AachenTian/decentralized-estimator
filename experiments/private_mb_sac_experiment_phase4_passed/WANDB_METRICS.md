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

## Phase-two additions

Phase two logs:

```text
owner/{i}/dynamics/loss
owner/{i}/dynamics/nll
owner/{i}/dynamics/normalized_mse
owner/{i}/dynamics/state_rmse
owner/{i}/dynamics/position_rmse
owner/{i}/dynamics/velocity_rmse
owner/{i}/dynamics/noncollision_velocity_rmse
owner/{i}/dynamics/collision_velocity_rmse
owner/{i}/dynamics/epistemic_mean
owner/{i}/dynamics/aleatoric_mean
owner/{i}/dynamics/coverage95
owner/{i}/dynamics/grad_norm
owner/{i}/dynamics/mean_logvar
owner/{i}/dynamics/nonterminal_fraction
owner/{i}/dynamics/updates_total
owner/{i}/dynamics/normalizer_samples
```

Public actor-normalization calibration is logged separately:

```text
system/calibration_env_steps
system/calibration_samples
system/calibration_std_min
system/calibration_std_max
```

## Phase-four system diagnostics

```text
system/critic_updates_total
system/actor_updates_total
system/actor_snapshot_drift_mean
system/actor_snapshot_drift_min
system/actor_pairwise_parameter_distance_mean
```

Owner namespaces additionally log the private real/model batch fractions,
critic losses, fixed TD error, actor entropy/Q/loss, alpha, gradient norms, and
cumulative update counters.
