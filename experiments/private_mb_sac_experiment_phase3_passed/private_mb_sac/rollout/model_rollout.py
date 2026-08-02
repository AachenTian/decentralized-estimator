"""Private synthetic rollout with frozen actor snapshots and no communication."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from private_mb_sac.core.types import ActorSnapshotBank, ModelRolloutOutput, RealRolloutBatch, ReplayBatch
from private_mb_sac.dynamics.prediction import (
    apply_agent_delta, epistemic_score, predict_dynamics_distribution,
    sample_moment_matched_delta,
)
from private_mb_sac.dynamics.termination import (
    TERMINATION_HORIZON, step_termination_reason,
)
from private_mb_sac.envs.model_state import (
    build_actor_observations_from_model_state,
    collision_metrics_from_model_state,
    simple_spread_rewards_from_model_state,
)
from private_mb_sac.rollout.real_collector import normalize_actor_observations


def model_rollout_horizon(round_index: int, *, horizon_min: int, horizon_max: int,
                          schedule_start_round: int, schedule_end_round: int) -> int:
    if horizon_min < 1 or horizon_max < horizon_min:
        raise ValueError('Invalid model rollout horizon bounds.')
    if round_index < schedule_start_round:
        return int(horizon_min)
    if round_index > schedule_end_round:
        return int(horizon_max)
    span = max(schedule_end_round - schedule_start_round, 1)
    fraction = (round_index - schedule_start_round) / span
    return int(round(horizon_min + (horizon_max-horizon_min)*fraction))


def make_private_model_rollout(actor_apply_fn, *, num_agents: int,
                               num_landmarks: int, ensemble_size: int,
                               horizon: int, max_steps: int,
                               local_ratio: float, collision_distance: float,
                               uncertainty_metric: str,
                               terminate_on_done: bool,
                               terminate_on_collision: bool,
                               terminate_on_uncertainty: bool,
                               stochastic_policy_actions: bool,
                               stochastic_dynamics: bool,
                               jit: bool=True):
    if horizon < 1:
        raise ValueError('horizon must be positive.')

    def rollout(snapshot_bank: ActorSnapshotBank, actor_normalizer,
                dynamics_train_state, dynamics_normalizer,
                initial_batch: ReplayBatch, key, uncertainty_threshold):
        states = jnp.asarray(initial_batch.model_states, dtype=jnp.float32)
        observations = jnp.asarray(initial_batch.observations, dtype=jnp.float32)
        if initial_batch.episode_steps is None:
            steps = jnp.zeros((states.shape[0],), dtype=jnp.int32)
        else:
            steps = jnp.asarray(initial_batch.episode_steps, dtype=jnp.int32)
        batch_size = states.shape[0]
        active = jnp.ones((batch_size,), dtype=jnp.bool_)
        lengths = jnp.zeros((batch_size,), dtype=jnp.int32)
        reasons = jnp.zeros((batch_size,), dtype=jnp.int32)
        initial_landmarks = states[:, num_agents*4:]

        def one_step(carry, _):
            state, raw_obs, episode_step, active, lengths, reasons, key = carry
            key, action_key, dynamics_key = jax.random.split(key, 3)
            actor_obs = normalize_actor_observations(raw_obs, actor_normalizer)
            means, log_stds = actor_apply_fn(
                {'params': snapshot_bank.params_by_agent}, actor_obs
            )
            if stochastic_policy_actions:
                noise = jax.random.normal(action_key, means.shape)
                actions = jnp.tanh(means + jnp.exp(log_stds)*noise)
            else:
                actions = jnp.tanh(means)

            prediction = predict_dynamics_distribution(
                dynamics_train_state, dynamics_normalizer, state, actions,
                ensemble_size=ensemble_size,
            )
            delta = sample_moment_matched_delta(
                dynamics_key, prediction, stochastic=stochastic_dynamics
            )
            predicted_next = apply_agent_delta(state, delta)
            next_obs = build_actor_observations_from_model_state(
                predicted_next, num_agents=num_agents, num_landmarks=num_landmarks
            )
            rewards = simple_spread_rewards_from_model_state(
                predicted_next, num_agents=num_agents, num_landmarks=num_landmarks,
                local_ratio=local_ratio, collision_distance=collision_distance,
            )
            collision_metrics = collision_metrics_from_model_state(
                predicted_next, num_agents=num_agents, num_landmarks=num_landmarks,
                collision_distance=collision_distance,
            )
            next_step = episode_step + 1
            actual_done = next_step >= max_steps
            score = epistemic_score(prediction, uncertainty_metric)
            uncertainty = score > uncertainty_threshold
            collision = collision_metrics['any_collision']
            stop_done = actual_done & bool(terminate_on_done)
            stop_collision = collision & bool(terminate_on_collision)
            stop_uncertainty = uncertainty & bool(terminate_on_uncertainty)
            stop = stop_done | stop_collision | stop_uncertainty
            step_reason = step_termination_reason(stop_done, stop_collision, stop_uncertainty)
            newly_stopped = active & stop
            next_reasons = jnp.where(newly_stopped, step_reason, reasons)
            valid = active
            next_active = active & ~stop
            next_lengths = lengths + valid.astype(jnp.int32)

            dones = jnp.broadcast_to(actual_done[:, None], (batch_size, num_agents)).astype(jnp.float32)
            landmark_drift = jnp.max(jnp.abs(predicted_next[:, num_agents*4:] - initial_landmarks), axis=-1)
            transition = (
                RealRolloutBatch(
                    observations=raw_obs,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_obs,
                    dones=dones,
                    model_states=state,
                    next_model_states=predicted_next,
                    episode_dones=actual_done,
                    pair_collision_rates=collision_metrics['pair_collision_rate'],
                    min_pair_distances=collision_metrics['min_pair_distance'],
                    episode_steps=episode_step,
                    next_episode_steps=next_step,
                ),
                valid,
                score,
                landmark_drift,
            )
            return (predicted_next, next_obs, next_step, next_active,
                    next_lengths, next_reasons, key), transition

        final, outputs = jax.lax.scan(
            one_step,
            (states, observations, steps, active, lengths, reasons, key),
            xs=None, length=horizon,
        )
        _, _, _, final_active, final_lengths, final_reasons, _ = final
        final_reasons = jnp.where(final_active, TERMINATION_HORIZON, final_reasons)
        batches, valid_mask, scores, drift = outputs
        return ModelRolloutOutput(
            batch=batches,
            valid_mask=valid_mask,
            trajectory_lengths=final_lengths,
            termination_reasons=final_reasons,
            epistemic_scores=scores,
            landmark_drift=drift,
        )

    return jax.jit(rollout) if jit else rollout
