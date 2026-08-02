"""Owner-private model rollout generation, validation, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import jax
import numpy as np

from private_mb_sac.dynamics.prediction import epistemic_score, predict_dynamics_distribution
from private_mb_sac.dynamics.termination import (
    TERMINATION_COLLISION, TERMINATION_DONE, TERMINATION_HORIZON,
    TERMINATION_UNCERTAINTY,
)
from private_mb_sac.envs.model_state import simple_spread_rewards_from_model_state
from private_mb_sac.replay.buffer import create_private_replays
from private_mb_sac.rollout.model_rollout import make_private_model_rollout


@dataclass
class OwnerModelRuntime:
    owner_id: int
    model_replay: Any
    generated_total: int = 0


def initialize_owner_model_runtimes(config):
    replays = create_private_replays(
        num_owners=int(config.experiment.num_agents),
        capacity=int(config.replay.model_capacity),
        num_agents=int(config.experiment.num_agents),
        observation_dim=int(config.actor.observation_dim),
        action_dim=int(config.actor.action_dim),
        model_state_dim=int(config.critic.state_dim),
    )
    return [OwnerModelRuntime(i, replay) for i, replay in enumerate(replays)]


def validate_real_reward_reconstruction(replay, config, *, batch_size: int,
                                        seed: int) -> dict[str, float]:
    size = min(int(batch_size), len(replay))
    if size < 1:
        raise ValueError('Real replay is empty.')
    batch = replay.sample(size, np.random.default_rng(seed), replace=False)
    predicted = simple_spread_rewards_from_model_state(
        batch.next_model_states,
        num_agents=int(config.experiment.num_agents),
        num_landmarks=int(config.experiment.num_landmarks),
        local_ratio=0.5,
        collision_distance=0.30,
    )
    errors = np.asarray(predicted) - np.asarray(batch.rewards)
    return {
        'model/reward_reconstruction_mae': float(np.mean(np.abs(errors))),
        'model/reward_reconstruction_max_abs': float(np.max(np.abs(errors))),
    }


def calibrate_uncertainty_threshold(dynamics_runtime, replay, config, *, seed: int):
    valid_count = len(replay.valid_dynamics_indices)
    minimum = int(config.model_rollout.uncertainty_min_samples)
    if valid_count < minimum:
        return float('inf'), {'model/uncertainty_threshold_samples': float(valid_count)}
    size = min(int(config.model_rollout.batch_size), valid_count)
    batch = replay.sample(size, np.random.default_rng(seed), nonterminal_only=True, replace=False)
    prediction = predict_dynamics_distribution(
        dynamics_runtime.train_state, dynamics_runtime.normalizer,
        batch.model_states, batch.actions,
        ensemble_size=int(config.dynamics.ensemble_size),
    )
    scores = np.asarray(epistemic_score(prediction, str(config.model_rollout.uncertainty_metric)))
    threshold = float(np.quantile(scores, float(config.model_rollout.uncertainty_quantile)))
    threshold = threshold * float(config.model_rollout.uncertainty_multiplier) + float(config.model_rollout.uncertainty_epsilon)
    return threshold, {
        'model/uncertainty_threshold': threshold,
        'model/uncertainty_threshold_samples': float(size),
        'model/real_epistemic_mean': float(scores.mean()),
        'model/real_epistemic_max': float(scores.max()),
    }


def generate_private_owner_model_rollout(*, runtime: OwnerModelRuntime,
                                         dynamics_runtime, real_replay,
                                         snapshot_bank, actor_apply_fn,
                                         actor_normalizer, config,
                                         round_index: int, batch_size: int,
                                         horizon: int, seed: int,
                                         jit: bool=True):
    if dynamics_runtime.normalizer is None:
        raise RuntimeError('Dynamics normalizer must be fitted before model rollout.')
    rng = np.random.default_rng(seed)
    initial = real_replay.sample(batch_size, rng, nonterminal_only=True, replace=True)
    threshold, threshold_metrics = calibrate_uncertainty_threshold(
        dynamics_runtime, real_replay, config, seed=seed+1
    )
    rollout_fn = make_private_model_rollout(
        actor_apply_fn,
        num_agents=int(config.experiment.num_agents),
        num_landmarks=int(config.experiment.num_landmarks),
        ensemble_size=int(config.dynamics.ensemble_size),
        horizon=int(horizon),
        max_steps=int(config.experiment.max_steps),
        local_ratio=0.5,
        collision_distance=0.30,
        uncertainty_metric=str(config.model_rollout.uncertainty_metric),
        terminate_on_done=bool(config.model_rollout.terminate_on_done),
        terminate_on_collision=bool(config.model_rollout.terminate_on_collision),
        terminate_on_uncertainty=bool(config.model_rollout.terminate_on_uncertainty),
        stochastic_policy_actions=bool(config.model_rollout.stochastic_policy_actions),
        stochastic_dynamics=bool(config.model_rollout.stochastic_dynamics),
        jit=jit,
    )
    start = perf_counter()
    output = rollout_fn(
        snapshot_bank, actor_normalizer, dynamics_runtime.train_state,
        dynamics_runtime.normalizer, initial, jax.random.PRNGKey(seed+2),
        np.float32(threshold),
    )
    jax.block_until_ready(output.valid_mask)
    seconds = perf_counter()-start
    generated = runtime.model_replay.add_rollout(output.batch, output.valid_mask)
    runtime.generated_total += generated

    valid = np.asarray(output.valid_mask, dtype=np.bool_)
    scores = np.asarray(output.epistemic_scores)
    reasons = np.asarray(output.termination_reasons)
    lengths = np.asarray(output.trajectory_lengths)
    drift = np.asarray(output.landmark_drift)
    valid_scores = scores[valid]
    def frac(code): return float(np.mean(reasons == code))
    metrics = {
        'model/generated_transitions': float(generated),
        'model/generated_transitions_total': float(runtime.generated_total),
        'model/replay_size': float(len(runtime.model_replay)),
        'model/mean_horizon': float(lengths.mean()),
        'model/max_horizon': float(lengths.max()),
        'model/termination_horizon_fraction': frac(TERMINATION_HORIZON),
        'model/termination_done_fraction': frac(TERMINATION_DONE),
        'model/termination_collision_fraction': frac(TERMINATION_COLLISION),
        'model/termination_uncertainty_fraction': frac(TERMINATION_UNCERTAINTY),
        'model/predicted_collision_rate': float(np.asarray(output.batch.pair_collision_rates)[valid].mean()) if generated else 0.0,
        'model/epistemic_mean': float(valid_scores.mean()) if generated else 0.0,
        'model/landmark_drift_max': float(drift[valid].max()) if generated else 0.0,
        'timing/model_rollout_seconds': float(seconds),
        **threshold_metrics,
    }
    return metrics, output
