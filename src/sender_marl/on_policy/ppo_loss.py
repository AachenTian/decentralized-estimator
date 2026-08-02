from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from sender_marl.core.distributions import (
    diagonal_gaussian_entropy,
    squashed_gaussian_log_prob_from_pre_tanh,
)
from sender_marl.on_policy.training_types import PPOTrainingBatch


Array = jax.Array
ApplyFn = Callable[..., Any]


def masked_mean(value: Array, mask: Array, epsilon: float = 1e-8) -> Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), epsilon)


def ppo_actor_loss(
    actor_params: Any,
    actor_apply_fn: ApplyFn,
    batch: PPOTrainingBatch,
    *,
    clip_epsilon: float,
    entropy_coefficient: float,
) -> tuple[Array, Mapping[str, Array]]:
    """Clipped PPO policy loss for a tanh-squashed Gaussian actor."""

    mean, log_std = actor_apply_fn(
        {"params": actor_params},
        batch.actor_obs,
    )
    new_log_probs = squashed_gaussian_log_prob_from_pre_tanh(
        batch.pre_tanh_actions,
        mean,
        log_std,
    )
    log_ratio = new_log_probs - batch.old_log_probs
    ratio = jnp.exp(log_ratio)

    unclipped = ratio * batch.advantages
    clipped_ratio = jnp.clip(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    clipped = clipped_ratio * batch.advantages
    surrogate = jnp.minimum(unclipped, clipped)

    policy_loss = -masked_mean(surrogate, batch.alive)
    entropy = masked_mean(
        diagonal_gaussian_entropy(log_std),
        batch.alive,
    )
    total_loss = policy_loss - entropy_coefficient * entropy

    approx_kl = masked_mean(
        batch.old_log_probs - new_log_probs,
        batch.alive,
    )
    clip_fraction = masked_mean(
        (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32),
        batch.alive,
    )

    metrics = {
        "actor_loss": total_loss,
        "policy_loss": policy_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "mean_ratio": masked_mean(ratio, batch.alive),
    }
    return total_loss, metrics


def ppo_value_loss(
    value_params: Any,
    value_apply_fn: ApplyFn,
    batch: PPOTrainingBatch,
    *,
    clip_epsilon: float,
    value_loss_coefficient: float,
) -> tuple[Array, Mapping[str, Array]]:
    """PPO clipped value loss."""

    new_values = value_apply_fn(
        {"params": value_params},
        batch.critic_obs,
    )
    value_delta = new_values - batch.old_values
    clipped_values = batch.old_values + jnp.clip(
        value_delta,
        -clip_epsilon,
        clip_epsilon,
    )

    squared_error = jnp.square(new_values - batch.returns)
    clipped_squared_error = jnp.square(clipped_values - batch.returns)
    unclipped_loss = 0.5 * masked_mean(squared_error, batch.alive)
    clipped_loss = 0.5 * masked_mean(
        jnp.maximum(squared_error, clipped_squared_error),
        batch.alive,
    )
    total_loss = value_loss_coefficient * clipped_loss

    metrics = {
        "value_loss": total_loss,
        "unscaled_value_loss": clipped_loss,
        "unclipped_value_loss": unclipped_loss,
        "value_mean": masked_mean(new_values, batch.alive),
        "return_mean": masked_mean(batch.returns, batch.alive),
    }
    return total_loss, metrics
