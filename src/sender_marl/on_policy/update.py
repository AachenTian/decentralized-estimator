from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from sender_marl.on_policy.gae import compute_gae
from sender_marl.on_policy.ppo_loss import (
    masked_mean,
    ppo_actor_loss,
    ppo_value_loss,
)
from sender_marl.on_policy.train_state import PPOConfig
from sender_marl.on_policy.training_types import (
    PPOTrainingBatch,
    PPOUpdateOutput,
)
from sender_marl.on_policy.types import RolloutOutput


Array = jax.Array


def _flatten_samples(array: Array) -> Array:
    """Flatten leading (T, E, N) axes while preserving feature axes."""

    array = jnp.asarray(array)
    if array.ndim < 3:
        raise ValueError(
            "PPO sample arrays must begin with (T, E, N); got "
            f"shape={array.shape}."
        )
    return array.reshape((-1,) + array.shape[3:])


def _normalize_advantages(advantages: Array, alive: Array) -> Array:
    mean = masked_mean(advantages, alive)
    variance = masked_mean(jnp.square(advantages - mean), alive)
    normalized = (advantages - mean) / jnp.sqrt(variance + 1e-8)
    return jnp.where(alive > 0.0, normalized, 0.0)


def prepare_ppo_training_batch(
    rollout: RolloutOutput,
    config: PPOConfig,
) -> PPOTrainingBatch:
    """Convert a time-major rollout into flat PPO samples."""

    targets = compute_gae(
        rollout.batch.rewards,
        rollout.batch.values,
        rollout.batch.dones,
        rollout.final_values,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    alive = jnp.asarray(rollout.batch.alive, dtype=jnp.float32)
    advantages = targets.advantages
    if config.normalize_advantages:
        advantages = _normalize_advantages(advantages, alive)

    return PPOTrainingBatch(
        actor_obs=_flatten_samples(rollout.batch.actor_obs),
        critic_obs=_flatten_samples(rollout.batch.critic_obs),
        pre_tanh_actions=_flatten_samples(
            rollout.batch.pre_tanh_actions
        ),
        old_log_probs=_flatten_samples(rollout.batch.log_probs),
        old_values=_flatten_samples(rollout.batch.values),
        advantages=_flatten_samples(advantages),
        returns=_flatten_samples(targets.returns),
        alive=_flatten_samples(alive),
    )


def explained_variance(predictions: Array, targets: Array, mask: Array) -> Array:
    target_mean = masked_mean(targets, mask)
    target_variance = masked_mean(jnp.square(targets - target_mean), mask)
    residuals = targets - predictions
    residual_mean = masked_mean(residuals, mask)
    residual_variance = masked_mean(
        jnp.square(residuals - residual_mean),
        mask,
    )
    return jnp.where(
        target_variance > 1e-8,
        1.0 - residual_variance / target_variance,
        0.0,
    )


def make_ppo_update(config: PPOConfig, *, jit: bool = True):
    """Build a JIT-compatible shared-parameter IPPO/PPO update."""

    config.validate()

    def update(
        key: Array,
        actor_state: Any,
        value_state: Any,
        rollout: RolloutOutput,
    ) -> PPOUpdateOutput:
        training_batch = prepare_ppo_training_batch(rollout, config)
        num_samples = training_batch.actor_obs.shape[0]
        if num_samples % config.num_minibatches != 0:
            raise ValueError(
                "The flattened rollout size must be divisible by "
                f"num_minibatches; got {num_samples} samples and "
                f"{config.num_minibatches} minibatches."
            )
        minibatch_size = num_samples // config.num_minibatches

        def run_epoch(carry, _):
            actor_state, value_state, epoch_key = carry
            next_epoch_key, permutation_key = jax.random.split(epoch_key)
            permutation = jax.random.permutation(
                permutation_key,
                num_samples,
            )
            minibatch_indices = permutation.reshape(
                config.num_minibatches,
                minibatch_size,
            )

            def run_minibatch(states, indices):
                actor_state, value_state = states
                minibatch = jax.tree_util.tree_map(
                    lambda value: value[indices],
                    training_batch,
                )

                def actor_objective(params):
                    return ppo_actor_loss(
                        params,
                        actor_state.apply_fn,
                        minibatch,
                        clip_epsilon=config.clip_epsilon,
                        entropy_coefficient=(
                            config.entropy_coefficient
                        ),
                    )

                (actor_loss, actor_metrics), actor_grads = (
                    jax.value_and_grad(actor_objective, has_aux=True)(
                        actor_state.params
                    )
                )
                actor_gradient_norm = optax.global_norm(actor_grads)
                actor_state = actor_state.apply_gradients(
                    grads=actor_grads
                )

                def value_objective(params):
                    return ppo_value_loss(
                        params,
                        value_state.apply_fn,
                        minibatch,
                        clip_epsilon=config.clip_epsilon,
                        value_loss_coefficient=(
                            config.value_loss_coefficient
                        ),
                    )

                (value_loss, value_metrics), value_grads = (
                    jax.value_and_grad(value_objective, has_aux=True)(
                        value_state.params
                    )
                )
                value_gradient_norm = optax.global_norm(value_grads)
                value_state = value_state.apply_gradients(
                    grads=value_grads
                )

                metrics = {
                    **actor_metrics,
                    **value_metrics,
                    "actor_gradient_norm": actor_gradient_norm,
                    "value_gradient_norm": value_gradient_norm,
                    "combined_loss": actor_loss + value_loss,
                }
                return (actor_state, value_state), metrics

            (actor_state, value_state), minibatch_metrics = jax.lax.scan(
                run_minibatch,
                (actor_state, value_state),
                minibatch_indices,
            )
            epoch_metrics = jax.tree_util.tree_map(
                lambda value: jnp.mean(value, axis=0),
                minibatch_metrics,
            )
            return (
                actor_state,
                value_state,
                next_epoch_key,
            ), epoch_metrics

        key, epoch_key = jax.random.split(key)
        (actor_state, value_state, _), epoch_metrics = jax.lax.scan(
            run_epoch,
            (actor_state, value_state, epoch_key),
            xs=None,
            length=config.update_epochs,
        )
        metrics = jax.tree_util.tree_map(
            lambda value: jnp.mean(value, axis=0),
            epoch_metrics,
        )

        updated_values = value_state.apply_fn(
            {"params": value_state.params},
            training_batch.critic_obs,
        )
        metrics = {
            **metrics,
            "explained_variance": explained_variance(
                updated_values,
                training_batch.returns,
                training_batch.alive,
            ),
            "advantage_mean": masked_mean(
                training_batch.advantages,
                training_batch.alive,
            ),
            "advantage_std": jnp.sqrt(
                masked_mean(
                    jnp.square(
                        training_batch.advantages
                        - masked_mean(
                            training_batch.advantages,
                            training_batch.alive,
                        )
                    ),
                    training_batch.alive,
                )
                + 1e-8
            ),
        }
        return PPOUpdateOutput(
            actor_state=actor_state,
            value_state=value_state,
            metrics=metrics,
            key=key,
        )

    return jax.jit(update) if jit else update
