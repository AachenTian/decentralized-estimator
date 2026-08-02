"""Bootstrap Gaussian NLL updates and one-step dynamics diagnostics."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from private_mb_sac.core.types import DynamicsNormalizer, ReplayBatch
from private_mb_sac.dynamics.ensemble import tile_for_ensemble
from private_mb_sac.dynamics.normalization import (
    agent_state_delta,
    denormalize_delta,
    normalize_delta,
    normalize_dynamics_inputs,
)


def diagonal_gaussian_nll(
    mean: jax.Array,
    logvar: jax.Array,
    target: jax.Array,
) -> jax.Array:
    inverse_variance = jnp.exp(-logvar)
    per_dimension = 0.5 * (
        jnp.square(target - mean) * inverse_variance
        + logvar
        + jnp.log(2.0 * jnp.pi)
    )
    return jnp.mean(jnp.sum(per_dimension, axis=-1))


def _prepare_batch(
    batch: ReplayBatch,
    normalizer: DynamicsNormalizer,
) -> tuple[jax.Array, jax.Array]:
    inputs = normalize_dynamics_inputs(
        batch.model_states,
        batch.actions,
        normalizer,
    )
    target_delta = agent_state_delta(
        batch.model_states,
        batch.next_model_states,
    )
    targets = normalize_delta(target_delta, normalizer)
    return inputs, targets


@jax.jit
def dynamics_train_step(
    train_state: Any,
    normalizer: DynamicsNormalizer,
    batch: ReplayBatch,
    bootstrap_indices: jax.Array,
):
    inputs, targets = _prepare_batch(batch, normalizer)
    member_inputs = inputs[bootstrap_indices]
    member_targets = targets[bootstrap_indices]

    def loss_function(params):
        means, logvars = train_state.apply_fn(
            {"params": params},
            member_inputs,
        )
        nll = diagonal_gaussian_nll(means, logvars, member_targets)
        normalized_mse = jnp.mean(jnp.square(means - member_targets))
        return nll, {
            "nll": nll,
            "normalized_mse": normalized_mse,
            "mean_logvar": jnp.mean(logvars),
        }

    (loss, metrics), gradients = jax.value_and_grad(
        loss_function,
        has_aux=True,
    )(train_state.params)
    gradient_norm = optax.global_norm(gradients)
    next_state = train_state.apply_gradients(grads=gradients)
    metrics = {
        **metrics,
        "loss": loss,
        "grad_norm": gradient_norm,
    }
    return next_state, metrics


def _masked_rmse(
    errors: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    count = jnp.sum(mask)
    squared = jnp.square(errors)
    selected_sum = jnp.sum(jnp.where(mask[:, None], squared, 0.0))
    denominator = jnp.maximum(count * errors.shape[-1], 1)
    value = jnp.sqrt(selected_sum / denominator)
    return jnp.where(count > 0, value, jnp.nan)


def evaluate_dynamics_batch(
    train_state: Any,
    normalizer: DynamicsNormalizer,
    batch: ReplayBatch,
    *,
    ensemble_size: int,
) -> dict[str, jax.Array]:
    inputs, normalized_targets = _prepare_batch(batch, normalizer)
    tiled_inputs = tile_for_ensemble(inputs, ensemble_size)
    normalized_means, normalized_logvars = train_state.apply_fn(
        {"params": train_state.params},
        tiled_inputs,
    )

    member_delta_means = denormalize_delta(
        normalized_means,
        normalizer,
    )
    member_delta_variances = (
        jnp.exp(normalized_logvars)
        * jnp.square(normalizer.delta_std)[None, None, :]
    )
    mean_delta = jnp.mean(member_delta_means, axis=0)
    aleatoric_variance = jnp.mean(member_delta_variances, axis=0)
    epistemic_variance = jnp.var(member_delta_means, axis=0)
    total_variance = aleatoric_variance + epistemic_variance

    true_delta = agent_state_delta(
        batch.model_states,
        batch.next_model_states,
    )
    errors = mean_delta - true_delta
    position_indices = jnp.asarray([0, 1, 4, 5, 8, 9])
    velocity_indices = jnp.asarray([2, 3, 6, 7, 10, 11])
    collision_mask = jnp.asarray(batch.pair_collision_rates) > 0.0

    lower = mean_delta - 1.96 * jnp.sqrt(jnp.maximum(total_variance, 1.0e-12))
    upper = mean_delta + 1.96 * jnp.sqrt(jnp.maximum(total_variance, 1.0e-12))
    coverage = jnp.mean((true_delta >= lower) & (true_delta <= upper))

    nll = diagonal_gaussian_nll(
        normalized_means,
        normalized_logvars,
        tile_for_ensemble(normalized_targets, ensemble_size),
    )
    velocity_errors = errors[:, velocity_indices]

    return {
        "nll": nll,
        "state_rmse": jnp.sqrt(jnp.mean(jnp.square(errors))),
        "position_rmse": jnp.sqrt(
            jnp.mean(jnp.square(errors[:, position_indices]))
        ),
        "velocity_rmse": jnp.sqrt(
            jnp.mean(jnp.square(velocity_errors))
        ),
        "noncollision_velocity_rmse": _masked_rmse(
            velocity_errors,
            jnp.logical_not(collision_mask),
        ),
        "collision_velocity_rmse": _masked_rmse(
            velocity_errors,
            collision_mask,
        ),
        "epistemic_mean": jnp.mean(epistemic_variance),
        "aleatoric_mean": jnp.mean(aleatoric_variance),
        "coverage95": coverage,
    }
