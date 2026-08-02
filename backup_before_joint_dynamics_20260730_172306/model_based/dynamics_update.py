from __future__ import annotations

import jax
import jax.numpy as jnp

from sender_marl.model_based.dynamics_state import DynamicsStandardizer
from sender_marl.model_based.types import EnsemblePrediction, LocalDynamicsBatch


Array = jax.Array


def _member_gaussian_nll(mean: Array, logvar: Array, target: Array) -> Array:
    inverse_variance = jnp.exp(-logvar)
    per_dimension = 0.5 * (
        jnp.square(mean - target) * inverse_variance
        + logvar
        + jnp.log(2.0 * jnp.pi)
    )
    return jnp.sum(per_dimension, axis=-1)


def make_shared_dynamics_train_step(
    *,
    ensemble_size: int,
    jit: bool = True,
):
    """Create one bootstrapped gradient update for the shared ensemble."""

    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive.")

    def train_step(
        key: Array,
        train_state,
        standardizer: DynamicsStandardizer,
        batch: LocalDynamicsBatch,
    ):
        batch_size = batch.local_states.shape[0]
        indices = jax.random.randint(
            key,
            shape=(ensemble_size, batch_size),
            minval=0,
            maxval=batch_size,
        )
        delta = batch.next_local_states - batch.local_states
        state_norm = standardizer.normalize_state(batch.local_states)
        action_norm = standardizer.normalize_action(batch.actions)
        delta_norm = standardizer.normalize_delta(delta)
        state_bootstrap = state_norm[indices]
        action_bootstrap = action_norm[indices]
        target_bootstrap = delta_norm[indices]
        inputs = jnp.concatenate(
            [state_bootstrap, action_bootstrap],
            axis=-1,
        )

        def loss_fn(params):
            mean, logvar = train_state.apply_fn(
                {"params": params},
                inputs,
            )
            member_nll = _member_gaussian_nll(
                mean,
                logvar,
                target_bootstrap,
            )
            loss = jnp.mean(member_nll)
            metrics = {
                "dynamics_nll": loss,
                "normalized_delta_mse": jnp.mean(
                    jnp.square(mean - target_bootstrap)
                ),
                "mean_logvar": jnp.mean(logvar),
                "max_abs_normalized_error": jnp.max(
                    jnp.abs(mean - target_bootstrap)
                ),
            }
            return loss, metrics

        (_, metrics), gradients = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(train_state.params)
        new_state = train_state.apply_gradients(grads=gradients)
        return new_state, metrics

    return jax.jit(train_step) if jit else train_step


def predict_shared_dynamics(
    train_state,
    standardizer: DynamicsStandardizer,
    local_states: Array,
    actions: Array,
) -> EnsemblePrediction:
    """Predict shared factorized next states and decomposed uncertainty."""

    states = jnp.asarray(local_states, dtype=jnp.float32)
    acts = jnp.asarray(actions, dtype=jnp.float32)
    if states.ndim != 2 or acts.ndim != 2:
        raise ValueError("local_states and actions must both be rank two.")
    if states.shape[0] != acts.shape[0]:
        raise ValueError("State and action batch sizes differ.")

    state_norm = standardizer.normalize_state(states)
    action_norm = standardizer.normalize_action(acts)
    inputs = jnp.concatenate([state_norm, action_norm], axis=-1)

    # Model uses ensemble-first input. Every member evaluates the same rows.
    ensemble_size = jax.tree_util.tree_leaves(train_state.params)[0].shape[0]
    member_inputs = jnp.broadcast_to(
        inputs[None, ...],
        (ensemble_size,) + inputs.shape,
    )
    delta_mean_norm_kbd, delta_logvar_norm_kbd = train_state.apply_fn(
        {"params": train_state.params},
        member_inputs,
    )
    delta_mean_norm = jnp.swapaxes(delta_mean_norm_kbd, 0, 1)
    delta_logvar_norm = jnp.swapaxes(delta_logvar_norm_kbd, 0, 1)

    delta_means = standardizer.denormalize_delta(delta_mean_norm)
    delta_variances = (
        jnp.exp(delta_logvar_norm)
        * jnp.square(standardizer.delta_std)[None, None, :]
    )
    ensemble_next_means = states[:, None, :] + delta_means
    next_mean = jnp.mean(ensemble_next_means, axis=1)
    aleatoric_variance = jnp.mean(delta_variances, axis=1)
    epistemic_variance = jnp.mean(
        jnp.square(ensemble_next_means - next_mean[:, None, :]),
        axis=1,
    )
    total_variance = aleatoric_variance + epistemic_variance
    return EnsemblePrediction(
        ensemble_next_means=ensemble_next_means,
        ensemble_next_variances=delta_variances,
        next_mean=next_mean,
        aleatoric_variance=aleatoric_variance,
        epistemic_variance=epistemic_variance,
        total_variance=total_variance,
        ensemble_delta_means_norm=delta_mean_norm,
        ensemble_delta_logvars_norm=delta_logvar_norm,
    )


def evaluate_shared_dynamics_batch(
    train_state,
    standardizer: DynamicsStandardizer,
    batch: LocalDynamicsBatch,
) -> dict[str, Array]:
    prediction = predict_shared_dynamics(
        train_state,
        standardizer,
        batch.local_states,
        batch.actions,
    )
    error = prediction.next_mean - batch.next_local_states
    target_delta_norm = standardizer.normalize_delta(
        batch.next_local_states - batch.local_states
    )
    member_log_prob = -_member_gaussian_nll(
        jnp.swapaxes(prediction.ensemble_delta_means_norm, 0, 1),
        jnp.swapaxes(prediction.ensemble_delta_logvars_norm, 0, 1),
        target_delta_norm[None, ...],
    )
    mixture_log_prob = jax.scipy.special.logsumexp(
        member_log_prob,
        axis=0,
    ) - jnp.log(member_log_prob.shape[0])
    standard_deviation = jnp.sqrt(
        jnp.maximum(prediction.total_variance, 1e-12)
    )
    interval_coverage_95 = jnp.mean(
        jnp.abs(error) <= 1.96 * standard_deviation
    )
    return {
        "state_rmse": jnp.sqrt(jnp.mean(jnp.square(error))),
        "state_mae": jnp.mean(jnp.abs(error)),
        "mixture_nll": -jnp.mean(mixture_log_prob),
        "mean_aleatoric_std": jnp.mean(
            jnp.sqrt(jnp.maximum(prediction.aleatoric_variance, 0.0))
        ),
        "mean_epistemic_std": jnp.mean(
            jnp.sqrt(jnp.maximum(prediction.epistemic_variance, 0.0))
        ),
        "interval_coverage_95": interval_coverage_95,
        "per_dimension_rmse": jnp.sqrt(
            jnp.mean(jnp.square(error), axis=0)
        ),
    }
