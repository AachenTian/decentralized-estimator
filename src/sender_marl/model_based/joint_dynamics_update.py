from __future__ import annotations

import jax
import jax.numpy as jnp

from sender_marl.model_based.joint_dynamics_state import (
    JointDynamicsStandardizer,
)
from sender_marl.model_based.joint_types import JointDynamicsBatch
from sender_marl.model_based.types import EnsemblePrediction


Array = jax.Array


def _member_gaussian_nll(mean: Array, logvar: Array, target: Array) -> Array:
    inverse_variance = jnp.exp(-logvar)
    per_dimension = 0.5 * (
        jnp.square(mean - target) * inverse_variance
        + logvar
        + jnp.log(2.0 * jnp.pi)
    )
    return jnp.sum(per_dimension, axis=-1)


def make_independent_joint_dynamics_train_step(
    *,
    ensemble_size: int,
    agent_id: int,
    jit: bool = True,
):
    """Create one bootstrapped update for one agent-owned model."""

    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive.")
    if agent_id < 0:
        raise ValueError("agent_id must be non-negative.")

    def train_step(
        key: Array,
        train_state,
        standardizer: JointDynamicsStandardizer,
        batch: JointDynamicsBatch,
    ):
        batch_size = batch.model_states.shape[0]
        indices = jax.random.randint(
            key,
            shape=(ensemble_size, batch_size),
            minval=0,
            maxval=batch_size,
        )

        model_state_norm = standardizer.normalize_model_state(
            batch.model_states
        )
        joint_actions_flat = batch.joint_actions.reshape(batch_size, -1)
        joint_action_norm = standardizer.normalize_joint_action(
            joint_actions_flat
        )
        local_state = batch.local_states[:, agent_id, :]
        next_local_state = batch.next_local_states[:, agent_id, :]
        delta_norm = standardizer.normalize_delta(
            next_local_state - local_state
        )

        state_bootstrap = model_state_norm[indices]
        action_bootstrap = joint_action_norm[indices]
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


def predict_independent_joint_dynamics(
    train_state,
    standardizer: JointDynamicsStandardizer,
    model_states: Array,
    joint_actions: Array,
    owned_local_states: Array,
) -> EnsemblePrediction:
    """Predict one owned agent's next local state."""

    states = jnp.asarray(model_states, dtype=jnp.float32)
    actions = jnp.asarray(joint_actions, dtype=jnp.float32)
    local = jnp.asarray(owned_local_states, dtype=jnp.float32)
    if states.ndim != 2:
        raise ValueError("model_states must have shape (B, S).")
    if actions.ndim != 3:
        raise ValueError("joint_actions must have shape (B, N, A).")
    if local.ndim != 2:
        raise ValueError("owned_local_states must have shape (B, D).")
    if states.shape[0] != actions.shape[0] or states.shape[0] != local.shape[0]:
        raise ValueError("Prediction batch sizes differ.")

    state_norm = standardizer.normalize_model_state(states)
    actions_flat = actions.reshape(actions.shape[0], -1)
    action_norm = standardizer.normalize_joint_action(actions_flat)
    inputs = jnp.concatenate([state_norm, action_norm], axis=-1)

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
    ensemble_next_means = local[:, None, :] + delta_means
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


def evaluate_independent_joint_dynamics_batch(
    train_state,
    standardizer: JointDynamicsStandardizer,
    batch: JointDynamicsBatch,
    *,
    agent_id: int,
) -> dict[str, Array]:
    local = batch.local_states[:, agent_id, :]
    next_local = batch.next_local_states[:, agent_id, :]
    prediction = predict_independent_joint_dynamics(
        train_state,
        standardizer,
        batch.model_states,
        batch.joint_actions,
        local,
    )
    error = prediction.next_mean - next_local
    target_delta_norm = standardizer.normalize_delta(next_local - local)
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
        "interval_coverage_95": jnp.mean(
            jnp.abs(error) <= 1.96 * standard_deviation
        ),
        "per_dimension_rmse": jnp.sqrt(
            jnp.mean(jnp.square(error), axis=0)
        ),
    }
