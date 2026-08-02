"""Moment-matched predictions from a five-member dynamics ensemble."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from private_mb_sac.core.types import DynamicsPrediction
from private_mb_sac.dynamics.ensemble import tile_for_ensemble
from private_mb_sac.dynamics.normalization import denormalize_delta, normalize_dynamics_inputs


def predict_dynamics_distribution(train_state, normalizer, model_states, actions,
                                  *, ensemble_size: int) -> DynamicsPrediction:
    inputs = normalize_dynamics_inputs(model_states, actions, normalizer)
    member_inputs = tile_for_ensemble(inputs, ensemble_size)
    normalized_means, normalized_logvars = train_state.apply_fn(
        {'params': train_state.params}, member_inputs
    )
    member_means = denormalize_delta(normalized_means, normalizer)
    member_variances = jnp.exp(normalized_logvars) * jnp.square(normalizer.delta_std)[None, None, :]
    mean_delta = jnp.mean(member_means, axis=0)
    aleatoric = jnp.mean(member_variances, axis=0)
    epistemic = jnp.var(member_means, axis=0)
    total = aleatoric + epistemic
    return DynamicsPrediction(member_means, member_variances, mean_delta,
                              aleatoric, epistemic, total)


def sample_moment_matched_delta(key, prediction: DynamicsPrediction, *, stochastic: bool):
    if not stochastic:
        return prediction.mean_delta
    noise = jax.random.normal(key, prediction.mean_delta.shape)
    return prediction.mean_delta + jnp.sqrt(jnp.maximum(prediction.total_variance, 1.0e-12)) * noise


def apply_agent_delta(model_states, delta, *, agent_state_dim: int=12):
    states = jnp.asarray(model_states, dtype=jnp.float32)
    delta = jnp.asarray(delta, dtype=jnp.float32)
    return jnp.concatenate([states[..., :agent_state_dim] + delta,
                            states[..., agent_state_dim:]], axis=-1)


def epistemic_score(prediction: DynamicsPrediction, metric: str):
    if metric == 'max_epistemic_variance':
        return jnp.max(prediction.epistemic_variance, axis=-1)
    if metric == 'mean_epistemic_variance':
        return jnp.mean(prediction.epistemic_variance, axis=-1)
    raise ValueError(f'Unsupported uncertainty metric: {metric}')
