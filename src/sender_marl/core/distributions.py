from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Array = jax.Array
_LOG_2PI = float(jnp.log(2.0 * jnp.pi))
_LOG_2 = float(jnp.log(2.0))


@dataclass(frozen=True)
class SquashedGaussianSample:
    """A sample from a diagonal Gaussian followed by a tanh transform."""

    action: Array
    pre_tanh_action: Array
    log_prob: Array


def clip_log_std(
    log_std: Array,
    minimum: float = -5.0,
    maximum: float = 2.0,
) -> Array:
    """Bound policy log standard deviations for numerical stability."""

    return jnp.clip(jnp.asarray(log_std), minimum, maximum)


def diagonal_gaussian_log_prob(
    value: Array,
    mean: Array,
    log_std: Array,
) -> Array:
    """Log probability of a diagonal Gaussian, summed over action dimensions."""

    value = jnp.asarray(value)
    mean = jnp.asarray(mean)
    log_std = clip_log_std(log_std)
    inv_std = jnp.exp(-log_std)
    normalized = (value - mean) * inv_std
    elementwise = -0.5 * (
        jnp.square(normalized) + 2.0 * log_std + _LOG_2PI
    )
    return jnp.sum(elementwise, axis=-1)


def tanh_log_abs_det_jacobian(pre_tanh_action: Array) -> Array:
    """Stable log |det d tanh(z) / dz|, summed over action dimensions."""

    z = jnp.asarray(pre_tanh_action)
    elementwise = 2.0 * (_LOG_2 - z - jax.nn.softplus(-2.0 * z))
    return jnp.sum(elementwise, axis=-1)


def squashed_gaussian_log_prob_from_pre_tanh(
    pre_tanh_action: Array,
    mean: Array,
    log_std: Array,
) -> Array:
    """Exact transformed log probability using the stored pre-tanh action."""

    base_log_prob = diagonal_gaussian_log_prob(
        pre_tanh_action,
        mean,
        log_std,
    )
    return base_log_prob - tanh_log_abs_det_jacobian(pre_tanh_action)


def inverse_tanh(action: Array, epsilon: float = 1e-6) -> Array:
    """Numerically stable inverse tanh for actions in [-1, 1]."""

    clipped = jnp.clip(jnp.asarray(action), -1.0 + epsilon, 1.0 - epsilon)
    return 0.5 * (jnp.log1p(clipped) - jnp.log1p(-clipped))


def squashed_gaussian_log_prob(
    action: Array,
    mean: Array,
    log_std: Array,
    epsilon: float = 1e-6,
) -> Array:
    """Transformed log probability when only the bounded action is available."""

    pre_tanh_action = inverse_tanh(action, epsilon=epsilon)
    return squashed_gaussian_log_prob_from_pre_tanh(
        pre_tanh_action,
        mean,
        log_std,
    )


def sample_squashed_gaussian(
    key: Array,
    mean: Array,
    log_std: Array,
) -> SquashedGaussianSample:
    """Sample bounded actions and return the corrected transformed log-probability."""

    mean = jnp.asarray(mean)
    log_std = clip_log_std(log_std)
    noise = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    pre_tanh_action = mean + jnp.exp(log_std) * noise
    action = jnp.tanh(pre_tanh_action)
    log_prob = squashed_gaussian_log_prob_from_pre_tanh(
        pre_tanh_action,
        mean,
        log_std,
    )
    return SquashedGaussianSample(
        action=action,
        pre_tanh_action=pre_tanh_action,
        log_prob=log_prob,
    )


def deterministic_squashed_action(mean: Array) -> Array:
    """Deterministic action used during policy evaluation."""

    return jnp.tanh(jnp.asarray(mean))


def diagonal_gaussian_entropy(log_std: Array) -> Array:
    """Entropy of the unsquashed Gaussian, summed over action dimensions.

    A tanh-squashed Gaussian has no simple closed-form entropy. PPO commonly
    uses this base-Gaussian entropy as a stable exploration diagnostic/bonus.
    """

    bounded_log_std = clip_log_std(log_std)
    return jnp.sum(
        bounded_log_std + 0.5 * (1.0 + _LOG_2PI),
        axis=-1,
    )
