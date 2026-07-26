"""Normalized full-state sender-trigger functions.

The local differential-drive state is:

    [p_x, p_y, phi, v, omega]

These components use different physical units, so the trigger first normalizes
each component by a user-specified positive scale.
"""

from __future__ import annotations

import jax.numpy as jnp


def _validated_state_scales(
    state_scales: jnp.ndarray,
    state_dim: int,
) -> jnp.ndarray:
    scales = jnp.asarray(state_scales)

    if scales.ndim != 1:
        raise ValueError(
            "state_scales must be a one-dimensional vector."
        )
    if scales.shape[0] != state_dim:
        raise ValueError(
            "state_scales length must match the state dimension: "
            f"{scales.shape[0]} versus {state_dim}."
        )
    if bool(jnp.any(scales <= 0.0)):
        raise ValueError("Every state scale must be positive.")

    return scales


def normalized_full_state_error_score(
    predicted_local_mean: jnp.ndarray,
    observed_local_state: jnp.ndarray,
    state_scales: jnp.ndarray,
) -> jnp.ndarray:
    """Return a dimensionless normalized full-state RMS error.

    For local state dimension D:

        e = sqrt(mean(((observation - prediction) / scale)^2))

    The heading is assumed to use the same unwrapped convention as the
    generated differential-drive dataset.
    """
    if predicted_local_mean.shape[-1] != observed_local_state.shape[-1]:
        raise ValueError(
            "Predicted and observed states must have equal dimensions."
        )

    state_dim = predicted_local_mean.shape[-1]
    scales = _validated_state_scales(state_scales, state_dim)

    normalized_residual = (
        observed_local_state - predicted_local_mean
    ) / scales

    return jnp.sqrt(
        jnp.mean(normalized_residual ** 2, axis=-1)
    )


def normalized_full_state_uncertainty_score(
    predicted_local_covariance: jnp.ndarray,
    state_scales: jnp.ndarray,
    sigma_scale: float = 2.0,
) -> jnp.ndarray:
    """Return a dimensionless normalized full-state uncertainty score.

    The covariance is transformed into normalized coordinates:

        P_n = D_s^{-1} P D_s^{-1}

    and the score is:

        u = sigma_scale * sqrt(trace(P_n) / D)

    Thus all five state variances contribute after unit normalization.
    """
    if sigma_scale <= 0.0:
        raise ValueError("sigma_scale must be positive.")

    if (
        predicted_local_covariance.shape[-1]
        != predicted_local_covariance.shape[-2]
    ):
        raise ValueError("Covariance must be square.")

    state_dim = predicted_local_covariance.shape[-1]
    scales = _validated_state_scales(state_scales, state_dim)

    normalized_covariance = (
        predicted_local_covariance
        / scales[..., :, None]
        / scales[..., None, :]
    )
    normalized_covariance = 0.5 * (
        normalized_covariance
        + jnp.swapaxes(normalized_covariance, -1, -2)
    )

    average_normalized_variance = (
        jnp.trace(
            normalized_covariance,
            axis1=-2,
            axis2=-1,
        )
        / float(state_dim)
    )
    average_normalized_variance = jnp.maximum(
        average_normalized_variance,
        0.0,
    )

    return sigma_scale * jnp.sqrt(
        average_normalized_variance
    )


def sender_normalized_full_state_trigger(
    predicted_local_mean: jnp.ndarray,
    observed_local_state: jnp.ndarray,
    predicted_local_covariance: jnp.ndarray,
    state_scales: jnp.ndarray,
    error_threshold: float,
    uncertainty_threshold: float,
    uncertainty_sigma_scale: float = 2.0,
) -> jnp.ndarray:
    """Broadcast when normalized full-state error OR uncertainty is large."""
    if error_threshold < 0.0:
        raise ValueError("error_threshold must be non-negative.")
    if uncertainty_threshold < 0.0:
        raise ValueError(
            "uncertainty_threshold must be non-negative."
        )

    error_score = normalized_full_state_error_score(
        predicted_local_mean=predicted_local_mean,
        observed_local_state=observed_local_state,
        state_scales=state_scales,
    )
    uncertainty_score = normalized_full_state_uncertainty_score(
        predicted_local_covariance=predicted_local_covariance,
        state_scales=state_scales,
        sigma_scale=uncertainty_sigma_scale,
    )

    return (
        (error_score > error_threshold)
        | (uncertainty_score > uncertainty_threshold)
    )
