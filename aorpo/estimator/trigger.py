# aorpo/estimator/trigger.py
from __future__ import annotations

import jax.numpy as jnp

from aorpo.estimator.uncertainty import (
    covariance_trace_per_dim,
)


def state_uncertainty_score(
    covariance: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the average per-dimension state uncertainty.

    Args:
        covariance:
            State covariance with shape (..., D, D).

    Returns:
        Uncertainty score with shape (...,).

    The score is trace(P) / D.
    """
    return covariance_trace_per_dim(covariance)


def trace_threshold_trigger(
    covariance: jnp.ndarray,
    threshold: float,
) -> jnp.ndarray:
    """
    Trigger communication when state uncertainty exceeds a threshold.

    Args:
        covariance:
            State covariance with shape (..., D, D).

        threshold:
            Non-negative threshold applied to trace(P) / D.

    Returns:
        Boolean trigger decision with shape (...,).
    """
    if threshold < 0.0:
        raise ValueError(
            f"threshold must be non-negative, got {threshold}."
        )

    return state_uncertainty_score(covariance) > threshold

def position_uncertainty_radius(
    covariance: jnp.ndarray,
    scale: float = 2.0,
) -> jnp.ndarray:
    """
    Compute the major semi-axis of the position uncertainty ellipse.

    Args:
        covariance:
            Local state covariance with shape (..., D, D).
            The first two state dimensions are assumed to be position.

        scale:
            Standard-deviation multiplier. For example, 2.0 gives the
            2-sigma position uncertainty radius.

    Returns:
        Position uncertainty radius with shape (...,), in environment
        position units.
    """
    if scale <= 0.0:
        raise ValueError(
            f"scale must be positive, got {scale}."
        )

    position_covariance = covariance[..., :2, :2]

    symmetric_position_covariance = 0.5 * (
        position_covariance
        + jnp.swapaxes(position_covariance, -1, -2)
    )

    eigenvalues = jnp.linalg.eigvalsh(
        symmetric_position_covariance
    )

    largest_eigenvalue = jnp.max(
        eigenvalues,
        axis=-1,
    )

    largest_eigenvalue = jnp.maximum(
        largest_eigenvalue,
        0.0,
    )

    return scale * jnp.sqrt(largest_eigenvalue)


def position_radius_threshold_trigger(
    covariance: jnp.ndarray,
    radius_threshold: float,
    scale: float = 2.0,
) -> jnp.ndarray:
    """
    Trigger communication when the position uncertainty radius exceeds a
    threshold.

    Args:
        covariance:
            Local state covariance with shape (..., D, D).

        radius_threshold:
            Non-negative threshold in environment position units.

        scale:
            Standard-deviation multiplier used for the position uncertainty
            radius.

    Returns:
        Boolean trigger decision with shape (...,).
    """
    if radius_threshold < 0.0:
        raise ValueError(
            "radius_threshold must be non-negative, "
            f"got {radius_threshold}."
        )

    radius = position_uncertainty_radius(
        covariance=covariance,
        scale=scale,
    )

    return radius > radius_threshold

def position_mean_error_score(
    predicted_local_mean: jnp.ndarray,
    observed_local_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute sender-side position mean error.

    Args:
        predicted_local_mean:
            Common predicted local mean with shape (..., 4).

        observed_local_state:
            Sender's private observed local state with shape (..., 4).

    Returns:
        Position error with shape (...,), in environment position units.
    """
    position_residual = (
        observed_local_state[..., :2]
        - predicted_local_mean[..., :2]
    )

    return jnp.linalg.norm(
        position_residual,
        axis=-1,
    )


def sender_position_trigger(
    predicted_local_mean: jnp.ndarray,
    observed_local_state: jnp.ndarray,
    predicted_local_covariance: jnp.ndarray,
    error_threshold: float,
    covariance_radius_threshold: float,
    covariance_scale: float = 2.0,
) -> jnp.ndarray:
    """
    Sender-side trigger using position mean error and position covariance.

    A sender broadcasts if either:
        position mean error > error_threshold
    or:
        position uncertainty radius > covariance_radius_threshold.
    """
    if error_threshold < 0.0:
        raise ValueError(
            "error_threshold must be non-negative, "
            f"got {error_threshold}."
        )

    if covariance_radius_threshold < 0.0:
        raise ValueError(
            "covariance_radius_threshold must be non-negative, "
            f"got {covariance_radius_threshold}."
        )

    error_score = position_mean_error_score(
        predicted_local_mean=predicted_local_mean,
        observed_local_state=observed_local_state,
    )

    covariance_score = position_uncertainty_radius(
        covariance=predicted_local_covariance,
        scale=covariance_scale,
    )

    return (
        (error_score > error_threshold)
        | (covariance_score > covariance_radius_threshold)
    )
