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
