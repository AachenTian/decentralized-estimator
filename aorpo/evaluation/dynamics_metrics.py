"""Metrics for deterministic and probabilistic local dynamics prediction."""

from __future__ import annotations

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp


Array = jax.Array
PredictNext = Callable[[Array, Array], Array]


def wrapped_angle_error(predicted: Array, target: Array) -> Array:
    """Shortest signed angular difference."""
    difference = predicted - target
    return jnp.arctan2(jnp.sin(difference), jnp.cos(difference))


def _state_error(predicted: Array, target: Array) -> Array:
    error = predicted - target
    return error.at[..., 2].set(
        wrapped_angle_error(predicted[..., 2], target[..., 2])
    )


def one_step_metrics(
    predicted_next: Array,
    true_next: Array,
) -> dict[str, Array]:
    """Return physically interpretable one-step errors."""
    if predicted_next.shape != true_next.shape:
        raise ValueError("predicted_next and true_next must have equal shapes.")
    if predicted_next.shape[-1] != 5:
        raise ValueError("Expected state order [px, py, phi, v, omega].")

    error = _state_error(predicted_next, true_next)
    squared = error**2

    return {
        "state_rmse": jnp.sqrt(jnp.mean(squared)),
        "position_rmse": jnp.sqrt(
            jnp.mean(jnp.sum(squared[..., 0:2], axis=-1))
        ),
        "heading_rmse": jnp.sqrt(jnp.mean(squared[..., 2])),
        "linear_velocity_rmse": jnp.sqrt(jnp.mean(squared[..., 3])),
        "angular_velocity_rmse": jnp.sqrt(jnp.mean(squared[..., 4])),
        "per_dimension_rmse": jnp.sqrt(
            jnp.mean(squared, axis=tuple(range(squared.ndim - 1)))
        ),
    }


def rollout_predictions(
    predict_next: PredictNext,
    initial_states: Array,
    actions: Array,
) -> Array:
    """Open-loop rollout using predicted state as the next model input.

    Args:
        initial_states: (..., 5)
        actions: (H, ..., 2)

    Returns:
        predictions: (H + 1, ..., 5)
    """
    predictions = [initial_states]
    current = initial_states
    for step in range(actions.shape[0]):
        current = predict_next(current, actions[step])
        predictions.append(current)
    return jnp.stack(predictions, axis=0)


def rollout_metrics(
    predicted_trajectory: Array,
    true_trajectory: Array,
    horizons: tuple[int, ...] = (1, 5, 10, 25, 50),
) -> dict[str, dict[str, Array]]:
    """Compute state metrics at selected open-loop horizons."""
    if predicted_trajectory.shape != true_trajectory.shape:
        raise ValueError("Predicted and true trajectories must match.")
    if predicted_trajectory.shape[0] < 2:
        raise ValueError("Trajectory must contain initial and next state.")

    max_horizon = predicted_trajectory.shape[0] - 1
    result: dict[str, dict[str, Array]] = {}
    for horizon in horizons:
        if horizon <= max_horizon:
            result[str(horizon)] = one_step_metrics(
                predicted_trajectory[horizon],
                true_trajectory[horizon],
            )
    return result


def gaussian_nll(
    predicted_mean: Array,
    predicted_covariance: Array,
    target: Array,
    jitter: float = 1e-6,
) -> Array:
    """Mean multivariate Gaussian negative log likelihood."""
    if predicted_mean.shape != target.shape:
        raise ValueError("Mean and target shapes must match.")
    dimension = predicted_mean.shape[-1]
    expected_cov_shape = predicted_mean.shape[:-1] + (dimension, dimension)
    if predicted_covariance.shape != expected_cov_shape:
        raise ValueError(
            f"Expected covariance shape {expected_cov_shape}, "
            f"got {predicted_covariance.shape}."
        )

    eye = jnp.eye(dimension, dtype=predicted_covariance.dtype)
    covariance = predicted_covariance + jitter * eye
    error = (target - predicted_mean)[..., None]

    sign, logdet = jnp.linalg.slogdet(covariance)
    quadratic = jnp.squeeze(
        jnp.swapaxes(error, -1, -2)
        @ jnp.linalg.solve(covariance, error),
        axis=(-1, -2),
    )
    nll = 0.5 * (
        dimension * math.log(2.0 * math.pi)
        + logdet
        + quadratic
    )
    return jnp.mean(jnp.where(sign > 0, nll, jnp.inf))


def marginal_coverage(
    predicted_mean: Array,
    predicted_covariance: Array,
    target: Array,
    sigma_scale: float,
) -> Array:
    """Per-dimension empirical Gaussian interval coverage."""
    variance = jnp.diagonal(
        predicted_covariance,
        axis1=-2,
        axis2=-1,
    )
    std = jnp.sqrt(jnp.maximum(variance, 1e-12))
    error = jnp.abs(_state_error(predicted_mean, target))
    covered = error <= sigma_scale * std
    return jnp.mean(covered, axis=tuple(range(covered.ndim - 1)))


def ensemble_epistemic_covariance(
    ensemble_means: Array,
) -> Array:
    """Estimate epistemic covariance from member mean spread.

    Args:
        ensemble_means: (..., E, D)

    Returns:
        covariance: (..., D, D)
    """
    ensemble_mean = jnp.mean(ensemble_means, axis=-2, keepdims=True)
    centered = ensemble_means - ensemble_mean
    ensemble_size = ensemble_means.shape[-2]
    denominator = max(ensemble_size - 1, 1)
    return jnp.einsum(
        "...ed,...ef->...df",
        centered,
        centered,
    ) / denominator
