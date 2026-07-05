# aorpo/estimator/local_belief.py
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from flax.training.train_state import TrainState

from aorpo.agents.independent_dynamics import LocalStandardizerRS
from aorpo.estimator.uncertainty import (
    compute_local_action_jacobian,
    compute_local_state_jacobian,
    predict_local_mean_and_process_covariance,
    propagate_local_covariance,
    symmetrize_covariance,
)


class LocalBelief(NamedTuple):
    """
    Gaussian belief over one agent's local physical state.

    Attributes:
        mean:       Estimated local state with shape (B, D).
        covariance: Estimated covariance with shape (B, D, D).
    """

    mean: jnp.ndarray
    covariance: jnp.ndarray


def make_isotropic_covariance(
    batch_size: int,
    dimension: int,
    variance: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Create a batch of isotropic covariance matrices.

    Args:
        batch_size: Number of covariance matrices.
        dimension:  State or action dimension.
        variance:   Shared diagonal variance.
        dtype:      Output dtype.

    Returns:
        Covariance matrices with shape (B, D, D).
    """
    if variance < 0.0:
        raise ValueError(
            f"variance must be non-negative, got {variance}."
        )

    covariance = variance * jnp.eye(
        dimension,
        dtype=dtype,
    )

    return jnp.broadcast_to(
        covariance,
        (batch_size, dimension, dimension),
    )


def initialize_local_belief(
    initial_mean: jnp.ndarray,
    initial_variance: float,
) -> LocalBelief:
    """
    Initialize a local Gaussian belief.

    The initial mean should normally be the true local state obtained
    after an initial synchronization or direct observation.

    Args:
        initial_mean:     Initial state mean with shape (B, D).
        initial_variance: Diagonal initial covariance variance.

    Returns:
        Initialized LocalBelief.
    """
    if initial_mean.ndim != 2:
        raise ValueError(
            "initial_mean must have shape (B, D), "
            f"got {initial_mean.shape}."
        )

    batch_size, state_dim = initial_mean.shape

    initial_covariance = make_isotropic_covariance(
        batch_size=batch_size,
        dimension=state_dim,
        variance=initial_variance,
        dtype=initial_mean.dtype,
    )

    return LocalBelief(
        mean=initial_mean,
        covariance=initial_covariance,
    )


def predict_local_belief(
    belief: LocalBelief,
    train_state: TrainState,
    standardizer: LocalStandardizerRS,
    local_action: jnp.ndarray,
    include_action_uncertainty: bool = False,
    action_covariance: Optional[jnp.ndarray] = None,
) -> Tuple[LocalBelief, Dict[str, jnp.ndarray]]:
    """
    Perform one EKF-style prediction step for a local belief.

    Without action uncertainty:
        P_next = F P F^T + Q

    With action uncertainty:
        P_next = F P F^T + G P_action G^T + Q

    Args:
        belief:                     Current local Gaussian belief.
        train_state:                Trained local dynamics model state.
        standardizer:               Local dynamics normalizer.
        local_action:               Action with shape (B, A).
        include_action_uncertainty: Whether to include G P_action G^T.
        action_covariance:          Action covariance with shape (B, A, A).

    Returns:
        predicted_belief: Predicted mean and covariance.
        prediction_info: Diagnostics including Jacobians and process covariance.
    """
    if belief.mean.ndim != 2:
        raise ValueError(
            "belief.mean must have shape (B, D), "
            f"got {belief.mean.shape}."
        )

    if belief.covariance.ndim != 3:
        raise ValueError(
            "belief.covariance must have shape (B, D, D), "
            f"got {belief.covariance.shape}."
        )

    if include_action_uncertainty and action_covariance is None:
        raise ValueError(
            "action_covariance is required when "
            "include_action_uncertainty=True."
        )

    predicted_mean, process_covariance, covariance_info = (
        predict_local_mean_and_process_covariance(
            train_state=train_state,
            standardizer=standardizer,
            local_state=belief.mean,
            local_action=local_action,
        )
    )

    state_jacobian = compute_local_state_jacobian(
        train_state=train_state,
        standardizer=standardizer,
        local_state=belief.mean,
        local_action=local_action,
    )

    prediction_info = {
        **covariance_info,
        "state_jacobian": state_jacobian,
    }

    if include_action_uncertainty:
        action_jacobian = compute_local_action_jacobian(
            train_state=train_state,
            standardizer=standardizer,
            local_state=belief.mean,
            local_action=local_action,
        )

        predicted_covariance = propagate_local_covariance(
            state_covariance=belief.covariance,
            state_jacobian=state_jacobian,
            process_covariance=process_covariance,
            action_jacobian=action_jacobian,
            action_covariance=action_covariance,
            include_action_uncertainty=True,
        )

        prediction_info["action_jacobian"] = action_jacobian
    else:
        predicted_covariance = propagate_local_covariance(
            state_covariance=belief.covariance,
            state_jacobian=state_jacobian,
            process_covariance=process_covariance,
            include_action_uncertainty=False,
        )

    return (
        LocalBelief(
            mean=predicted_mean,
            covariance=predicted_covariance,
        ),
        prediction_info,
    )


def direct_observation_update(
    predicted_belief: LocalBelief,
    measurement: jnp.ndarray,
    measurement_covariance: jnp.ndarray,
    jitter: float = 1e-9,
) -> Tuple[LocalBelief, Dict[str, jnp.ndarray]]:
    """
    Apply a direct-state Kalman correction in Joseph form.

    The measurement model is:
        y = z + v

    Therefore:
        H = I
        S = P_minus + R
        K = P_minus S^{-1}

    Args:
        predicted_belief:     Prior local belief after prediction.
        measurement:          Received local state with shape (B, D).
        measurement_covariance:
                              Measurement covariance R with shape (B, D, D).
        jitter:               Numerical stabilization term.

    Returns:
        posterior_belief: Updated local belief after communication.
        update_info:      Innovation, innovation covariance, and Kalman gain.
    """
    prior_mean = predicted_belief.mean
    prior_covariance = predicted_belief.covariance

    if measurement.shape != prior_mean.shape:
        raise ValueError(
            "measurement and belief mean must have identical shapes. "
            f"Received measurement={measurement.shape}, "
            f"mean={prior_mean.shape}."
        )

    if measurement_covariance.shape != prior_covariance.shape:
        raise ValueError(
            "measurement_covariance and belief covariance must have "
            "identical shapes. "
            f"Received measurement_covariance={measurement_covariance.shape}, "
            f"covariance={prior_covariance.shape}."
        )

    innovation = measurement - prior_mean

    innovation_covariance = symmetrize_covariance(
        prior_covariance + measurement_covariance,
        jitter=jitter,
    )

    kalman_gain_transpose = jnp.linalg.solve(
        jnp.swapaxes(innovation_covariance, -1, -2),
        jnp.swapaxes(prior_covariance, -1, -2),
    )

    kalman_gain = jnp.swapaxes(
        kalman_gain_transpose,
        -1,
        -2,
    )

    posterior_mean = prior_mean + jnp.einsum(
        "bij,bj->bi",
        kalman_gain,
        innovation,
    )

    state_dim = prior_mean.shape[-1]

    identity = jnp.broadcast_to(
        jnp.eye(
            state_dim,
            dtype=prior_covariance.dtype,
        ),
        prior_covariance.shape,
    )

    residual_transform = identity - kalman_gain

    posterior_covariance = (
        residual_transform
        @ prior_covariance
        @ jnp.swapaxes(residual_transform, -1, -2)
        + kalman_gain
        @ measurement_covariance
        @ jnp.swapaxes(kalman_gain, -1, -2)
    )

    posterior_covariance = symmetrize_covariance(
        posterior_covariance,
        jitter=jitter,
    )

    posterior_belief = LocalBelief(
        mean=posterior_mean,
        covariance=posterior_covariance,
    )

    update_info = {
        "innovation": innovation,
        "innovation_covariance": innovation_covariance,
        "kalman_gain": kalman_gain,
    }

    return posterior_belief, update_info