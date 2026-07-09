# aorpo/estimator/infoprop.py
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class InfopropPrediction(NamedTuple):
    """
    Container for Infoprop-style ensemble fusion outputs.

    All quantities are represented in the original state coordinate system.
    """

    fused_mean: jnp.ndarray
    fused_covariance: jnp.ndarray
    epistemic_covariance: jnp.ndarray
    posterior_mean: jnp.ndarray
    posterior_covariance: jnp.ndarray


def symmetrize_matrix(
    matrix: jnp.ndarray,
) -> jnp.ndarray:
    """
    Symmetrize a square matrix or a batch of square matrices.
    """
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def add_jitter(
    covariance: jnp.ndarray,
    jitter: float,
) -> jnp.ndarray:
    """
    Add diagonal jitter to a covariance matrix or batch of matrices.
    """
    state_dim = covariance.shape[-1]
    identity = jnp.eye(
        state_dim,
        dtype=covariance.dtype,
    )

    return covariance + jitter * identity


def invert_spd_matrix(
    matrix: jnp.ndarray,
    jitter: float,
) -> jnp.ndarray:
    """
    Invert a symmetric positive-definite matrix with jitter.
    """
    stabilized_matrix = add_jitter(
        covariance=symmetrize_matrix(matrix),
        jitter=jitter,
    )

    return jnp.linalg.inv(stabilized_matrix)


def covariance_intersection_fusion(
    ensemble_means: jnp.ndarray,
    ensemble_covariances: jnp.ndarray,
    jitter: float = 1.0e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fuse ensemble Gaussian predictions using uniform-weight covariance
    intersection.

    Args:
        ensemble_means:
            Ensemble means with shape (..., E, D).

        ensemble_covariances:
            Ensemble covariances with shape (..., E, D, D).

        jitter:
            Numerical stabilization added to covariance matrices.

    Returns:
        fused_mean:
            Fused mean with shape (..., D).

        fused_covariance:
            Fused covariance with shape (..., D, D).

    The fusion rule is:
        P_bar = (1/E * sum_e P_e^{-1})^{-1}
        m_bar = P_bar * (1/E * sum_e P_e^{-1} m_e)
    """
    if ensemble_means.ndim < 2:
        raise ValueError(
            "ensemble_means must have shape (..., E, D)."
        )

    if ensemble_covariances.ndim < 3:
        raise ValueError(
            "ensemble_covariances must have shape (..., E, D, D)."
        )

    if ensemble_means.shape[-2] != ensemble_covariances.shape[-3]:
        raise ValueError(
            "The ensemble dimension of means and covariances does not match. "
            f"Got {ensemble_means.shape[-2]} and "
            f"{ensemble_covariances.shape[-3]}."
        )

    if ensemble_means.shape[-1] != ensemble_covariances.shape[-1]:
        raise ValueError(
            "The state dimension of means and covariances does not match. "
            f"Got {ensemble_means.shape[-1]} and "
            f"{ensemble_covariances.shape[-1]}."
        )

    inverse_covariances = jax.vmap(
        lambda covariance: invert_spd_matrix(
            matrix=covariance,
            jitter=jitter,
        ),
        in_axes=-3,
        out_axes=-3,
    )(ensemble_covariances)

    mean_precision = jnp.mean(
        inverse_covariances,
        axis=-3,
    )

    fused_covariance = invert_spd_matrix(
        matrix=mean_precision,
        jitter=jitter,
    )

    precision_weighted_means = jnp.einsum(
        "...edc,...ec->...ed",
        inverse_covariances,
        ensemble_means,
    )

    mean_precision_weighted_mean = jnp.mean(
        precision_weighted_means,
        axis=-2,
    )

    fused_mean = jnp.einsum(
        "...dc,...c->...d",
        fused_covariance,
        mean_precision_weighted_mean,
    )

    return fused_mean, symmetrize_matrix(fused_covariance)


def estimate_epistemic_covariance(
    ensemble_means: jnp.ndarray,
    fused_mean: jnp.ndarray,
) -> jnp.ndarray:
    """
    Estimate epistemic model-error covariance from ensemble mean spread.

    Args:
        ensemble_means:
            Ensemble means with shape (..., E, D).

        fused_mean:
            Fused mean with shape (..., D).

    Returns:
        Epistemic covariance with shape (..., D, D).

    The estimator is:
        P_delta = 1/E * sum_e (m_e - m_bar)(m_e - m_bar)^T
    """
    centered_means = ensemble_means - fused_mean[..., None, :]

    epistemic_covariance = jnp.einsum(
        "...ed,...ef->...df",
        centered_means,
        centered_means,
    ) / float(ensemble_means.shape[-2])

    return symmetrize_matrix(epistemic_covariance)


def infoprop_posterior(
    fused_mean: jnp.ndarray,
    fused_covariance: jnp.ndarray,
    epistemic_covariance: jnp.ndarray,
    model_sample: jnp.ndarray | None = None,
    jitter: float = 1.0e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the Infoprop posterior state.

    Args:
        fused_mean:
            Estimated environment mean with shape (..., D).

        fused_covariance:
            Estimated environment covariance with shape (..., D, D).

        epistemic_covariance:
            Epistemic model-error covariance with shape (..., D, D).

        model_sample:
            Observed model sample with shape (..., D). If None, fused_mean
            is used as the model sample.

        jitter:
            Numerical stabilization.

    Returns:
        posterior_mean:
            Infoprop posterior mean with shape (..., D).

        posterior_covariance:
            Infoprop posterior covariance with shape (..., D, D).

    The conditioning model is:
        s_hat = s_bar + delta
        s_bar ~ N(fused_mean, fused_covariance)
        delta ~ N(0, epistemic_covariance)
    """
    if model_sample is None:
        model_sample = fused_mean

    innovation_covariance = (
        fused_covariance + epistemic_covariance
    )

    inverse_innovation_covariance = invert_spd_matrix(
        matrix=innovation_covariance,
        jitter=jitter,
    )

    kalman_gain = jnp.einsum(
        "...dc,...cf->...df",
        fused_covariance,
        inverse_innovation_covariance,
    )

    innovation = model_sample - fused_mean

    posterior_mean = fused_mean + jnp.einsum(
        "...dc,...c->...d",
        kalman_gain,
        innovation,
    )

    identity = jnp.eye(
        fused_covariance.shape[-1],
        dtype=fused_covariance.dtype,
    )

    posterior_covariance = (
        (identity - kalman_gain)
        @ fused_covariance
        @ jnp.swapaxes(identity - kalman_gain, -1, -2)
        + kalman_gain
        @ epistemic_covariance
        @ jnp.swapaxes(kalman_gain, -1, -2)
    )

    return posterior_mean, symmetrize_matrix(posterior_covariance)


def infoprop_ensemble_prediction(
    ensemble_means: jnp.ndarray,
    ensemble_covariances: jnp.ndarray,
    model_sample: jnp.ndarray | None = None,
    jitter: float = 1.0e-6,
) -> InfopropPrediction:
    """
    Apply Infoprop-style processing to ensemble Gaussian predictions.

    Args:
        ensemble_means:
            Ensemble means with shape (..., E, D).

        ensemble_covariances:
            Ensemble aleatoric covariances with shape (..., E, D, D).

        model_sample:
            Optional model sample with shape (..., D).

        jitter:
            Numerical stabilization.

    Returns:
        InfopropPrediction containing fused, epistemic, and posterior terms.
    """
    fused_mean, fused_covariance = covariance_intersection_fusion(
        ensemble_means=ensemble_means,
        ensemble_covariances=ensemble_covariances,
        jitter=jitter,
    )

    epistemic_covariance = estimate_epistemic_covariance(
        ensemble_means=ensemble_means,
        fused_mean=fused_mean,
    )

    posterior_mean, posterior_covariance = infoprop_posterior(
        fused_mean=fused_mean,
        fused_covariance=fused_covariance,
        epistemic_covariance=epistemic_covariance,
        model_sample=model_sample,
        jitter=jitter,
    )

    return InfopropPrediction(
        fused_mean=fused_mean,
        fused_covariance=fused_covariance,
        epistemic_covariance=epistemic_covariance,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
    )