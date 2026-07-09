# aorpo/estimator/uncertainty.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from aorpo.agents.independent_dynamics import (
    LocalStandardizerRS,
    predict_local_mean,
    predict_local_ensemble_next_gaussians,
)

from aorpo.estimator.infoprop import (
    infoprop_ensemble_prediction,
)

def covariance_trace_per_dim(covariance: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a covariance matrix into average per-dimension variance.

    Args:
        covariance: (..., D, D)

    Returns:
        (...,) equal to trace(P) / D.
    """
    state_dim = covariance.shape[-1]
    return jnp.trace(covariance, axis1=-2, axis2=-1) / float(state_dim)


def symmetrize_covariance(
    covariance: jnp.ndarray,
    jitter: float = 1e-9,
) -> jnp.ndarray:
    """
    Symmetrize covariance matrices and add a small diagonal jitter.
    """
    state_dim = covariance.shape[-1]
    identity = jnp.eye(state_dim, dtype=covariance.dtype)

    return (
        0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
        + jitter * identity
    )


def delta_denormalization_scale(
    standardizer: LocalStandardizerRS,
    state_dim: int,
) -> jnp.ndarray:
    """
    Return the elementwise scale from normalized delta coordinates
    to physical delta coordinates.
    """
    zeros = jnp.zeros((1, state_dim), dtype=jnp.float32)
    ones = jnp.ones((1, state_dim), dtype=jnp.float32)

    return (
        standardizer.denorm_delta(ones)
        - standardizer.denorm_delta(zeros)
    )[0]


def ensemble_process_covariance(
    ensemble_mu_norm: jnp.ndarray,
    ensemble_logvar_norm: jnp.ndarray,
    standardizer: LocalStandardizerRS,
    jitter: float = 1e-9,
) -> Dict[str, jnp.ndarray]:
    """
    Moment-match an ensemble of diagonal Gaussian predictions.

    Args:
        ensemble_mu_norm:     (K, B, D), normalized delta means.
        ensemble_logvar_norm: (K, B, D), normalized delta log-variances.
        standardizer:         normalizer for local dynamics.
        jitter:               diagonal stabilization term.

    Returns:
        Dictionary containing aleatoric, epistemic, and total covariance
        both in normalized delta coordinates and physical delta coordinates.
    """
    num_members = ensemble_mu_norm.shape[0]
    state_dim = ensemble_mu_norm.shape[-1]

    member_var_norm = jnp.exp(ensemble_logvar_norm)

    aleatoric_var_norm = jnp.mean(member_var_norm, axis=0)
    ensemble_mean_norm = jnp.mean(ensemble_mu_norm, axis=0)

    centered_means = ensemble_mu_norm - ensemble_mean_norm[None, ...]

    epistemic_cov_norm = (
        jnp.einsum("kbd,kbe->bde", centered_means, centered_means)
        / float(num_members)
    )

    aleatoric_cov_norm = jax.vmap(jnp.diag)(aleatoric_var_norm)
    total_cov_norm = aleatoric_cov_norm + epistemic_cov_norm

    scale = delta_denormalization_scale(
        standardizer=standardizer,
        state_dim=state_dim,
    )

    aleatoric_cov = jnp.einsum(
        "d,bde,e->bde",
        scale,
        aleatoric_cov_norm,
        scale,
    )

    epistemic_cov = jnp.einsum(
        "d,bde,e->bde",
        scale,
        epistemic_cov_norm,
        scale,
    )

    total_cov = jnp.einsum(
        "d,bde,e->bde",
        scale,
        total_cov_norm,
        scale,
    )

    total_cov = symmetrize_covariance(total_cov, jitter=jitter)

    return {
        "aleatoric_cov_norm": aleatoric_cov_norm,
        "epistemic_cov_norm": epistemic_cov_norm,
        "total_cov_norm": total_cov_norm,
        "aleatoric_cov": aleatoric_cov,
        "epistemic_cov": epistemic_cov,
        "total_cov": total_cov,
    }


def predict_local_mean_and_process_covariance(
    train_state: TrainState,
    standardizer: LocalStandardizerRS,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Predict local next-state mean and process covariance.

    The returned process covariance corresponds to Q_theta in physical
    state coordinates.
    """
    state_norm = standardizer.norm_local_state(local_state)
    action_norm = standardizer.norm_action(local_action)

    model_input = jnp.concatenate([state_norm, action_norm], axis=-1)

    ensemble_mu_norm, ensemble_logvar_norm = train_state.apply_fn(
        {"params": train_state.params},
        model_input,
    )

    mean_delta_norm = jnp.mean(ensemble_mu_norm, axis=0)
    mean_delta = standardizer.denorm_delta(mean_delta_norm)

    next_local_state = local_state + mean_delta

    covariance_info = ensemble_process_covariance(
        ensemble_mu_norm=ensemble_mu_norm,
        ensemble_logvar_norm=ensemble_logvar_norm,
        standardizer=standardizer,
    )

    return next_local_state, covariance_info["total_cov"], covariance_info

class LocalInfopropDynamicsPrediction(NamedTuple):
    """
    Infoprop-style local dynamics prediction in original coordinates.

    next_mean:
        CI-fused estimated environment mean, shape (B, D).

    process_covariance:
        Process covariance used for belief prediction, shape (B, D, D).

    ci_fused_covariance:
        CI-fused aleatoric covariance, shape (B, D, D).

    epistemic_covariance:
        Ensemble mean-spread covariance, shape (B, D, D).

    information_loss_covariance:
        Infoprop posterior covariance, shape (B, D, D).
        This is useful as an information-loss diagnostic, but it should not
        be directly used as the process covariance in the Kalman belief
        prediction.
    """

    next_mean: jnp.ndarray
    process_covariance: jnp.ndarray
    ci_fused_covariance: jnp.ndarray
    epistemic_covariance: jnp.ndarray
    information_loss_covariance: jnp.ndarray


def predict_local_infoprop_mean_and_process_covariance(
    train_state,
    standardizer,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
    jitter: float = 1.0e-6,
    epistemic_process_scale: float = 0.0,
) -> LocalInfopropDynamicsPrediction:
    """
    Predict local next-state mean and covariance using Infoprop-style
    ensemble fusion.

    Args:
        train_state:
            TrainState containing the local ensemble dynamics model.

        standardizer:
            Local dynamics standardizer.

        local_state:
            Local physical state with shape (B, 4).

        local_action:
            Local action with shape (B, A).

        jitter:
            Numerical stabilization for covariance inverses.

        epistemic_process_scale:
            Optional scale for adding epistemic covariance back into the
            process covariance. The default 0.0 follows the Infoprop idea of
            removing epistemic sampling noise from the propagated environment
            distribution.

    Returns:
        LocalInfopropDynamicsPrediction.
    """
    if epistemic_process_scale < 0.0:
        raise ValueError(
            "epistemic_process_scale must be non-negative, "
            f"got {epistemic_process_scale}."
        )

    raw_prediction = predict_local_ensemble_next_gaussians(
        train_state=train_state,
        standardizer=standardizer,
        local_state=local_state,
        local_action=local_action,
    )

    infoprop_prediction = infoprop_ensemble_prediction(
        ensemble_means=raw_prediction.ensemble_next_means,
        ensemble_covariances=(
            raw_prediction.ensemble_next_covariances
        ),
        model_sample=None,
        jitter=jitter,
    )

    process_covariance = (
        infoprop_prediction.fused_covariance
        + epistemic_process_scale
        * infoprop_prediction.epistemic_covariance
    )

    process_covariance = symmetrize_covariance(
        process_covariance
    )

    return LocalInfopropDynamicsPrediction(
        next_mean=infoprop_prediction.fused_mean,
        process_covariance=process_covariance,
        ci_fused_covariance=(
            infoprop_prediction.fused_covariance
        ),
        epistemic_covariance=(
            infoprop_prediction.epistemic_covariance
        ),
        information_loss_covariance=(
            infoprop_prediction.posterior_covariance
        ),
    )


def compute_local_state_jacobian(
    train_state: TrainState,
    standardizer: LocalStandardizerRS,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute F = d f_theta(z, a) / d z.

    Args:
        local_state:  (B, D)
        local_action: (B, A)

    Returns:
        F: (B, D, D)
    """
    def single_transition_mean(
        state_single: jnp.ndarray,
        action_single: jnp.ndarray,
    ) -> jnp.ndarray:
        next_state = predict_local_mean(
            train_state=train_state,
            standardizer=standardizer,
            local_state=state_single[None, :],
            local_action=action_single[None, :],
        )
        return next_state[0]

    jacobian_fn = jax.jacfwd(single_transition_mean, argnums=0)

    return jax.vmap(jacobian_fn)(local_state, local_action)


def compute_local_action_jacobian(
    train_state: TrainState,
    standardizer: LocalStandardizerRS,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute G = d f_theta(z, a) / d a.

    This is only needed when action uncertainty is propagated.
    """
    def single_transition_mean(
        state_single: jnp.ndarray,
        action_single: jnp.ndarray,
    ) -> jnp.ndarray:
        next_state = predict_local_mean(
            train_state=train_state,
            standardizer=standardizer,
            local_state=state_single[None, :],
            local_action=action_single[None, :],
        )
        return next_state[0]

    jacobian_fn = jax.jacfwd(single_transition_mean, argnums=1)

    return jax.vmap(jacobian_fn)(local_state, local_action)


def propagate_local_covariance(
    state_covariance: jnp.ndarray,
    state_jacobian: jnp.ndarray,
    process_covariance: jnp.ndarray,
    action_jacobian: Optional[jnp.ndarray] = None,
    action_covariance: Optional[jnp.ndarray] = None,
    include_action_uncertainty: bool = False,
    jitter: float = 1e-9,
) -> jnp.ndarray:
    """
    Propagate local belief covariance.

    Without action uncertainty:
        P_next = F P F^T + Q

    With action uncertainty:
        P_next = F P F^T + G P_a G^T + Q

    Args:
        state_covariance:            (B, D, D)
        state_jacobian:              (B, D, D)
        process_covariance:          (B, D, D)
        action_jacobian:             (B, D, A), optional.
        action_covariance:           (B, A, A), optional.
        include_action_uncertainty:  whether to include G P_a G^T.
        jitter:                      diagonal stabilization term.

    Returns:
        next_state_covariance: (B, D, D)
    """
    if include_action_uncertainty:
        if action_jacobian is None:
            raise ValueError(
                "action_jacobian is required when include_action_uncertainty=True."
            )
        if action_covariance is None:
            raise ValueError(
                "action_covariance is required when include_action_uncertainty=True."
            )

    def propagate_one(
        F: jnp.ndarray,
        P: jnp.ndarray,
        Q: jnp.ndarray,
        G: Optional[jnp.ndarray],
        P_action: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        predicted_covariance = F @ P @ F.T + Q

        if include_action_uncertainty:
            predicted_covariance = (
                predicted_covariance
                + G @ P_action @ G.T
            )

        return symmetrize_covariance(
            predicted_covariance,
            jitter=jitter,
        )

    if include_action_uncertainty:
        return jax.vmap(propagate_one)(
            state_jacobian,
            state_covariance,
            process_covariance,
            action_jacobian,
            action_covariance,
        )

    dummy_action_jacobian = jnp.zeros(
        (
            state_covariance.shape[0],
            state_covariance.shape[-1],
            1,
        ),
        dtype=state_covariance.dtype,
    )

    dummy_action_covariance = jnp.zeros(
        (
            state_covariance.shape[0],
            1,
            1,
        ),
        dtype=state_covariance.dtype,
    )

    return jax.vmap(propagate_one)(
        state_jacobian,
        state_covariance,
        process_covariance,
        dummy_action_jacobian,
        dummy_action_covariance,
    )