"""Global-belief prediction for the structured differential-drive model.

This module adapts the 5D physics-structured dynamics model to the existing
full Gaussian GlobalBelief representation.

Local state:
    [p_x, p_y, phi, v, omega]

Action:
    [a_t, alpha_z]

The local models remain factorized across agents, while the global covariance
is propagated as a full matrix:

    P_next = F P F^T + Q

where F and Q are assembled from local block-diagonal terms.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from aorpo.agents.structured_diff_drive_dynamics import (
    StructuredDiffDriveStandardizerRS,
    predict_structured_next,
)
from aorpo.estimator.global_belief import (
    GlobalBelief,
    assemble_block_diagonal_covariance,
    assemble_block_diagonal_matrices,
    extract_agent_state,
    global_state_dim,
)
from aorpo.estimator.trigger import position_uncertainty_radius
from aorpo.estimator.uncertainty import symmetrize_covariance


LOCAL_STATE_DIM = 5
ACTION_DIM = 2


class StructuredGlobalPredictionInfo(NamedTuple):
    local_next_means: jnp.ndarray
    local_state_jacobians: jnp.ndarray
    local_process_covariances: jnp.ndarray
    local_aleatoric_covariances: jnp.ndarray
    local_epistemic_covariances: jnp.ndarray
    global_state_jacobian: jnp.ndarray
    global_process_covariance: jnp.ndarray


def _validate_global_prediction_inputs(
    belief: GlobalBelief,
    model_states: Sequence[TrainState],
    standardizers: Sequence[StructuredDiffDriveStandardizerRS],
    joint_actions: jnp.ndarray,
    num_agents: int,
) -> None:
    if len(model_states) != num_agents:
        raise ValueError(
            f"Expected {num_agents} model states, got {len(model_states)}."
        )
    if len(standardizers) != num_agents:
        raise ValueError(
            f"Expected {num_agents} standardizers, got "
            f"{len(standardizers)}."
        )
    if belief.mean.ndim != 2:
        raise ValueError("belief.mean must have shape (B, N*5).")
    if belief.covariance.ndim != 3:
        raise ValueError(
            "belief.covariance must have shape (B, N*5, N*5)."
        )

    batch_size = belief.mean.shape[0]
    expected_dim = global_state_dim(num_agents, LOCAL_STATE_DIM)

    if belief.mean.shape != (batch_size, expected_dim):
        raise ValueError(
            f"Expected belief.mean shape {(batch_size, expected_dim)}, "
            f"got {belief.mean.shape}."
        )
    if belief.covariance.shape != (
        batch_size,
        expected_dim,
        expected_dim,
    ):
        raise ValueError(
            "Global belief covariance has an unexpected shape: "
            f"{belief.covariance.shape}."
        )
    if joint_actions.shape != (
        batch_size,
        num_agents,
        ACTION_DIM,
    ):
        raise ValueError(
            "Expected joint_actions shape "
            f"{(batch_size, num_agents, ACTION_DIM)}, "
            f"got {joint_actions.shape}."
        )


def compute_structured_local_state_jacobian(
    train_state: TrainState,
    standardizer: StructuredDiffDriveStandardizerRS,
    local_state: jnp.ndarray,
    local_action: jnp.ndarray,
    dt: float,
    min_v: float,
    max_v: float,
    max_omega: float,
    prediction_mode: str,
    epistemic_process_scale: float,
) -> jnp.ndarray:
    """Compute F = d mean_next / d local_state, shape (B, 5, 5)."""

    def single_transition_mean(
        state_single: jnp.ndarray,
        action_single: jnp.ndarray,
    ) -> jnp.ndarray:
        next_mean, _ = predict_structured_next(
            train_state=train_state,
            standardizer=standardizer,
            local_state=state_single[None, :],
            local_action=action_single[None, :],
            dt=dt,
            min_v=min_v,
            max_v=max_v,
            max_omega=max_omega,
            prediction_mode=prediction_mode,
            epistemic_process_scale=epistemic_process_scale,
        )
        return next_mean[0]

    jacobian_fn = jax.jacfwd(single_transition_mean, argnums=0)
    return jax.vmap(jacobian_fn)(local_state, local_action)


def predict_structured_global_belief_oracle_actions(
    belief: GlobalBelief,
    model_states: Sequence[TrainState],
    standardizers: Sequence[StructuredDiffDriveStandardizerRS],
    joint_actions: jnp.ndarray,
    num_agents: int,
    dt: float,
    min_v: float,
    max_v: float,
    max_omega: float,
    prediction_mode: str = "ensemble_mean",
    epistemic_process_scale: float = 1.0,
    jitter: float = 1.0e-9,
) -> tuple[GlobalBelief, StructuredGlobalPredictionInfo]:
    """Predict the full Gaussian belief using known joint actions."""
    _validate_global_prediction_inputs(
        belief=belief,
        model_states=model_states,
        standardizers=standardizers,
        joint_actions=joint_actions,
        num_agents=num_agents,
    )

    local_next_means = []
    local_state_jacobians = []
    local_process_covariances = []
    local_aleatoric_covariances = []
    local_epistemic_covariances = []

    for agent_id in range(num_agents):
        local_state = extract_agent_state(
            belief=belief,
            agent_id=agent_id,
            num_agents=num_agents,
            local_state_dim=LOCAL_STATE_DIM,
        )
        local_action = joint_actions[:, agent_id, :]

        next_mean, covariance_info = predict_structured_next(
            train_state=model_states[agent_id],
            standardizer=standardizers[agent_id],
            local_state=local_state,
            local_action=local_action,
            dt=dt,
            min_v=min_v,
            max_v=max_v,
            max_omega=max_omega,
            prediction_mode=prediction_mode,
            epistemic_process_scale=epistemic_process_scale,
        )

        state_jacobian = compute_structured_local_state_jacobian(
            train_state=model_states[agent_id],
            standardizer=standardizers[agent_id],
            local_state=local_state,
            local_action=local_action,
            dt=dt,
            min_v=min_v,
            max_v=max_v,
            max_omega=max_omega,
            prediction_mode=prediction_mode,
            epistemic_process_scale=epistemic_process_scale,
        )

        local_next_means.append(next_mean)
        local_state_jacobians.append(state_jacobian)
        local_process_covariances.append(
            covariance_info["process_covariance"]
        )
        local_aleatoric_covariances.append(
            covariance_info["aleatoric_covariance"]
        )
        local_epistemic_covariances.append(
            covariance_info["epistemic_covariance"]
        )

    local_next_means_array = jnp.stack(
        local_next_means,
        axis=1,
    )
    local_state_jacobians_array = jnp.stack(
        local_state_jacobians,
        axis=1,
    )
    local_process_covariances_array = jnp.stack(
        local_process_covariances,
        axis=1,
    )
    local_aleatoric_covariances_array = jnp.stack(
        local_aleatoric_covariances,
        axis=1,
    )
    local_epistemic_covariances_array = jnp.stack(
        local_epistemic_covariances,
        axis=1,
    )

    global_state_jacobian = assemble_block_diagonal_matrices(
        local_state_jacobians_array
    )
    global_process_covariance = assemble_block_diagonal_covariance(
        local_process_covariances_array
    )

    predicted_covariance = (
        global_state_jacobian
        @ belief.covariance
        @ jnp.swapaxes(global_state_jacobian, -1, -2)
        + global_process_covariance
    )
    predicted_covariance = symmetrize_covariance(
        predicted_covariance,
        jitter=jitter,
    )

    predicted_belief = GlobalBelief(
        mean=local_next_means_array.reshape(
            belief.mean.shape[0],
            num_agents * LOCAL_STATE_DIM,
        ),
        covariance=predicted_covariance,
    )

    info = StructuredGlobalPredictionInfo(
        local_next_means=local_next_means_array,
        local_state_jacobians=local_state_jacobians_array,
        local_process_covariances=local_process_covariances_array,
        local_aleatoric_covariances=(
            local_aleatoric_covariances_array
        ),
        local_epistemic_covariances=(
            local_epistemic_covariances_array
        ),
        global_state_jacobian=global_state_jacobian,
        global_process_covariance=global_process_covariance,
    )
    return predicted_belief, info


def all_agent_position_uncertainty_radii(
    belief: GlobalBelief,
    num_agents: int,
    scale: float = 2.0,
) -> jnp.ndarray:
    """Return position-ellipse major radii for all agents, shape (B, N)."""
    radii = []

    for agent_id in range(num_agents):
        start = agent_id * LOCAL_STATE_DIM
        end = start + LOCAL_STATE_DIM
        covariance_block = belief.covariance[
            :,
            start:end,
            start:end,
        ]
        radii.append(
            position_uncertainty_radius(
                covariance=covariance_block,
                scale=scale,
            )
        )

    return jnp.stack(radii, axis=-1)
