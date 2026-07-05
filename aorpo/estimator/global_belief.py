# aorpo/estimator/global_belief.py
from __future__ import annotations

from typing import Dict, NamedTuple, Sequence, Tuple

import jax.numpy as jnp
from flax.training.train_state import TrainState

from aorpo.agents.independent_dynamics import LocalStandardizerRS
from aorpo.estimator.uncertainty import (
    compute_local_state_jacobian,
    predict_local_mean_and_process_covariance,
    symmetrize_covariance,
)


class GlobalBelief(NamedTuple):
    """
    Gaussian belief over the concatenated physical states of all agents.

    Attributes:
        mean:
            Global state mean with shape (B, N * D).

        covariance:
            Full global covariance with shape (B, N * D, N * D).

    Here, N is the number of agents and D is the local physical state
    dimension of one agent.
    """

    mean: jnp.ndarray
    covariance: jnp.ndarray


def global_state_dim(
    num_agents: int,
    local_state_dim: int,
) -> int:
    """
    Return the dimension of the concatenated physical global state.
    """
    if num_agents < 1:
        raise ValueError(
            f"num_agents must be at least 1, got {num_agents}."
        )

    if local_state_dim < 1:
        raise ValueError(
            "local_state_dim must be at least 1, "
            f"got {local_state_dim}."
        )

    return num_agents * local_state_dim


def agent_state_slice(
    agent_id: int,
    num_agents: int,
    local_state_dim: int,
) -> slice:
    """
    Return the global-state slice corresponding to one agent.
    """
    if not 0 <= agent_id < num_agents:
        raise ValueError(
            f"agent_id must be in [0, {num_agents - 1}], "
            f"got {agent_id}."
        )

    start = agent_id * local_state_dim
    end = start + local_state_dim

    return slice(start, end)


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
        dimension: Covariance dimension.
        variance: Shared diagonal variance.
        dtype: Output dtype.

    Returns:
        Covariance matrices with shape (B, D, D).
    """
    if batch_size < 1:
        raise ValueError(
            f"batch_size must be at least 1, got {batch_size}."
        )

    if dimension < 1:
        raise ValueError(
            f"dimension must be at least 1, got {dimension}."
        )

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


def initialize_global_belief(
    initial_local_states: jnp.ndarray,
    initial_variance: float,
) -> GlobalBelief:
    """
    Initialize a full global Gaussian belief.

    Args:
        initial_local_states:
            Physical local states with shape (B, N, D).

        initial_variance:
            Initial isotropic state variance. A small value corresponds
            to an exact or nearly exact initial synchronization.

    Returns:
        GlobalBelief with a flattened mean and full covariance.
    """
    if initial_local_states.ndim != 3:
        raise ValueError(
            "initial_local_states must have shape (B, N, D), "
            f"got {initial_local_states.shape}."
        )

    batch_size, num_agents, local_state_dim = (
        initial_local_states.shape
    )

    state_dim = global_state_dim(
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    initial_mean = initial_local_states.reshape(
        batch_size,
        state_dim,
    )

    initial_covariance = make_isotropic_covariance(
        batch_size=batch_size,
        dimension=state_dim,
        variance=initial_variance,
        dtype=initial_local_states.dtype,
    )

    return GlobalBelief(
        mean=initial_mean,
        covariance=initial_covariance,
    )


def extract_agent_state(
    belief: GlobalBelief,
    agent_id: int,
    num_agents: int,
    local_state_dim: int,
) -> jnp.ndarray:
    """
    Extract the mean local physical state of one agent.

    Returns:
        Local state mean with shape (B, D).
    """
    state_slice = agent_state_slice(
        agent_id=agent_id,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    return belief.mean[:, state_slice]


def extract_agent_covariance(
    belief: GlobalBelief,
    agent_id: int,
    num_agents: int,
    local_state_dim: int,
) -> jnp.ndarray:
    """
    Extract one agent's diagonal covariance block.

    Returns:
        Covariance block with shape (B, D, D).
    """
    state_slice = agent_state_slice(
        agent_id=agent_id,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    return belief.covariance[
        :,
        state_slice,
        state_slice,
    ]


def extract_agent_cross_covariance(
    belief: GlobalBelief,
    row_agent_id: int,
    column_agent_id: int,
    num_agents: int,
    local_state_dim: int,
) -> jnp.ndarray:
    """
    Extract a cross-covariance block between two agents.

    Returns:
        Cross-covariance with shape (B, D, D).
    """
    row_slice = agent_state_slice(
        agent_id=row_agent_id,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    column_slice = agent_state_slice(
        agent_id=column_agent_id,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    return belief.covariance[
        :,
        row_slice,
        column_slice,
    ]


def assemble_block_diagonal_covariance(
    local_covariances: jnp.ndarray,
) -> jnp.ndarray:
    """
    Assemble a block-diagonal global covariance from local blocks.

    This helper is useful for initialization and independent physical
    process noise. It should not be used to overwrite an already dense
    global covariance after closed-loop propagation.

    Args:
        local_covariances:
            Local covariance blocks with shape (B, N, D, D).

    Returns:
        Block-diagonal covariance with shape (B, N * D, N * D).
    """
    if local_covariances.ndim != 4:
        raise ValueError(
            "local_covariances must have shape (B, N, D, D), "
            f"got {local_covariances.shape}."
        )

    batch_size, num_agents, local_state_dim, second_dim = (
        local_covariances.shape
    )

    if local_state_dim != second_dim:
        raise ValueError(
            "Each local covariance block must be square, "
            f"got shape {local_covariances.shape}."
        )

    state_dim = global_state_dim(
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    global_covariance = jnp.zeros(
        (
            batch_size,
            state_dim,
            state_dim,
        ),
        dtype=local_covariances.dtype,
    )

    for agent_id in range(num_agents):
        state_slice = agent_state_slice(
            agent_id=agent_id,
            num_agents=num_agents,
            local_state_dim=local_state_dim,
        )

        global_covariance = global_covariance.at[
            :,
            state_slice,
            state_slice,
        ].set(
            local_covariances[:, agent_id]
        )

    return global_covariance

def assemble_block_diagonal_matrices(
    local_matrices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Assemble block-diagonal matrices from local square blocks.

    Args:
        local_matrices:
            Local square matrices with shape (B, N, D, D).

    Returns:
        Block-diagonal matrices with shape (B, N * D, N * D).
    """
    if local_matrices.ndim != 4:
        raise ValueError(
            "local_matrices must have shape (B, N, D, D), "
            f"got {local_matrices.shape}."
        )

    batch_size, num_agents, local_dim, second_dim = (
        local_matrices.shape
    )

    if local_dim != second_dim:
        raise ValueError(
            "Each local matrix must be square, "
            f"got {local_matrices.shape}."
        )

    total_dim = global_state_dim(
        num_agents=num_agents,
        local_state_dim=local_dim,
    )

    global_matrix = jnp.zeros(
        (batch_size, total_dim, total_dim),
        dtype=local_matrices.dtype,
    )

    for agent_id in range(num_agents):
        state_slice = agent_state_slice(
            agent_id=agent_id,
            num_agents=num_agents,
            local_state_dim=local_dim,
        )

        global_matrix = global_matrix.at[
            :,
            state_slice,
            state_slice,
        ].set(local_matrices[:, agent_id])

    return global_matrix


def make_agent_selection_matrix(
    agent_id: int,
    num_agents: int,
    local_state_dim: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """
    Construct H_j that selects one agent's local state from the global state.

    Returns:
        Selection matrix H_j with shape (D, N * D).
    """
    state_dim = global_state_dim(
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    state_slice = agent_state_slice(
        agent_id=agent_id,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    selection_matrix = jnp.zeros(
        (local_state_dim, state_dim),
        dtype=dtype,
    )

    return selection_matrix.at[:, state_slice].set(
        jnp.eye(
            local_state_dim,
            dtype=dtype,
        )
    )


def direct_agent_observation_update(
    predicted_belief: GlobalBelief,
    observed_agent_id: int,
    measurement: jnp.ndarray,
    measurement_covariance: jnp.ndarray,
    num_agents: int,
    local_state_dim: int,
    jitter: float = 1e-9,
) -> Tuple[GlobalBelief, Dict[str, jnp.ndarray]]:
    """
    Apply a direct local-state measurement update to a full global belief.

    The measurement model is:

        y_j = H_j s + v_j

    where H_j selects the physical local state of observed_agent_id.

    The Joseph covariance update preserves numerical symmetry and improves
    positive-semidefinite stability.

    Args:
        predicted_belief:
            Prior full global belief.

        observed_agent_id:
            Agent whose local physical state is observed or communicated.

        measurement:
            Direct local-state measurement with shape (B, D).

        measurement_covariance:
            Measurement covariance R_j with shape (B, D, D), or shape
            (D, D), which will be broadcast across the batch.

        num_agents:
            Number of physical agents.

        local_state_dim:
            Dimension of each physical local state.

        jitter:
            Small diagonal stabilization value.

    Returns:
        posterior_belief:
            Full global belief after the measurement correction.

        update_info:
            Innovation, innovation covariance, Kalman gain, and H matrix.
    """
    prior_mean = predicted_belief.mean
    prior_covariance = predicted_belief.covariance

    expected_state_dim = global_state_dim(
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    if prior_mean.ndim != 2:
        raise ValueError(
            "predicted_belief.mean must have shape (B, S), "
            f"got {prior_mean.shape}."
        )

    if prior_covariance.ndim != 3:
        raise ValueError(
            "predicted_belief.covariance must have shape (B, S, S), "
            f"got {prior_covariance.shape}."
        )

    if prior_mean.shape[-1] != expected_state_dim:
        raise ValueError(
            "Global belief mean dimension does not match num_agents and "
            f"local_state_dim. Expected {expected_state_dim}, "
            f"got {prior_mean.shape[-1]}."
        )

    if prior_covariance.shape[-2:] != (
        expected_state_dim,
        expected_state_dim,
    ):
        raise ValueError(
            "Global belief covariance shape does not match the expected "
            f"state dimension {expected_state_dim}. "
            f"Got {prior_covariance.shape}."
        )

    batch_size = prior_mean.shape[0]

    if measurement.shape != (batch_size, local_state_dim):
        raise ValueError(
            "measurement must have shape "
            f"({batch_size}, {local_state_dim}), "
            f"got {measurement.shape}."
        )

    if measurement_covariance.ndim == 2:
        measurement_covariance = jnp.broadcast_to(
            measurement_covariance,
            (
                batch_size,
                local_state_dim,
                local_state_dim,
            ),
        )

    expected_measurement_covariance_shape = (
        batch_size,
        local_state_dim,
        local_state_dim,
    )

    if measurement_covariance.shape != expected_measurement_covariance_shape:
        raise ValueError(
            "measurement_covariance must have shape "
            f"{expected_measurement_covariance_shape}, "
            f"got {measurement_covariance.shape}."
        )

    measurement_matrix = make_agent_selection_matrix(
        agent_id=observed_agent_id,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
        dtype=prior_mean.dtype,
    )

    predicted_measurement = jnp.einsum(
        "ds,bs->bd",
        measurement_matrix,
        prior_mean,
    )

    innovation = measurement - predicted_measurement

    projected_covariance = jnp.einsum(
        "ds,bsr,er->bde",
        measurement_matrix,
        prior_covariance,
        measurement_matrix,
    )

    innovation_covariance = symmetrize_covariance(
        projected_covariance + measurement_covariance,
        jitter=jitter,
    )

    cross_covariance = jnp.einsum(
        "bsr,dr->bsd",
        prior_covariance,
        measurement_matrix,
    )

    kalman_gain_transpose = jnp.linalg.solve(
        jnp.swapaxes(
            innovation_covariance,
            -1,
            -2,
        ),
        jnp.swapaxes(
            cross_covariance,
            -1,
            -2,
        ),
    )

    kalman_gain = jnp.swapaxes(
        kalman_gain_transpose,
        -1,
        -2,
    )

    posterior_mean = prior_mean + jnp.einsum(
        "bsd,bd->bs",
        kalman_gain,
        innovation,
    )

    identity = jnp.broadcast_to(
        jnp.eye(
            expected_state_dim,
            dtype=prior_covariance.dtype,
        ),
        prior_covariance.shape,
    )

    kalman_measurement_matrix = jnp.einsum(
        "bsd,dr->bsr",
        kalman_gain,
        measurement_matrix,
    )

    residual_transform = identity - kalman_measurement_matrix

    posterior_covariance = (
        residual_transform
        @ prior_covariance
        @ jnp.swapaxes(
            residual_transform,
            -1,
            -2,
        )
        + kalman_gain
        @ measurement_covariance
        @ jnp.swapaxes(
            kalman_gain,
            -1,
            -2,
        )
    )

    posterior_covariance = symmetrize_covariance(
        posterior_covariance,
        jitter=jitter,
    )

    posterior_belief = GlobalBelief(
        mean=posterior_mean,
        covariance=posterior_covariance,
    )

    update_info = {
        "measurement_matrix": measurement_matrix,
        "innovation": innovation,
        "innovation_covariance": innovation_covariance,
        "kalman_gain": kalman_gain,
    }

    return posterior_belief, update_info

def predict_global_belief_oracle_actions(
    belief: GlobalBelief,
    model_states: Sequence[TrainState],
    standardizers: Sequence[LocalStandardizerRS],
    joint_actions: jnp.ndarray,
    num_agents: int,
    local_state_dim: int,
) -> Tuple[GlobalBelief, Dict[str, jnp.ndarray]]:
    """
    Predict a full global belief using independent local dynamics models.

    The physical dynamics are factorized across agents, while the global
    covariance remains full. This function uses known oracle joint actions.

    Args:
        belief:
            Current full global Gaussian belief.

        model_states:
            One trained local dynamics model state per agent.

        standardizers:
            One local dynamics standardizer per agent.

        joint_actions:
            Known actions with shape (B, N, A).

        num_agents:
            Number of agents.

        local_state_dim:
            Physical state dimension of one agent.

    Returns:
        predicted_belief:
            Predicted global belief after one dynamics step.

        prediction_info:
            Diagnostics containing local and global Jacobians and
            process covariance matrices.
    """
    if len(model_states) != num_agents:
        raise ValueError(
            "model_states must contain one model per agent. "
            f"Expected {num_agents}, got {len(model_states)}."
        )

    if len(standardizers) != num_agents:
        raise ValueError(
            "standardizers must contain one standardizer per agent. "
            f"Expected {num_agents}, got {len(standardizers)}."
        )

    if belief.mean.ndim != 2:
        raise ValueError(
            "belief.mean must have shape (B, S), "
            f"got {belief.mean.shape}."
        )

    if belief.covariance.ndim != 3:
        raise ValueError(
            "belief.covariance must have shape (B, S, S), "
            f"got {belief.covariance.shape}."
        )

    batch_size = belief.mean.shape[0]

    state_dim = global_state_dim(
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )

    if belief.mean.shape != (batch_size, state_dim):
        raise ValueError(
            "belief.mean has an unexpected shape. "
            f"Expected ({batch_size}, {state_dim}), "
            f"got {belief.mean.shape}."
        )

    if belief.covariance.shape != (
        batch_size,
        state_dim,
        state_dim,
    ):
        raise ValueError(
            "belief.covariance has an unexpected shape. "
            f"Expected ({batch_size}, {state_dim}, {state_dim}), "
            f"got {belief.covariance.shape}."
        )

    if joint_actions.ndim != 3:
        raise ValueError(
            "joint_actions must have shape (B, N, A), "
            f"got {joint_actions.shape}."
        )

    if joint_actions.shape[0] != batch_size:
        raise ValueError(
            "joint_actions batch size must match belief batch size. "
            f"Got {joint_actions.shape[0]} and {batch_size}."
        )

    if joint_actions.shape[1] != num_agents:
        raise ValueError(
            "joint_actions agent dimension must match num_agents. "
            f"Expected {num_agents}, got {joint_actions.shape[1]}."
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
            local_state_dim=local_state_dim,
        )

        local_action = joint_actions[:, agent_id, :]

        next_local_state, process_covariance, covariance_info = (
            predict_local_mean_and_process_covariance(
                train_state=model_states[agent_id],
                standardizer=standardizers[agent_id],
                local_state=local_state,
                local_action=local_action,
            )
        )

        state_jacobian = compute_local_state_jacobian(
            train_state=model_states[agent_id],
            standardizer=standardizers[agent_id],
            local_state=local_state,
            local_action=local_action,
        )

        local_next_means.append(next_local_state)
        local_state_jacobians.append(state_jacobian)
        local_process_covariances.append(process_covariance)

        local_aleatoric_covariances.append(
            covariance_info["aleatoric_cov"]
        )

        local_epistemic_covariances.append(
            covariance_info["epistemic_cov"]
        )

    predicted_mean = jnp.concatenate(
        local_next_means,
        axis=-1,
    )

    local_state_jacobians = jnp.stack(
        local_state_jacobians,
        axis=1,
    )

    local_process_covariances = jnp.stack(
        local_process_covariances,
        axis=1,
    )

    local_aleatoric_covariances = jnp.stack(
        local_aleatoric_covariances,
        axis=1,
    )

    local_epistemic_covariances = jnp.stack(
        local_epistemic_covariances,
        axis=1,
    )

    global_state_jacobian = assemble_block_diagonal_matrices(
        local_state_jacobians
    )

    global_process_covariance = assemble_block_diagonal_matrices(
        local_process_covariances
    )

    predicted_covariance = (
        global_state_jacobian
        @ belief.covariance
        @ jnp.swapaxes(global_state_jacobian, -1, -2)
        + global_process_covariance
    )

    predicted_covariance = symmetrize_covariance(
        predicted_covariance
    )

    predicted_belief = GlobalBelief(
        mean=predicted_mean,
        covariance=predicted_covariance,
    )

    prediction_info = {
        "global_state_jacobian": global_state_jacobian,
        "global_process_covariance": global_process_covariance,
        "local_state_jacobians": local_state_jacobians,
        "local_process_covariances": local_process_covariances,
        "local_aleatoric_covariances": local_aleatoric_covariances,
        "local_epistemic_covariances": local_epistemic_covariances,
    }

    return predicted_belief, prediction_info