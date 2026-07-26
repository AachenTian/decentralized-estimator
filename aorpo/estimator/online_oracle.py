# aorpo/estimator/online_oracle.py
from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra.utils import to_absolute_path

from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from omegaconf import DictConfig

from aorpo.agents.independent_dynamics import LocalStandardizerRS
from aorpo.envs.jaxmarl_simple_spread_v3_env_wrapper import make_mpe_env
from aorpo.estimator.global_belief import (
    GlobalBelief,
    direct_agent_observation_update,
    extract_agent_covariance,
    extract_agent_state,
    initialize_global_belief,
    make_isotropic_covariance,
    predict_global_belief_infoprop_oracle_actions,
    predict_global_belief_oracle_actions,
)
from aorpo.estimator.uncertainty import covariance_trace_per_dim

from aorpo.estimator.trigger import (
    position_radius_threshold_trigger,
    trace_threshold_trigger,
    position_mean_error_score,
    sender_position_trigger,
)

def extract_true_local_physical_states(
    env_state: Any,
    num_agents: int,
) -> jnp.ndarray:
    """
    Extract physical local states [p_x, p_y, v_x, v_y] for all agents.

    Returns:
        Array with shape (N, 4).
    """
    positions = jnp.asarray(env_state.p_pos[:num_agents])
    velocities = jnp.asarray(env_state.p_vel[:num_agents])

    return jnp.concatenate(
        [positions, velocities],
        axis=-1,
    )


def build_action_dict(
    env: Any,
    joint_actions: jnp.ndarray,
    num_agents: int,
) -> Dict[str, jnp.ndarray]:
    """
    Convert a joint action array into the action dictionary expected by JaxMARL.
    """
    if joint_actions.shape[0] != num_agents:
        raise ValueError(
            "joint_actions has an unexpected agent dimension. "
            f"Expected {num_agents}, got {joint_actions.shape[0]}."
        )

    if len(env.agents) != num_agents:
        raise ValueError(
            "Environment agent count does not match estimator configuration. "
            f"Expected {num_agents}, got {len(env.agents)}."
        )

    return {
        agent_name: joint_actions[agent_id]
        for agent_id, agent_name in enumerate(env.agents)
    }


def should_scheduled_communication(
    communication_mode: str,
    step_index: int,
    periodic_interval: int,
) -> bool:
    """
    Decide whether scheduled communication occurs at the current step.

    Supported scheduled modes:
        initial_sync_only
        always_communicate
        periodic
    """
    if communication_mode == "initial_sync_only":
        return False

    if communication_mode == "always_communicate":
        return True

    if communication_mode == "periodic":
        if periodic_interval < 1:
            raise ValueError(
                "periodic_interval must be at least 1, "
                f"got {periodic_interval}."
            )

        return (step_index + 1) % periodic_interval == 0

    raise ValueError(
        "Unsupported scheduled communication mode: "
        f"{communication_mode}."
    )


def decide_remote_communications(
    communication_mode: str,
    belief_before_remote_messages: GlobalBelief,
    remote_agent_ids: Sequence[int],
    step_index: int,
    periodic_interval: int,
    event_trigger_threshold: float,
    position_trigger_radius: float,
    position_trigger_scale: float,
    num_agents: int,
    local_state_dim: int,
) -> Dict[int, bool]:
    """
    Decide communication independently for each remote agent.

    Event-trigger decisions are evaluated using the same belief before any
    remote measurement update is applied. This makes the trigger decisions
    independent of the sequential update order.
    """
    scheduled_modes = {
        "initial_sync_only",
        "always_communicate",
        "periodic",
    }

    if communication_mode in scheduled_modes:
        scheduled_decision = should_scheduled_communication(
            communication_mode=communication_mode,
            step_index=step_index,
            periodic_interval=periodic_interval,
        )

        return {
            remote_agent_id: scheduled_decision
            for remote_agent_id in remote_agent_ids
        }

    if communication_mode == "event_triggered":
        decisions: Dict[int, bool] = {}

        for remote_agent_id in remote_agent_ids:
            remote_covariance = extract_agent_covariance(
                belief=belief_before_remote_messages,
                agent_id=remote_agent_id,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

            decisions[remote_agent_id] = bool(
                trace_threshold_trigger(
                    covariance=remote_covariance,
                    threshold=event_trigger_threshold,
                )[0]
            )

        return decisions

    if communication_mode == "position_event_triggered":
        decisions: Dict[int, bool] = {}

        for remote_agent_id in remote_agent_ids:
            remote_covariance = extract_agent_covariance(
                belief=belief_before_remote_messages,
                agent_id=remote_agent_id,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

            decisions[remote_agent_id] = bool(
                position_radius_threshold_trigger(
                    covariance=remote_covariance,
                    radius_threshold=position_trigger_radius,
                    scale=position_trigger_scale,
                )[0]
            )

        return decisions

    raise ValueError(
        "Unsupported communication_mode: "
        f"{communication_mode}. "
        "Expected one of: initial_sync_only, always_communicate, "
        "periodic, event_triggered, position_event_triggered."
    )

def decide_sender_broadcasts(
    predicted_common_belief: GlobalBelief,
    true_local_states: jnp.ndarray,
    sender_agent_ids: Sequence[int],
    sender_error_threshold: float,
    sender_covariance_radius_threshold: float,
    sender_covariance_scale: float,
    num_agents: int,
    local_state_dim: int,
) -> Dict[int, bool]:
    """
    Decide which agents broadcast their private local observations.

    This is sender-triggered:
        each agent checks the common belief block of itself against its
        private observed local state.
    """
    decisions: Dict[int, bool] = {}

    for sender_agent_id in sender_agent_ids:
        predicted_local_mean = extract_agent_state(
            belief=predicted_common_belief,
            agent_id=sender_agent_id,
            num_agents=num_agents,
            local_state_dim=local_state_dim,
        )

        predicted_local_covariance = extract_agent_covariance(
            belief=predicted_common_belief,
            agent_id=sender_agent_id,
            num_agents=num_agents,
            local_state_dim=local_state_dim,
        )

        observed_local_state = true_local_states[
            sender_agent_id
        ][None, :]

        should_broadcast = sender_position_trigger(
            predicted_local_mean=predicted_local_mean,
            observed_local_state=observed_local_state,
            predicted_local_covariance=predicted_local_covariance,
            error_threshold=sender_error_threshold,
            covariance_radius_threshold=(
                sender_covariance_radius_threshold
            ),
            covariance_scale=sender_covariance_scale,
        )

        decisions[sender_agent_id] = bool(
            should_broadcast[0]
        )

    return decisions


def mean_or_nan(values: list[float]) -> float:
    """
    Return the mean of a list, or NaN when the list is empty.
    """
    if not values:
        return float("nan")

    return float(jnp.mean(jnp.asarray(values)))

def belief_local_state_means(
    belief: GlobalBelief,
    num_agents: int,
    local_state_dim: int,
) -> jnp.ndarray:
    """
    Reshape a global belief mean into per-agent local state means.

    Returns:
        Array with shape (B, N, D).
    """
    return belief.mean.reshape(
        belief.mean.shape[0],
        num_agents,
        local_state_dim,
    )


def all_agent_covariance_proxies(
    belief: GlobalBelief,
    num_agents: int,
    local_state_dim: int,
) -> jnp.ndarray:
    """
    Compute trace(P_jj) / D for every agent covariance block.

    Returns:
        Array with shape (N,).
    """
    scores = []

    for agent_id in range(num_agents):
        local_covariance = extract_agent_covariance(
            belief=belief,
            agent_id=agent_id,
            num_agents=num_agents,
            local_state_dim=local_state_dim,
        )

        scores.append(
            covariance_trace_per_dim(local_covariance)[0]
        )

    return jnp.stack(scores, axis=0)


def all_agent_state_mse(
    belief: GlobalBelief,
    true_local_states: jnp.ndarray,
    num_agents: int,
    local_state_dim: int,
) -> jnp.ndarray:
    """
    Compute one per-agent physical-state MSE.

    Args:
        belief:
            Full global belief with batch size one.

        true_local_states:
            True local physical states with shape (N, D).

    Returns:
        Per-agent MSE with shape (N,).
    """
    estimated_local_states = belief_local_state_means(
        belief=belief,
        num_agents=num_agents,
        local_state_dim=local_state_dim,
    )[0]

    return jnp.mean(
        (estimated_local_states - true_local_states) ** 2,
        axis=-1,
    )


def build_trace_path(
    trace_directory: str,
    communication_mode: str,
    position_trigger_radius: float,
    position_trigger_scale: float,
    periodic_interval: int,
    event_trigger_threshold: float,
    ego_agent_id: int,
    episode_index: int,
) -> Path:
    """
    Construct an output path for one saved online estimator trace.
    """
    output_directory = Path(
        to_absolute_path(trace_directory)
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if communication_mode == "periodic":
        mode_tag = f"periodic_k{periodic_interval}"
    elif communication_mode == "event_triggered":
        threshold_tag = (
            f"{event_trigger_threshold:.4f}"
            .replace(".", "p")
        )
        mode_tag = f"event_tau{threshold_tag}"
    elif communication_mode == "position_event_triggered":
        radius_tag = (
            f"{position_trigger_radius:.4f}"
            .replace(".", "p")
        )
        scale_tag = (
            f"{position_trigger_scale:.1f}"
            .replace(".", "p")
        )
        mode_tag = f"pos_event_r{radius_tag}_s{scale_tag}"
    else:
        mode_tag = communication_mode

    filename = (
        f"{mode_tag}_ego{ego_agent_id}_"
        f"episode{episode_index + 1:03d}.npz"
    )

    return output_directory / filename


def run_online_oracle_global_belief_evaluation(
    model_states: Sequence[TrainState],
    standardizers: Sequence[LocalStandardizerRS],
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Evaluate a full global belief in the real environment using oracle actions.

    The ego agent maintains a full global Gaussian belief. Its own local state
    is measured at every time step. Remote local states are updated according
    to the configured communication baseline.

    This evaluation intentionally uses the real joint action applied to the
    environment. It isolates state-estimation and communication behavior before
    remote stochastic policy inference is introduced.
    """
    num_agents = cfg.train.num_opponents + 1
    local_state_dim = 4
    action_dim = cfg.env.act_dim

    ego_agent_id = int(cfg.online_estimator.ego_agent_id)
    num_episodes = int(cfg.online_estimator.num_episodes)
    max_steps_per_episode = int(
        cfg.online_estimator.max_steps_per_episode
    )

    communication_mode = str(
        cfg.online_estimator.communication_mode
    )

    dynamics_prediction_mode = str(
        cfg.online_estimator.dynamics_prediction_mode
    )

    epistemic_process_scale = float(
        cfg.online_estimator.epistemic_process_scale
    )

    periodic_interval = int(
        cfg.online_estimator.periodic_interval
    )

    event_trigger_threshold = float(
        cfg.online_estimator.event_trigger_threshold
    )

    position_trigger_radius = float(
        cfg.online_estimator.position_trigger_radius
    )

    position_trigger_scale = float(
        cfg.online_estimator.position_trigger_scale
    )

    initial_variance = float(
        cfg.online_estimator.initial_variance
    )

    measurement_variance = float(
        cfg.online_estimator.measurement_variance
    )

    action_min = float(cfg.online_estimator.action_min)
    action_max = float(cfg.online_estimator.action_max)

    trace_enabled = bool(
        cfg.online_estimator.trace_enabled
    )

    trace_episode_index = int(
        cfg.online_estimator.trace_episode_index
    )

    trace_directory = str(
        cfg.online_estimator.trace_directory
    )

    sender_error_threshold = float(
        cfg.online_estimator.sender_error_threshold
    )

    sender_covariance_radius_threshold = float(
        cfg.online_estimator.sender_covariance_radius_threshold
    )

    sender_covariance_scale = float(
        cfg.online_estimator.sender_covariance_scale
    )

    if not 0 <= ego_agent_id < num_agents:
        raise ValueError(
            f"ego_agent_id must be in [0, {num_agents - 1}], "
            f"got {ego_agent_id}."
        )

    if num_episodes < 1:
        raise ValueError(
            f"num_episodes must be at least 1, got {num_episodes}."
        )

    if max_steps_per_episode < 1:
        raise ValueError(
            "max_steps_per_episode must be at least 1, "
            f"got {max_steps_per_episode}."
        )

    if action_min >= action_max:
        raise ValueError(
            "action_min must be smaller than action_max. "
            f"Received {action_min} and {action_max}."
        )

    if event_trigger_threshold < 0.0:
        raise ValueError(
            "event_trigger_threshold must be non-negative, "
            f"got {event_trigger_threshold}."
        )

    if position_trigger_radius < 0.0:
        raise ValueError(
            "position_trigger_radius must be non-negative, "
            f"got {position_trigger_radius}."
        )

    if position_trigger_scale <= 0.0:
        raise ValueError(
            "position_trigger_scale must be positive, "
            f"got {position_trigger_scale}."
        )

    if trace_enabled and not 0 <= trace_episode_index < num_episodes:
        raise ValueError(
            "trace_episode_index must refer to an episode in the evaluation. "
            f"Received {trace_episode_index} with "
            f"num_episodes={num_episodes}."
        )

    valid_dynamics_prediction_modes = {
        "standard",
        "infoprop_ci",
    }

    if dynamics_prediction_mode not in valid_dynamics_prediction_modes:
        raise ValueError(
            "Unsupported dynamics_prediction_mode: "
            f"{dynamics_prediction_mode}. "
            f"Expected one of {sorted(valid_dynamics_prediction_modes)}."
        )

    if epistemic_process_scale < 0.0:
        raise ValueError(
            "epistemic_process_scale must be non-negative, "
            f"got {epistemic_process_scale}."
        )

    if sender_error_threshold < 0.0:
        raise ValueError(
            "sender_error_threshold must be non-negative, "
            f"got {sender_error_threshold}."
        )

    if sender_covariance_radius_threshold < 0.0:
        raise ValueError(
            "sender_covariance_radius_threshold must be non-negative, "
            f"got {sender_covariance_radius_threshold}."
        )

    if sender_covariance_scale <= 0.0:
        raise ValueError(
            "sender_covariance_scale must be positive, "
            f"got {sender_covariance_scale}."
        )

    env = make_mpe_env(cfg)

    rng = jax.random.PRNGKey(
        int(cfg.seed) + 12345
    )

    remote_agent_ids = [
        agent_id
        for agent_id in range(num_agents)
        if agent_id != ego_agent_id
    ]

    sender_trigger_modes = {
        "sender_position_triggered",
    }

    sender_agent_ids = list(range(num_agents))

    communicating_agent_ids = (
        sender_agent_ids
        if communication_mode in sender_trigger_modes
        else remote_agent_ids
    )

    pre_update_remote_mse_values: list[float] = []
    post_update_remote_mse_values: list[float] = []

    pre_update_covariance_proxy_values: list[float] = []
    post_update_covariance_proxy_values: list[float] = []

    total_remote_messages = 0
    total_environment_steps = 0

    remote_message_counts = {
        agent_id: 0
        for agent_id in communicating_agent_ids
    }

    print(
        "\n===== Online Global Belief Evaluation "
        "with Oracle Actions ====="
    )

    print(f"Ego receiver agent: {ego_agent_id}")
    print(f"Communication mode: {communication_mode}")
    print(f"Number of episodes: {num_episodes}")

    print(f"Dynamics prediction mode: {dynamics_prediction_mode}")

    if dynamics_prediction_mode == "infoprop_ci":
        print(
            "Epistemic process scale: "
            f"{epistemic_process_scale:.6f}"
        )

    if trace_enabled:
        print(
            "Trace logging enabled for episode index: "
            f"{trace_episode_index}"
        )

    if communication_mode == "event_triggered":
        print(
            "Event trigger threshold "
            f"(trace(P_jj) / local_state_dim): "
            f"{event_trigger_threshold:.6f}"
        )

    if communication_mode == "position_event_triggered":
        print(
            "Position trigger radius "
            f"({position_trigger_scale:.1f} sigma): "
            f"{position_trigger_radius:.6f}"
        )

    if communication_mode == "sender_position_triggered":
        print(
            "Sender position trigger: "
            f"error_threshold={sender_error_threshold:.6f}, "
            "covariance_radius_threshold="
            f"{sender_covariance_radius_threshold:.6f}, "
            f"scale={sender_covariance_scale:.1f}"
        )

    for episode_index in range(num_episodes):
        rng, reset_key = jax.random.split(rng)

        _, env_state = env.reset(reset_key)

        initial_local_states = extract_true_local_physical_states(
            env_state=env_state,
            num_agents=num_agents,
        )

        belief = initialize_global_belief(
            initial_local_states=initial_local_states[None, ...],
            initial_variance=initial_variance,
        )

        trace_this_episode = (
            trace_enabled
            and episode_index == trace_episode_index
        )

        if trace_this_episode:
            trace_steps = {
                "true_local_states": [],
                "landmark_positions": [],
                "joint_actions": [],
                "predicted_local_means": [],
                "after_ego_local_means": [],
                "posterior_local_means": [],
                "predicted_global_covariances": [],
                "after_ego_global_covariances": [],
                "posterior_global_covariances": [],
                "predicted_covariance_proxy": [],
                "after_ego_covariance_proxy": [],
                "posterior_covariance_proxy": [],
                "predicted_state_mse": [],
                "after_ego_state_mse": [],
                "posterior_state_mse": [],
                "communication_decisions": [],
            }

            trace_initial_local_states = np.asarray(
                initial_local_states
            )

            trace_initial_mean = np.asarray(
                belief.mean[0]
            )

            trace_initial_covariance = np.asarray(
                belief.covariance[0]
            )

            trace_initial_landmark_positions = np.asarray(
                env_state.p_pos[num_agents:]
            )

        episode_message_count = 0
        episode_post_update_mse_values: list[float] = []

        for step_index in range(max_steps_per_episode):
            rng, action_key, environment_key = jax.random.split(
                rng,
                3,
            )

            joint_actions = jax.random.uniform(
                action_key,
                shape=(num_agents, action_dim),
                minval=action_min,
                maxval=action_max,
                dtype=jnp.float32,
            )

            if dynamics_prediction_mode == "standard":
                predicted_belief, _ = predict_global_belief_oracle_actions(
                    belief=belief,
                    model_states=model_states,
                    standardizers=standardizers,
                    joint_actions=joint_actions[None, ...],
                    num_agents=num_agents,
                    local_state_dim=local_state_dim,
                )
            elif dynamics_prediction_mode == "infoprop_ci":
                predicted_belief = predict_global_belief_infoprop_oracle_actions(
                    belief=belief,
                    model_states=model_states,
                    standardizers=standardizers,
                    joint_actions=joint_actions[None, ...],
                    num_agents=num_agents,
                    local_state_dim=local_state_dim,
                    epistemic_process_scale=epistemic_process_scale,
                )
            else:
                raise RuntimeError(
                    "Unexpected dynamics_prediction_mode after validation: "
                    f"{dynamics_prediction_mode}."
                )

            action_dict = build_action_dict(
                env=env,
                joint_actions=joint_actions,
                num_agents=num_agents,
            )

            _, next_env_state, _, dones, _ = env.step(
                environment_key,
                env_state,
                action_dict,
            )

            true_next_local_states = extract_true_local_physical_states(
                env_state=next_env_state,
                num_agents=num_agents,
            )

            measurement_covariance = make_isotropic_covariance(
                batch_size=1,
                dimension=local_state_dim,
                variance=measurement_variance,
                dtype=true_next_local_states.dtype,
            )

            if communication_mode == "sender_position_triggered":
                pre_communication_belief = predicted_belief
                belief_after_ego = predicted_belief

                for agent_id in communicating_agent_ids:
                    estimated_state = extract_agent_state(
                        belief=pre_communication_belief,
                        agent_id=agent_id,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    true_state = true_next_local_states[
                        agent_id
                    ][None, ...]

                    state_mse = jnp.mean(
                        (estimated_state - true_state) ** 2
                    )

                    local_covariance = extract_agent_covariance(
                        belief=pre_communication_belief,
                        agent_id=agent_id,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    covariance_proxy = covariance_trace_per_dim(
                        local_covariance
                    )[0]

                    pre_update_remote_mse_values.append(
                        float(state_mse)
                    )

                    pre_update_covariance_proxy_values.append(
                        float(covariance_proxy)
                    )

                communication_decisions = decide_sender_broadcasts(
                    predicted_common_belief=pre_communication_belief,
                    true_local_states=true_next_local_states,
                    sender_agent_ids=sender_agent_ids,
                    sender_error_threshold=sender_error_threshold,
                    sender_covariance_radius_threshold=(
                        sender_covariance_radius_threshold
                    ),
                    sender_covariance_scale=sender_covariance_scale,
                    num_agents=num_agents,
                    local_state_dim=local_state_dim,
                )

                belief = pre_communication_belief

                for sender_agent_id in sender_agent_ids:
                    if not communication_decisions[sender_agent_id]:
                        continue

                    belief, _ = direct_agent_observation_update(
                        predicted_belief=belief,
                        observed_agent_id=sender_agent_id,
                        measurement=true_next_local_states[
                            sender_agent_id
                        ][None, ...],
                        measurement_covariance=measurement_covariance,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    total_remote_messages += 1
                    remote_message_counts[sender_agent_id] += 1
                    episode_message_count += 1

                for agent_id in communicating_agent_ids:
                    estimated_state = extract_agent_state(
                        belief=belief,
                        agent_id=agent_id,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    true_state = true_next_local_states[
                        agent_id
                    ][None, ...]

                    state_mse = jnp.mean(
                        (estimated_state - true_state) ** 2
                    )

                    local_covariance = extract_agent_covariance(
                        belief=belief,
                        agent_id=agent_id,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    covariance_proxy = covariance_trace_per_dim(
                        local_covariance
                    )[0]

                    post_update_remote_mse_values.append(
                        float(state_mse)
                    )

                    post_update_covariance_proxy_values.append(
                        float(covariance_proxy)
                    )

                    episode_post_update_mse_values.append(
                        float(state_mse)
                    )

            else:
                belief_after_ego, _ = direct_agent_observation_update(
                    predicted_belief=predicted_belief,
                    observed_agent_id=ego_agent_id,
                    measurement=true_next_local_states[
                        ego_agent_id
                    ][None, ...],
                    measurement_covariance=measurement_covariance,
                    num_agents=num_agents,
                    local_state_dim=local_state_dim,
                )

                pre_communication_belief = belief_after_ego
                belief = belief_after_ego

                for remote_agent_id in remote_agent_ids:
                    estimated_remote_state = extract_agent_state(
                        belief=pre_communication_belief,
                        agent_id=remote_agent_id,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    true_remote_state = true_next_local_states[
                        remote_agent_id
                    ][None, ...]

                    remote_mse = jnp.mean(
                        (estimated_remote_state - true_remote_state) ** 2
                    )

                    remote_covariance = extract_agent_covariance(
                        belief=pre_communication_belief,
                        agent_id=remote_agent_id,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    covariance_proxy = covariance_trace_per_dim(
                        remote_covariance
                    )[0]

                    pre_update_remote_mse_values.append(
                        float(remote_mse)
                    )

                    pre_update_covariance_proxy_values.append(
                        float(covariance_proxy)
                    )

                communication_decisions = decide_remote_communications(
                    communication_mode=communication_mode,
                    belief_before_remote_messages=pre_communication_belief,
                    remote_agent_ids=remote_agent_ids,
                    step_index=step_index,
                    periodic_interval=periodic_interval,
                    event_trigger_threshold=event_trigger_threshold,
                    position_trigger_radius=position_trigger_radius,
                    position_trigger_scale=position_trigger_scale,
                    num_agents=num_agents,
                    local_state_dim=local_state_dim,
                )

                for remote_agent_id in remote_agent_ids:
                    if not communication_decisions[remote_agent_id]:
                        continue

                    belief, _ = direct_agent_observation_update(
                        predicted_belief=belief,
                        observed_agent_id=remote_agent_id,
                        measurement=true_next_local_states[
                            remote_agent_id
                        ][None, ...],
                        measurement_covariance=measurement_covariance,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    total_remote_messages += 1
                    remote_message_counts[remote_agent_id] += 1
                    episode_message_count += 1

                for remote_agent_id in remote_agent_ids:
                    estimated_remote_state = extract_agent_state(
                        belief=belief,
                        agent_id=remote_agent_id,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    true_remote_state = true_next_local_states[
                        remote_agent_id
                    ][None, ...]

                    remote_mse = jnp.mean(
                        (estimated_remote_state - true_remote_state) ** 2
                    )

                    remote_covariance = extract_agent_covariance(
                        belief=belief,
                        agent_id=remote_agent_id,
                        num_agents=num_agents,
                        local_state_dim=local_state_dim,
                    )

                    covariance_proxy = covariance_trace_per_dim(
                        remote_covariance
                    )[0]

                    post_update_remote_mse_values.append(
                        float(remote_mse)
                    )

                    post_update_covariance_proxy_values.append(
                        float(covariance_proxy)
                    )

                    episode_post_update_mse_values.append(
                        float(remote_mse)
                    )

            if trace_this_episode:
                communication_mask = np.zeros(
                    (num_agents,),
                    dtype=np.bool_,
                )

                for agent_id, should_communicate in (
                        communication_decisions.items()
                ):
                    communication_mask[agent_id] = should_communicate

                trace_steps["true_local_states"].append(
                    np.asarray(true_next_local_states)
                )

                trace_steps["landmark_positions"].append(
                    np.asarray(
                        next_env_state.p_pos[num_agents:]
                    )
                )

                trace_steps["joint_actions"].append(
                    np.asarray(joint_actions)
                )

                trace_steps["predicted_local_means"].append(
                    np.asarray(
                        belief_local_state_means(
                            belief=predicted_belief,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )[0]
                    )
                )

                trace_steps["after_ego_local_means"].append(
                    np.asarray(
                        belief_local_state_means(
                            belief=belief_after_ego,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )[0]
                    )
                )

                trace_steps["posterior_local_means"].append(
                    np.asarray(
                        belief_local_state_means(
                            belief=belief,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )[0]
                    )
                )

                trace_steps["predicted_global_covariances"].append(
                    np.asarray(predicted_belief.covariance[0])
                )

                trace_steps["after_ego_global_covariances"].append(
                    np.asarray(belief_after_ego.covariance[0])
                )

                trace_steps["posterior_global_covariances"].append(
                    np.asarray(belief.covariance[0])
                )

                trace_steps["predicted_covariance_proxy"].append(
                    np.asarray(
                        all_agent_covariance_proxies(
                            belief=predicted_belief,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )
                    )
                )

                trace_steps["after_ego_covariance_proxy"].append(
                    np.asarray(
                        all_agent_covariance_proxies(
                            belief=belief_after_ego,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )
                    )
                )

                trace_steps["posterior_covariance_proxy"].append(
                    np.asarray(
                        all_agent_covariance_proxies(
                            belief=belief,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )
                    )
                )

                trace_steps["predicted_state_mse"].append(
                    np.asarray(
                        all_agent_state_mse(
                            belief=predicted_belief,
                            true_local_states=true_next_local_states,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )
                    )
                )

                trace_steps["after_ego_state_mse"].append(
                    np.asarray(
                        all_agent_state_mse(
                            belief=belief_after_ego,
                            true_local_states=true_next_local_states,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )
                    )
                )

                trace_steps["posterior_state_mse"].append(
                    np.asarray(
                        all_agent_state_mse(
                            belief=belief,
                            true_local_states=true_next_local_states,
                            num_agents=num_agents,
                            local_state_dim=local_state_dim,
                        )
                    )
                )

                trace_steps["communication_decisions"].append(
                    communication_mask
                )
            total_environment_steps += 1
            env_state = next_env_state

            if bool(dones["__all__"]):
                break

        if trace_this_episode:
            trace_path = build_trace_path(
                trace_directory=trace_directory,
                communication_mode=communication_mode,
                position_trigger_radius=position_trigger_radius,
                position_trigger_scale=position_trigger_scale,
                periodic_interval=periodic_interval,
                event_trigger_threshold=event_trigger_threshold,
                ego_agent_id=ego_agent_id,
                episode_index=episode_index,
            )

            trace_payload = {
                key: np.stack(values, axis=0)
                for key, values in trace_steps.items()
            }

            trace_payload.update(
                {
                    "time_indices": np.arange(
                        1,
                        len(trace_steps["true_local_states"]) + 1,
                        dtype=np.int32,
                    ),
                    "initial_true_local_states": trace_initial_local_states,
                    "initial_mean": trace_initial_mean,
                    "initial_covariance": trace_initial_covariance,
                    "initial_landmark_positions": (
                        trace_initial_landmark_positions
                    ),
                    "ego_agent_id": np.int32(ego_agent_id),
                    "num_agents": np.int32(num_agents),
                    "local_state_dim": np.int32(local_state_dim),
                    "event_trigger_threshold": np.float32(
                        event_trigger_threshold
                    ),
                    "periodic_interval": np.int32(
                        periodic_interval
                    ),
                    "communication_mode": np.asarray(
                        communication_mode
                    ),
                }
            )

            np.savez_compressed(
                trace_path,
                **trace_payload,
            )

            print(f"Saved episode trace to: {trace_path}")

        print(
            f"Episode {episode_index + 1:02d} | "
            f"steps={total_environment_steps} | "
            f"remote messages={episode_message_count} | "
            f"post-update remote MSE="
            f"{mean_or_nan(episode_post_update_mse_values):.6f}"
        )

    maximum_remote_messages = (
            total_environment_steps * len(communicating_agent_ids)
    )

    remote_message_rate = (
        total_remote_messages / maximum_remote_messages
        if maximum_remote_messages > 0
        else float("nan")
    )

    metrics = {
        "communication_mode": communication_mode,
        "dynamics_prediction_mode": dynamics_prediction_mode,
        "epistemic_process_scale": epistemic_process_scale,
        "ego_agent_id": ego_agent_id,
        "event_trigger_threshold": event_trigger_threshold,
        "position_trigger_radius": position_trigger_radius,
        "position_trigger_scale": position_trigger_scale,
        "total_environment_steps": total_environment_steps,
        "total_remote_messages": total_remote_messages,
        "remote_message_counts": remote_message_counts,
        "remote_message_rate": remote_message_rate,
        "mean_pre_update_remote_mse": mean_or_nan(
            pre_update_remote_mse_values
        ),
        "mean_post_update_remote_mse": mean_or_nan(
            post_update_remote_mse_values
        ),
        "mean_pre_update_covariance_proxy": mean_or_nan(
            pre_update_covariance_proxy_values
        ),
        "mean_post_update_covariance_proxy": mean_or_nan(
            post_update_covariance_proxy_values
        ),
        "pre_update_remote_mse": jnp.asarray(
            pre_update_remote_mse_values
        ),
        "post_update_remote_mse": jnp.asarray(
            post_update_remote_mse_values
        ),
        "pre_update_covariance_proxy": jnp.asarray(
            pre_update_covariance_proxy_values
        ),
        "post_update_covariance_proxy": jnp.asarray(
            post_update_covariance_proxy_values
        ),
        "sender_error_threshold": sender_error_threshold,
        "sender_covariance_radius_threshold": (
            sender_covariance_radius_threshold
        ),
        "sender_covariance_scale": sender_covariance_scale,
    }

    print("\n===== Online Evaluation Summary =====")
    print(
        "Mean pre-update remote MSE: "
        f"{metrics['mean_pre_update_remote_mse']:.6f}"
    )
    print(
        "Mean post-update remote MSE: "
        f"{metrics['mean_post_update_remote_mse']:.6f}"
    )
    print(
        "Mean pre-update covariance proxy: "
        f"{metrics['mean_pre_update_covariance_proxy']:.6f}"
    )
    print(
        "Mean post-update covariance proxy: "
        f"{metrics['mean_post_update_covariance_proxy']:.6f}"
    )
    print(
        f"Remote messages: {total_remote_messages} / "
        f"{maximum_remote_messages}"
    )
    print(
        f"Remote message rate: {remote_message_rate:.4f}"
    )

    for agent_id in communicating_agent_ids:
        print(
            f"Messages from agent_{agent_id}: "
            f"{remote_message_counts[agent_id]}"
        )

    return metrics