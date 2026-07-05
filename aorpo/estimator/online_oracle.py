# aorpo/estimator/online_oracle.py
from __future__ import annotations

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
    predict_global_belief_oracle_actions,
)
from aorpo.estimator.uncertainty import covariance_trace_per_dim

from aorpo.estimator.trigger import (
    trace_threshold_trigger,
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

    raise ValueError(
        "Unsupported communication_mode: "
        f"{communication_mode}. "
        "Expected one of: initial_sync_only, always_communicate, "
        "periodic, event_triggered."
    )


def mean_or_nan(values: list[float]) -> float:
    """
    Return the mean of a list, or NaN when the list is empty.
    """
    if not values:
        return float("nan")

    return float(jnp.mean(jnp.asarray(values)))


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

    periodic_interval = int(
        cfg.online_estimator.periodic_interval
    )

    event_trigger_threshold = float(
        cfg.online_estimator.event_trigger_threshold
    )

    initial_variance = float(
        cfg.online_estimator.initial_variance
    )

    measurement_variance = float(
        cfg.online_estimator.measurement_variance
    )

    action_min = float(cfg.online_estimator.action_min)
    action_max = float(cfg.online_estimator.action_max)

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

    env = make_mpe_env(cfg)

    rng = jax.random.PRNGKey(
        int(cfg.seed) + 12345
    )

    remote_agent_ids = [
        agent_id
        for agent_id in range(num_agents)
        if agent_id != ego_agent_id
    ]

    pre_update_remote_mse_values: list[float] = []
    post_update_remote_mse_values: list[float] = []

    pre_update_covariance_proxy_values: list[float] = []
    post_update_covariance_proxy_values: list[float] = []

    total_remote_messages = 0
    total_environment_steps = 0

    remote_message_counts = {
        remote_agent_id: 0
        for remote_agent_id in remote_agent_ids
    }

    print(
        "\n===== Online Global Belief Evaluation "
        "with Oracle Actions ====="
    )

    print(f"Ego receiver agent: {ego_agent_id}")
    print(f"Communication mode: {communication_mode}")
    print(f"Number of episodes: {num_episodes}")

    if communication_mode == "event_triggered":
        print(
            "Event trigger threshold "
            f"(trace(P_jj) / local_state_dim): "
            f"{event_trigger_threshold:.6f}"
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

            predicted_belief, _ = (
                predict_global_belief_oracle_actions(
                    belief=belief,
                    model_states=model_states,
                    standardizers=standardizers,
                    joint_actions=joint_actions[None, ...],
                    num_agents=num_agents,
                    local_state_dim=local_state_dim,
                )
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

            belief, _ = direct_agent_observation_update(
                predicted_belief=predicted_belief,
                observed_agent_id=ego_agent_id,
                measurement=true_next_local_states[
                    ego_agent_id
                ][None, ...],
                measurement_covariance=measurement_covariance,
                num_agents=num_agents,
                local_state_dim=local_state_dim,
            )

            pre_communication_belief = belief

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

            total_environment_steps += 1
            env_state = next_env_state

            if bool(dones["__all__"]):
                break

        print(
            f"Episode {episode_index + 1:02d} | "
            f"steps={total_environment_steps} | "
            f"remote messages={episode_message_count} | "
            f"post-update remote MSE="
            f"{mean_or_nan(episode_post_update_mse_values):.6f}"
        )

    maximum_remote_messages = (
        total_environment_steps * len(remote_agent_ids)
    )

    remote_message_rate = (
        total_remote_messages / maximum_remote_messages
        if maximum_remote_messages > 0
        else float("nan")
    )

    metrics = {
        "communication_mode": communication_mode,
        "ego_agent_id": ego_agent_id,
        "event_trigger_threshold": event_trigger_threshold,
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

    for remote_agent_id in remote_agent_ids:
        print(
            f"Messages from agent_{remote_agent_id}: "
            f"{remote_message_counts[remote_agent_id]}"
        )

    return metrics