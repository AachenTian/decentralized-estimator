from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from sender_marl.off_policy.normalization import (
    NormalizationStats,
    normalize,
)
from sender_marl.off_policy.sac_state import SACConfig, SACLearnerState


Array = jax.Array
ApplyFn = Callable[..., Any]


def _optimizer(learning_rate: float, max_gradient_norm: float):
    return optax.chain(
        optax.clip_by_global_norm(max_gradient_norm),
        optax.adam(learning_rate),
    )


def make_independent_sac_actor_apply(
    actor_apply_fn: ApplyFn,
    num_agents: int,
    observation_stats: NormalizationStats | None = None,
) -> ApplyFn:
    """Apply separate actor parameters along the joint observation agent axis."""

    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    def apply(variables: dict[str, Any], observations: Array):
        params_by_agent = variables["params"]
        if len(params_by_agent) != num_agents:
            raise ValueError(
                "Expected one actor parameter tree per agent; got "
                f"{len(params_by_agent)} for {num_agents} agents."
            )

        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observation_stats is not None:
            observations = normalize(observations, observation_stats)
        if observations.shape[-2] != num_agents:
            raise ValueError(
                "Actor observations must contain the agent axis at -2; "
                f"got shape={observations.shape}."
            )

        means = []
        log_stds = []
        for agent_id in range(num_agents):
            mean_i, log_std_i = actor_apply_fn(
                {"params": params_by_agent[agent_id]},
                observations[..., agent_id, :],
            )
            means.append(mean_i)
            log_stds.append(log_std_i)
        return jnp.stack(means, axis=-2), jnp.stack(log_stds, axis=-2)

    return apply


def create_independent_sac_states(
    key: Array,
    actor: Any,
    critic: Any,
    *,
    num_agents: int,
    actor_obs_dim: int,
    critic_state_dim: int,
    action_dim: int,
    config: SACConfig,
) -> tuple[SACLearnerState, ...]:
    """Initialize private actor, centralized twin critics, and alpha per agent."""

    if num_agents < 1:
        raise ValueError("num_agents must be positive.")

    dummy_actor_observation = jnp.zeros(
        (actor_obs_dim,), dtype=jnp.float32
    )
    dummy_model_state = jnp.zeros(
        (critic_state_dim,), dtype=jnp.float32
    )
    dummy_joint_action = jnp.zeros(
        (num_agents, action_dim), dtype=jnp.float32
    )
    agent_keys = jax.random.split(key, num_agents)

    learners = []
    for agent_key in agent_keys:
        actor_key, critic_key = jax.random.split(agent_key)
        actor_params = actor.init(
            actor_key, dummy_actor_observation
        )["params"]
        critic_params = critic.init(
            critic_key, dummy_model_state, dummy_joint_action
        )["params"]

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=_optimizer(
                config.actor_learning_rate,
                config.max_gradient_norm,
            ),
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=_optimizer(
                config.critic_learning_rate,
                config.max_gradient_norm,
            ),
        )
        alpha_state = TrainState.create(
            apply_fn=lambda variables: jnp.exp(
                variables["params"]["log_alpha"]
            ),
            params={
                "log_alpha": jnp.asarray(
                    jnp.log(config.initial_alpha), dtype=jnp.float32
                )
            },
            tx=_optimizer(
                config.alpha_learning_rate,
                config.max_gradient_norm,
            ),
        )

        learners.append(
            SACLearnerState(
                actor_state=actor_state,
                critic_state=critic_state,
                alpha_state=alpha_state,
                target_critic_params=critic_params,
            )
        )
    return tuple(learners)


def independent_actor_params(
    learner_states: Sequence[SACLearnerState],
) -> tuple[Any, ...]:
    """Return a frozen snapshot tuple without averaging or overwriting actors."""

    return tuple(learner.actor_state.params for learner in learner_states)
