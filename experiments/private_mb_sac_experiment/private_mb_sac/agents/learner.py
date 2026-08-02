"""One fully independent actor/critic/temperature learner per rollout owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from private_mb_sac.agents.networks import SACActor, TwinQNetwork


def _optimizer(learning_rate: float, maximum_gradient_norm: float):
    return optax.chain(
        optax.clip_by_global_norm(maximum_gradient_norm),
        optax.adam(learning_rate),
    )


@struct.dataclass
class SACLearnerState:
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    target_critic_params: Any


@dataclass
class OwnerSACRuntime:
    owner_id: int
    learner_state: SACLearnerState
    critic_updates_total: int = 0
    actor_updates_total: int = 0
    fixed_batch: Any | None = None
    update_fn: Any | None = None
    fixed_evaluator_fn: Any | None = None


def initialize_owner_sac_runtimes(
    config: Any,
    *,
    actor: SACActor,
    initial_actor_params: Sequence[Any],
) -> tuple[TwinQNetwork, list[OwnerSACRuntime]]:
    """Create three private SAC learners without sharing optimizer state."""

    num_agents = int(config.experiment.num_agents)
    if len(initial_actor_params) != num_agents:
        raise ValueError(
            "The number of actor parameter trees must equal num_agents."
        )

    critic = TwinQNetwork(
        hidden_dims=tuple(int(value) for value in config.critic.hidden_dims)
    )
    dummy_state = jnp.zeros(
        (int(config.critic.state_dim),),
        dtype=jnp.float32,
    )
    dummy_action = jnp.zeros(
        (
            num_agents,
            int(config.actor.action_dim),
        ),
        dtype=jnp.float32,
    )
    critic_keys = jax.random.split(
        jax.random.PRNGKey(int(config.experiment.seed) + 90_000),
        num_agents,
    )

    runtimes: list[OwnerSACRuntime] = []
    for owner_id in range(num_agents):
        critic_params = critic.init(
            critic_keys[owner_id],
            dummy_state,
            dummy_action,
        )["params"]

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=initial_actor_params[owner_id],
            tx=_optimizer(
                float(config.actor.learning_rate),
                float(config.actor.maximum_gradient_norm),
            ),
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=_optimizer(
                float(config.critic.learning_rate),
                float(config.critic.maximum_gradient_norm),
            ),
        )
        alpha_state = TrainState.create(
            apply_fn=lambda variables: jnp.exp(
                variables["params"]["log_alpha"]
            ),
            params={
                "log_alpha": jnp.asarray(
                    jnp.log(float(config.temperature.initial_alpha)),
                    dtype=jnp.float32,
                )
            },
            tx=_optimizer(
                float(config.temperature.learning_rate),
                float(config.temperature.maximum_gradient_norm),
            ),
        )

        runtimes.append(
            OwnerSACRuntime(
                owner_id=owner_id,
                learner_state=SACLearnerState(
                    actor_state=actor_state,
                    critic_state=critic_state,
                    alpha_state=alpha_state,
                    target_critic_params=critic_params,
                ),
            )
        )

    if len({id(runtime) for runtime in runtimes}) != num_agents:
        raise RuntimeError("SAC runtime object sharing was detected.")
    if len(
        {
            id(runtime.learner_state.actor_state.opt_state)
            for runtime in runtimes
        }
    ) != num_agents:
        raise RuntimeError("Actor optimizer-state sharing was detected.")

    return critic, runtimes


def live_actor_params(
    runtimes: Sequence[OwnerSACRuntime],
) -> tuple[Any, ...]:
    return tuple(
        runtime.learner_state.actor_state.params
        for runtime in runtimes
    )


def attach_owner_sac_functions(
    runtime: OwnerSACRuntime,
    *,
    config: Any,
    actor_normalizer: Any,
    dynamics_normalizer: Any,
    jit: bool,
) -> None:
    """Attach functions after the owner's private normalizer has been fitted."""

    from private_mb_sac.training.sac_update import (
        make_fixed_batch_critic_evaluator,
        make_private_sac_update,
    )

    if runtime.update_fn is not None:
        return

    runtime.update_fn = make_private_sac_update(
        config,
        owner_id=runtime.owner_id,
        actor_normalizer=actor_normalizer,
        dynamics_normalizer=dynamics_normalizer,
        jit=jit,
    )
    runtime.fixed_evaluator_fn = make_fixed_batch_critic_evaluator(
        config,
        owner_id=runtime.owner_id,
        actor_normalizer=actor_normalizer,
        dynamics_normalizer=dynamics_normalizer,
        jit=jit,
    )
