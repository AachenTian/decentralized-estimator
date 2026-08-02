from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import optax
from flax.training.train_state import TrainState


Array = jax.Array


@dataclass(frozen=True)
class PPOConfig:
    """Static hyperparameters for one compiled PPO update."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    actor_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    max_gradient_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    normalize_advantages: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1].")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive.")
        if self.update_epochs < 1:
            raise ValueError("update_epochs must be positive.")
        if self.num_minibatches < 1:
            raise ValueError("num_minibatches must be positive.")


def _optimizer(learning_rate: float, max_gradient_norm: float):
    return optax.chain(
        optax.clip_by_global_norm(max_gradient_norm),
        optax.adam(learning_rate),
    )


def create_ppo_train_states(
    key: Array,
    actor: Any,
    value_network: Any,
    dummy_actor_obs: Array,
    dummy_critic_obs: Array,
    config: PPOConfig,
) -> tuple[TrainState, TrainState]:
    """Initialize separate actor and value-function train states."""

    config.validate()
    actor_key, value_key = jax.random.split(key)
    actor_params = actor.init(actor_key, dummy_actor_obs)["params"]
    value_params = value_network.init(value_key, dummy_critic_obs)["params"]

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=_optimizer(
            config.actor_learning_rate,
            config.max_gradient_norm,
        ),
    )
    value_state = TrainState.create(
        apply_fn=value_network.apply,
        params=value_params,
        tx=_optimizer(
            config.value_learning_rate,
            config.max_gradient_norm,
        ),
    )
    return actor_state, value_state
