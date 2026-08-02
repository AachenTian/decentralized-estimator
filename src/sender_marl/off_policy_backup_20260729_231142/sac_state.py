from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flax import struct
from flax.training.train_state import TrainState


@dataclass(frozen=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float = -2.0
    max_gradient_norm: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if self.actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive.")
        if self.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive.")
        if self.alpha_learning_rate <= 0.0:
            raise ValueError("alpha_learning_rate must be positive.")
        if self.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive.")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive.")


@struct.dataclass
class SACLearnerState:
    """Trainable state owned exclusively by one agent."""

    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    target_critic_params: Any


@struct.dataclass
class SACUpdateOutput:
    learner_state: SACLearnerState
    metrics: dict[str, Any]
