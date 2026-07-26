from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    """Static metadata for one compiled multi-agent training run.

    The values may differ between environments, but they remain fixed during one
    JAX compilation. Algorithms read these dimensions instead of hard-coding
    Simple Spread, vehicle, or drone-specific sizes.
    """

    env_name: str
    num_agents: int

    actor_obs_dim: int
    critic_obs_dim: int
    private_obs_dim: int
    public_context_dim: int
    local_state_dim: int
    model_state_dim: int

    policy_action_dim: int
    native_action_dim: int

    homogeneous_agents: bool = True
    shared_reward: bool = True

    def __post_init__(self) -> None:
        integer_fields = {
            "num_agents": self.num_agents,
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "private_obs_dim": self.private_obs_dim,
            "public_context_dim": self.public_context_dim,
            "local_state_dim": self.local_state_dim,
            "model_state_dim": self.model_state_dim,
            "policy_action_dim": self.policy_action_dim,
            "native_action_dim": self.native_action_dim,
        }
        invalid = {name: value for name, value in integer_fields.items() if value <= 0}
        if invalid:
            raise ValueError(f"All dimensions must be positive, got {invalid}.")

    @property
    def joint_policy_action_dim(self) -> int:
        return self.num_agents * self.policy_action_dim
