from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax

from sender_marl.core.specs import EnvSpec
from sender_marl.core.types import EnvFeatures, EnvStep


Array = jax.Array


@runtime_checkable
class MultiAgentEnvAdapter(Protocol):
    """Contract used by IPPO, MAPPO, DPO, MAMBPO, and communication layers."""

    spec: EnvSpec
    agents: tuple[str, ...]

    def reset(self, key: Array) -> tuple[Any, EnvFeatures]:
        """Reset and return native environment state plus semantic features."""
        ...

    def step(
        self,
        key: Array,
        env_state: Any,
        policy_actions: Array,
    ) -> EnvStep:
        """Advance one step using standardized policy actions in [-1, 1]."""
        ...

    def encode_policy_actions(self, policy_actions: Array) -> Any:
        """Convert (..., N, A_policy) actions into the native environment format."""
        ...

    def build_actor_observations(
        self,
        features: EnvFeatures,
        common_local_states: Array,
    ) -> Array:
        """Build (..., N, actor_obs_dim) without leaking true remote states."""
        ...

    def build_critic_observations(self, features: EnvFeatures) -> Array:
        """Build centralized training input."""
        ...
