from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import jax

from sender_marl.core.types import EnvFeatures, EnvStep


Array = jax.Array


@runtime_checkable
class CommonStateSource(Protocol):
    """Supplies the remote-agent state used to construct actor observations.

    The oracle implementation below exposes true local states and is intended
    for policy-optimizer smoke tests/baselines. A sender-triggered estimator can
    later implement the same interface without changing the rollout collector.
    """

    def initialize(self, features: EnvFeatures) -> Any:
        ...

    def read(self, source_state: Any) -> Array:
        """Return (..., num_agents, local_state_dim) common local states."""
        ...

    def update(
        self,
        key: Array,
        source_state: Any,
        previous_features: EnvFeatures,
        policy_actions: Array,
        step_output: EnvStep,
    ) -> tuple[Any, Mapping[str, Array]]:
        ...


@dataclass(frozen=True)
class OracleCommonStateSource:
    """Full-information state source used to validate policy training code."""

    def initialize(self, features: EnvFeatures) -> Array:
        return features.local_states

    def read(self, source_state: Array) -> Array:
        return source_state

    def update(
        self,
        key: Array,
        source_state: Array,
        previous_features: EnvFeatures,
        policy_actions: Array,
        step_output: EnvStep,
    ) -> tuple[Array, Mapping[str, Array]]:
        del key, source_state, previous_features, policy_actions
        return step_output.features.local_states, {}
