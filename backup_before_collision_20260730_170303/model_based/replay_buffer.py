from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from sender_marl.model_based.types import LocalDynamicsBatch


@dataclass(frozen=True)
class DynamicsBufferStats:
    size: int
    capacity: int
    position: int


class PooledLocalDynamicsBuffer:
    """Host-side circular buffer pooling homogeneous-agent transitions.

    The model input remains factorized: each row contains only one agent's
    local physical state and local action. Pooling does not introduce remote
    state or action features.

    ``collision_flags`` is metadata used only for collision-conditioned
    validation. It is not part of the dynamics model input or training loss.
    """

    def __init__(
        self,
        *,
        capacity: int,
        local_state_dim: int,
        action_dim: int,
    ) -> None:
        if capacity < 1 or local_state_dim < 1 or action_dim < 1:
            raise ValueError("All replay dimensions must be positive.")

        self.capacity = int(capacity)
        self.local_state_dim = int(local_state_dim)
        self.action_dim = int(action_dim)

        self.local_states = np.empty(
            (capacity, local_state_dim),
            dtype=np.float32,
        )
        self.actions = np.empty(
            (capacity, action_dim),
            dtype=np.float32,
        )
        self.next_local_states = np.empty_like(self.local_states)
        self.agent_ids = np.empty((capacity,), dtype=np.int32)

        # Per-local-transition metadata. True means that this particular agent
        # participates in a collision in the transition start state x_t.
        self.collision_flags = np.empty((capacity,), dtype=np.bool_)

        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def stats(self) -> DynamicsBufferStats:
        return DynamicsBufferStats(
            size=self._size,
            capacity=self.capacity,
            position=self._position,
        )

    def add_batch(
        self,
        *,
        local_states,
        actions,
        next_local_states,
        agent_ids,
        collision_flags,
    ) -> None:
        local_states = np.asarray(local_states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_local_states = np.asarray(
            next_local_states,
            dtype=np.float32,
        )
        agent_ids = np.asarray(agent_ids, dtype=np.int32).reshape(-1)
        collision_flags = np.asarray(
            collision_flags,
            dtype=np.bool_,
        ).reshape(-1)

        if (
            local_states.ndim != 2
            or local_states.shape[1] != self.local_state_dim
        ):
            raise ValueError(
                "local_states must have shape "
                f"(B, {self.local_state_dim}), got {local_states.shape}."
            )
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"actions must have shape (B, {self.action_dim}), "
                f"got {actions.shape}."
            )
        if next_local_states.shape != local_states.shape:
            raise ValueError(
                "next_local_states must match local_states shape."
            )

        batch_size = local_states.shape[0]
        if (
            actions.shape[0] != batch_size
            or agent_ids.shape[0] != batch_size
            or collision_flags.shape[0] != batch_size
        ):
            raise ValueError(
                "Dynamics batch sizes do not match: "
                f"states={batch_size}, actions={actions.shape[0]}, "
                f"agent_ids={agent_ids.shape[0]}, "
                f"collision_flags={collision_flags.shape[0]}."
            )

        if batch_size >= self.capacity:
            local_states = local_states[-self.capacity :]
            actions = actions[-self.capacity :]
            next_local_states = next_local_states[-self.capacity :]
            agent_ids = agent_ids[-self.capacity :]
            collision_flags = collision_flags[-self.capacity :]
            batch_size = self.capacity

        indices = (
            np.arange(batch_size, dtype=np.int64) + self._position
        ) % self.capacity

        self.local_states[indices] = local_states
        self.actions[indices] = actions
        self.next_local_states[indices] = next_local_states
        self.agent_ids[indices] = agent_ids
        self.collision_flags[indices] = collision_flags

        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> LocalDynamicsBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} rows from buffer size "
                f"{self._size}."
            )

        indices = rng.integers(
            0,
            self._size,
            size=batch_size,
            endpoint=False,
        )
        return self.batch_from_indices(indices)

    def batch_from_indices(self, indices) -> LocalDynamicsBatch:
        indices = np.asarray(indices, dtype=np.int64)
        return LocalDynamicsBatch(
            local_states=jnp.asarray(self.local_states[indices]),
            actions=jnp.asarray(self.actions[indices]),
            next_local_states=jnp.asarray(
                self.next_local_states[indices]
            ),
            agent_ids=jnp.asarray(self.agent_ids[indices]),
            collision_flags=jnp.asarray(
                self.collision_flags[indices]
            ),
        )

    def all_data(self) -> LocalDynamicsBatch:
        if self._size == 0:
            raise ValueError("Buffer is empty.")
        return self.batch_from_indices(
            np.arange(self._size, dtype=np.int64)
        )
