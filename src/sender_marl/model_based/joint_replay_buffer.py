from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from sender_marl.model_based.joint_types import JointDynamicsBatch


@dataclass(frozen=True)
class JointDynamicsBufferStats:
    size: int
    capacity: int
    position: int
    collision_joint_rows: int


class JointDynamicsReplayBuffer:
    """Host-side buffer of complete multi-agent transitions.

    A row is a joint transition. All independent model owners can sample from
    the same physical data, while using separate parameters, optimizers,
    bootstrap indices, and minibatch RNG streams.
    """

    def __init__(
        self,
        *,
        capacity: int,
        model_state_dim: int,
        num_agents: int,
        action_dim: int,
        local_state_dim: int,
    ) -> None:
        values = {
            "capacity": capacity,
            "model_state_dim": model_state_dim,
            "num_agents": num_agents,
            "action_dim": action_dim,
            "local_state_dim": local_state_dim,
        }
        invalid = {name: value for name, value in values.items() if value < 1}
        if invalid:
            raise ValueError(f"Replay dimensions must be positive: {invalid}.")

        self.capacity = int(capacity)
        self.model_state_dim = int(model_state_dim)
        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.local_state_dim = int(local_state_dim)

        self.model_states = np.empty(
            (capacity, model_state_dim), dtype=np.float32
        )
        self.joint_actions = np.empty(
            (capacity, num_agents, action_dim), dtype=np.float32
        )
        self.local_states = np.empty(
            (capacity, num_agents, local_state_dim), dtype=np.float32
        )
        self.next_local_states = np.empty_like(self.local_states)
        self.agent_collision_flags = np.empty(
            (capacity, num_agents), dtype=np.bool_
        )
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def stats(self) -> JointDynamicsBufferStats:
        collision_rows = 0
        if self._size:
            collision_rows = int(
                np.sum(np.any(self.agent_collision_flags[: self._size], axis=-1))
            )
        return JointDynamicsBufferStats(
            size=self._size,
            capacity=self.capacity,
            position=self._position,
            collision_joint_rows=collision_rows,
        )

    def add_batch(
        self,
        *,
        model_states,
        joint_actions,
        local_states,
        next_local_states,
        agent_collision_flags,
    ) -> None:
        model_states = np.asarray(model_states, dtype=np.float32)
        joint_actions = np.asarray(joint_actions, dtype=np.float32)
        local_states = np.asarray(local_states, dtype=np.float32)
        next_local_states = np.asarray(next_local_states, dtype=np.float32)
        agent_collision_flags = np.asarray(
            agent_collision_flags, dtype=np.bool_
        )

        expected_action_tail = (self.num_agents, self.action_dim)
        expected_local_tail = (self.num_agents, self.local_state_dim)
        if model_states.ndim != 2 or model_states.shape[1] != self.model_state_dim:
            raise ValueError(
                "model_states must have shape "
                f"(B, {self.model_state_dim}), got {model_states.shape}."
            )
        if joint_actions.ndim != 3 or joint_actions.shape[1:] != expected_action_tail:
            raise ValueError(
                "joint_actions must have shape "
                f"(B, {expected_action_tail[0]}, {expected_action_tail[1]}), "
                f"got {joint_actions.shape}."
            )
        if local_states.ndim != 3 or local_states.shape[1:] != expected_local_tail:
            raise ValueError(
                "local_states must have shape "
                f"(B, {expected_local_tail[0]}, {expected_local_tail[1]}), "
                f"got {local_states.shape}."
            )
        if next_local_states.shape != local_states.shape:
            raise ValueError("next_local_states must match local_states shape.")
        if agent_collision_flags.shape != (model_states.shape[0], self.num_agents):
            raise ValueError(
                "agent_collision_flags must have shape "
                f"(B, {self.num_agents}), got {agent_collision_flags.shape}."
            )

        batch_size = model_states.shape[0]
        if (
            joint_actions.shape[0] != batch_size
            or local_states.shape[0] != batch_size
        ):
            raise ValueError("Joint dynamics batch sizes do not match.")

        if batch_size >= self.capacity:
            model_states = model_states[-self.capacity :]
            joint_actions = joint_actions[-self.capacity :]
            local_states = local_states[-self.capacity :]
            next_local_states = next_local_states[-self.capacity :]
            agent_collision_flags = agent_collision_flags[-self.capacity :]
            batch_size = self.capacity

        indices = (
            np.arange(batch_size, dtype=np.int64) + self._position
        ) % self.capacity
        self.model_states[indices] = model_states
        self.joint_actions[indices] = joint_actions
        self.local_states[indices] = local_states
        self.next_local_states[indices] = next_local_states
        self.agent_collision_flags[indices] = agent_collision_flags
        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        agent_id: int | None = None,
        collision_fraction: float | None = None,
    ) -> JointDynamicsBatch:
        """Sample uniformly or balance collision involvement for one agent.

        collision_fraction=None gives a strictly uniform sample, which should
        be used for the first fair local-input versus joint-input comparison.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} rows from buffer size {self._size}."
            )

        if collision_fraction is None:
            indices = rng.integers(
                0, self._size, size=batch_size, endpoint=False
            )
            return self.batch_from_indices(indices)

        if agent_id is None:
            raise ValueError(
                "agent_id is required when collision_fraction is specified."
            )
        if not 0 <= agent_id < self.num_agents:
            raise ValueError(f"Invalid agent_id={agent_id}.")
        if not 0.0 <= collision_fraction <= 1.0:
            raise ValueError("collision_fraction must lie in [0, 1].")

        flags = self.agent_collision_flags[: self._size, agent_id]
        collision_indices = np.flatnonzero(flags)
        noncollision_indices = np.flatnonzero(~flags)
        requested_collision = int(round(batch_size * collision_fraction))
        requested_noncollision = batch_size - requested_collision

        if collision_indices.size == 0 or noncollision_indices.size == 0:
            indices = rng.integers(
                0, self._size, size=batch_size, endpoint=False
            )
            return self.batch_from_indices(indices)

        collision_sample = rng.choice(
            collision_indices,
            size=requested_collision,
            replace=collision_indices.size < requested_collision,
        )
        noncollision_sample = rng.choice(
            noncollision_indices,
            size=requested_noncollision,
            replace=noncollision_indices.size < requested_noncollision,
        )
        indices = np.concatenate([collision_sample, noncollision_sample])
        rng.shuffle(indices)
        return self.batch_from_indices(indices)

    def batch_from_indices(self, indices) -> JointDynamicsBatch:
        indices = np.asarray(indices, dtype=np.int64)
        return JointDynamicsBatch(
            model_states=jnp.asarray(self.model_states[indices]),
            joint_actions=jnp.asarray(self.joint_actions[indices]),
            local_states=jnp.asarray(self.local_states[indices]),
            next_local_states=jnp.asarray(self.next_local_states[indices]),
            agent_collision_flags=jnp.asarray(
                self.agent_collision_flags[indices]
            ),
        )

    def all_data(self) -> JointDynamicsBatch:
        if self._size == 0:
            raise ValueError("Buffer is empty.")
        return self.batch_from_indices(np.arange(self._size, dtype=np.int64))
