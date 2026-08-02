from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from sender_marl.off_policy.types import SACBatch


@dataclass(frozen=True)
class ReplayBufferStats:
    size: int
    capacity: int
    position: int


class JointReplayBuffer:
    """Host-side circular replay buffer for complete joint transitions.

    Real environment data are stored once. Every independent learner owns a
    separate NumPy RNG and samples its own minibatches from this common real
    data pool. Sharing observations is not parameter sharing: actors, critics,
    targets, temperatures, and optimizers remain fully independent.

    The same schema can later be instantiated separately for each agent's own
    model-generated transitions.
    """

    def __init__(
        self,
        *,
        capacity: int,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        model_state_dim: int,
    ) -> None:
        positive = {
            "capacity": capacity,
            "num_agents": num_agents,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "model_state_dim": model_state_dim,
        }
        invalid = {name: value for name, value in positive.items() if value < 1}
        if invalid:
            raise ValueError(f"Replay dimensions must be positive: {invalid}.")

        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.model_state_dim = int(model_state_dim)

        self.observations = np.empty(
            (capacity, num_agents, observation_dim), dtype=np.float32
        )
        self.actions = np.empty(
            (capacity, num_agents, action_dim), dtype=np.float32
        )
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.dones = np.empty((capacity,), dtype=np.float32)
        self.model_states = np.empty(
            (capacity, model_state_dim), dtype=np.float32
        )
        self.next_model_states = np.empty_like(self.model_states)

        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def stats(self) -> ReplayBufferStats:
        return ReplayBufferStats(
            size=self._size,
            capacity=self.capacity,
            position=self._position,
        )

    def add_batch(
        self,
        *,
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        model_states,
        next_model_states,
    ) -> None:
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)
        model_states = np.asarray(model_states, dtype=np.float32)
        next_model_states = np.asarray(next_model_states, dtype=np.float32)

        expected_obs_tail = (self.num_agents, self.observation_dim)
        expected_action_tail = (self.num_agents, self.action_dim)
        if observations.ndim != 3 or observations.shape[1:] != expected_obs_tail:
            raise ValueError(
                "observations must have shape "
                f"(B, {self.num_agents}, {self.observation_dim}), got "
                f"{observations.shape}."
            )
        if actions.ndim != 3 or actions.shape[1:] != expected_action_tail:
            raise ValueError(
                "actions must have shape "
                f"(B, {self.num_agents}, {self.action_dim}), got "
                f"{actions.shape}."
            )
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations must match observations shape.")
        if model_states.ndim != 2 or model_states.shape[1] != self.model_state_dim:
            raise ValueError(
                "model_states must have shape "
                f"(B, {self.model_state_dim}), got {model_states.shape}."
            )
        if next_model_states.shape != model_states.shape:
            raise ValueError("next_model_states must match model_states shape.")

        batch_size = observations.shape[0]
        for name, array in {
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "dones": dones,
            "model_states": model_states,
            "next_model_states": next_model_states,
        }.items():
            if array.shape[0] != batch_size:
                raise ValueError(f"{name} batch size mismatch.")

        if batch_size >= self.capacity:
            observations = observations[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_observations = next_observations[-self.capacity :]
            dones = dones[-self.capacity :]
            model_states = model_states[-self.capacity :]
            next_model_states = next_model_states[-self.capacity :]
            batch_size = self.capacity

        indices = (
            np.arange(batch_size, dtype=np.int64) + self._position
        ) % self.capacity
        self.observations[indices] = observations
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_observations[indices] = next_observations
        self.dones[indices] = dones
        self.model_states[indices] = model_states
        self.next_model_states[indices] = next_model_states

        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> SACBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} items from size {self._size}."
            )

        indices = rng.integers(0, self._size, size=batch_size, endpoint=False)
        return SACBatch(
            observations=jnp.asarray(self.observations[indices]),
            actions=jnp.asarray(self.actions[indices]),
            rewards=jnp.asarray(self.rewards[indices]),
            next_observations=jnp.asarray(self.next_observations[indices]),
            dones=jnp.asarray(self.dones[indices]),
            model_states=jnp.asarray(self.model_states[indices]),
            next_model_states=jnp.asarray(self.next_model_states[indices]),
        )

    def sample_model_states(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ):
        """Sample real joint states as independent model-rollout branch points."""
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} items from size {self._size}."
            )
        indices = rng.integers(0, self._size, size=batch_size, endpoint=False)
        return jnp.asarray(self.model_states[indices])


# Backwards-compatible name for code that imported the first foundation.
LocalReplayBuffer = JointReplayBuffer
