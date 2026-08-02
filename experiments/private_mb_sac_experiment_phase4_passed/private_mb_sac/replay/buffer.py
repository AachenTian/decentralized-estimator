"""Strictly private circular replay buffers for complete joint transitions."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from private_mb_sac.core.types import RealRolloutBatch, ReplayBatch


@dataclass(frozen=True)
class ReplayBufferStats:
    size: int
    capacity: int
    position: int
    terminal_count: int


class PrivateJointReplayBuffer:
    """One owner's host-side replay. Never share this object across owners."""

    def __init__(self, *, capacity: int, num_agents: int, observation_dim: int,
                 action_dim: int, model_state_dim: int) -> None:
        values = dict(capacity=capacity, num_agents=num_agents,
                      observation_dim=observation_dim, action_dim=action_dim,
                      model_state_dim=model_state_dim)
        invalid = {k: v for k, v in values.items() if int(v) < 1}
        if invalid:
            raise ValueError(f"Replay dimensions must be positive: {invalid}")
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.model_state_dim = int(model_state_dim)

        self.observations = np.empty((self.capacity, self.num_agents, self.observation_dim), np.float32)
        self.actions = np.empty((self.capacity, self.num_agents, self.action_dim), np.float32)
        self.rewards = np.empty((self.capacity, self.num_agents), np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.dones = np.empty((self.capacity, self.num_agents), np.float32)
        self.model_states = np.empty((self.capacity, self.model_state_dim), np.float32)
        self.next_model_states = np.empty_like(self.model_states)
        self.episode_dones = np.empty((self.capacity,), np.bool_)
        self.pair_collision_rates = np.empty((self.capacity,), np.float32)
        self.min_pair_distances = np.empty((self.capacity,), np.float32)
        self.episode_steps = np.empty((self.capacity,), np.int32)
        self.next_episode_steps = np.empty((self.capacity,), np.int32)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def stats(self) -> ReplayBufferStats:
        return ReplayBufferStats(
            size=self._size, capacity=self.capacity, position=self._position,
            terminal_count=int(np.sum(self.episode_dones[:self._size])),
        )

    @property
    def valid_dynamics_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.episode_dones[:self._size])

    def add_rollout(self, batch: RealRolloutBatch, valid_mask=None) -> int:
        """Flatten `(T,E,...)`, optionally retain only `valid_mask`, and append."""
        t, e = np.asarray(batch.episode_dones).shape
        def flatten(value):
            array = np.asarray(value)
            return array.reshape((t * e,) + array.shape[2:])
        mask = np.ones((t * e,), dtype=np.bool_) if valid_mask is None else np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
        if mask.shape != (t * e,):
            raise ValueError(f"valid_mask must flatten to {(t*e,)}, got {mask.shape}.")
        episode_steps = batch.episode_steps
        next_episode_steps = batch.next_episode_steps
        if episode_steps is None:
            episode_steps = np.zeros((t, e), dtype=np.int32)
        if next_episode_steps is None:
            next_episode_steps = np.asarray(episode_steps, dtype=np.int32) + 1
        self.add_batch(
            observations=flatten(batch.observations)[mask],
            actions=flatten(batch.actions)[mask],
            rewards=flatten(batch.rewards)[mask],
            next_observations=flatten(batch.next_observations)[mask],
            dones=flatten(batch.dones)[mask],
            model_states=flatten(batch.model_states)[mask],
            next_model_states=flatten(batch.next_model_states)[mask],
            episode_dones=np.asarray(batch.episode_dones).reshape(-1)[mask],
            pair_collision_rates=np.asarray(batch.pair_collision_rates).reshape(-1)[mask],
            min_pair_distances=np.asarray(batch.min_pair_distances).reshape(-1)[mask],
            episode_steps=np.asarray(episode_steps).reshape(-1)[mask],
            next_episode_steps=np.asarray(next_episode_steps).reshape(-1)[mask],
        )
        return int(mask.sum())

    def add_batch(self, *, observations, actions, rewards, next_observations,
                  dones, model_states, next_model_states, episode_dones,
                  pair_collision_rates, min_pair_distances,
                  episode_steps=None, next_episode_steps=None) -> None:
        observations = np.asarray(observations, dtype=np.float32)
        batch_size = observations.shape[0]
        if episode_steps is None:
            episode_steps = np.zeros((batch_size,), dtype=np.int32)
        if next_episode_steps is None:
            next_episode_steps = np.asarray(episode_steps, dtype=np.int32) + 1
        arrays = {
            'observations': observations,
            'actions': np.asarray(actions, dtype=np.float32),
            'rewards': np.asarray(rewards, dtype=np.float32),
            'next_observations': np.asarray(next_observations, dtype=np.float32),
            'dones': np.asarray(dones, dtype=np.float32),
            'model_states': np.asarray(model_states, dtype=np.float32),
            'next_model_states': np.asarray(next_model_states, dtype=np.float32),
            'episode_dones': np.asarray(episode_dones, dtype=np.bool_).reshape(-1),
            'pair_collision_rates': np.asarray(pair_collision_rates, dtype=np.float32).reshape(-1),
            'min_pair_distances': np.asarray(min_pair_distances, dtype=np.float32).reshape(-1),
            'episode_steps': np.asarray(episode_steps, dtype=np.int32).reshape(-1),
            'next_episode_steps': np.asarray(next_episode_steps, dtype=np.int32).reshape(-1),
        }
        expected = {
            'observations': (batch_size, self.num_agents, self.observation_dim),
            'actions': (batch_size, self.num_agents, self.action_dim),
            'rewards': (batch_size, self.num_agents),
            'next_observations': (batch_size, self.num_agents, self.observation_dim),
            'dones': (batch_size, self.num_agents),
            'model_states': (batch_size, self.model_state_dim),
            'next_model_states': (batch_size, self.model_state_dim),
            'episode_dones': (batch_size,),
            'pair_collision_rates': (batch_size,),
            'min_pair_distances': (batch_size,),
            'episode_steps': (batch_size,),
            'next_episode_steps': (batch_size,),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arrays[name].shape}.")
        if batch_size >= self.capacity:
            arrays = {name: value[-self.capacity:] for name, value in arrays.items()}
            batch_size = self.capacity
        indices = (np.arange(batch_size) + self._position) % self.capacity
        for name, value in arrays.items():
            getattr(self, name)[indices] = value
        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def _sample_indices(self, batch_size: int, rng: np.random.Generator, *,
                        nonterminal_only: bool, replace: bool | None) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        candidates = self.valid_dynamics_indices if nonterminal_only else np.arange(self._size, dtype=np.int64)
        if len(candidates) < 1:
            raise ValueError("No eligible replay items are available.")
        if replace is None:
            replace = len(candidates) < batch_size
        if not replace and len(candidates) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} unique items from {len(candidates)} eligible replay items.")
        return rng.choice(candidates, size=batch_size, replace=bool(replace))

    def sample(self, batch_size: int, rng: np.random.Generator, *,
               nonterminal_only: bool=False, replace: bool|None=None) -> ReplayBatch:
        indices = self._sample_indices(batch_size, rng, nonterminal_only=nonterminal_only, replace=replace)
        return ReplayBatch(
            observations=jnp.asarray(self.observations[indices]),
            actions=jnp.asarray(self.actions[indices]),
            rewards=jnp.asarray(self.rewards[indices]),
            next_observations=jnp.asarray(self.next_observations[indices]),
            dones=jnp.asarray(self.dones[indices]),
            model_states=jnp.asarray(self.model_states[indices]),
            next_model_states=jnp.asarray(self.next_model_states[indices]),
            pair_collision_rates=jnp.asarray(self.pair_collision_rates[indices]),
            min_pair_distances=jnp.asarray(self.min_pair_distances[indices]),
            episode_steps=jnp.asarray(self.episode_steps[indices]),
            next_episode_steps=jnp.asarray(self.next_episode_steps[indices]),
        )


def create_private_replays(*, num_owners: int, capacity: int, num_agents: int,
                           observation_dim: int, action_dim: int,
                           model_state_dim: int) -> tuple[PrivateJointReplayBuffer, ...]:
    if num_owners < 1:
        raise ValueError("num_owners must be positive.")
    replays = tuple(PrivateJointReplayBuffer(
        capacity=capacity, num_agents=num_agents, observation_dim=observation_dim,
        action_dim=action_dim, model_state_dim=model_state_dim,
    ) for _ in range(num_owners))
    if len({id(replay) for replay in replays}) != num_owners:
        raise RuntimeError("Replay construction accidentally shared an object.")
    return replays
