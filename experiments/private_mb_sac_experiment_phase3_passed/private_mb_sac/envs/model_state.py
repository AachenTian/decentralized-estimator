"""Pure JAX Simple-Spread computations from the 18D model state."""

from __future__ import annotations

import jax.numpy as jnp


def split_model_state(model_states, *, num_agents: int=3, num_landmarks: int=3):
    states = jnp.asarray(model_states, dtype=jnp.float32)
    expected = num_agents * 4 + num_landmarks * 2
    if states.shape[-1] != expected:
        raise ValueError(f"Expected model-state dimension {expected}, got {states.shape[-1]}.")
    local_states = states[..., :num_agents*4].reshape(states.shape[:-1] + (num_agents, 4))
    landmarks = states[..., num_agents*4:].reshape(states.shape[:-1] + (num_landmarks, 2))
    return local_states, landmarks


def compose_model_state(local_states, landmarks):
    local = jnp.asarray(local_states, dtype=jnp.float32)
    land = jnp.asarray(landmarks, dtype=jnp.float32)
    return jnp.concatenate([
        local.reshape(local.shape[:-2] + (-1,)),
        land.reshape(land.shape[:-2] + (-1,)),
    ], axis=-1)


def build_actor_observations_from_model_state(model_states, *, num_agents: int=3,
                                               num_landmarks: int=3):
    local, landmarks = split_model_state(model_states, num_agents=num_agents, num_landmarks=num_landmarks)
    agent_pos = local[..., :, :2]
    agent_vel = local[..., :, 2:4]
    landmark_relative = landmarks[..., None, :, :] - agent_pos[..., :, None, :]
    public_context = landmark_relative.reshape(landmark_relative.shape[:-2] + (2*num_landmarks,))
    relative_pos = agent_pos[..., None, :, :] - agent_pos[..., :, None, :]
    remote_vel = jnp.broadcast_to(agent_vel[..., None, :, :], relative_pos.shape)
    pairwise = jnp.concatenate([relative_pos, remote_vel], axis=-1)
    remote_indices = jnp.stack([
        jnp.concatenate([jnp.arange(num_agents)[:i], jnp.arange(num_agents)[i+1:]])
        for i in range(num_agents)
    ], axis=0)
    ego_indices = jnp.arange(num_agents)[:, None]
    remote = pairwise[..., ego_indices, remote_indices, :]
    remote = remote.reshape(pairwise.shape[:-3] + (num_agents, (num_agents-1)*4))
    return jnp.concatenate([local, public_context, remote], axis=-1)


def collision_metrics_from_model_state(model_states, *, num_agents: int=3,
                                       num_landmarks: int=3,
                                       collision_distance: float=0.30):
    local, _ = split_model_state(model_states, num_agents=num_agents, num_landmarks=num_landmarks)
    positions = local[..., :, :2]
    relative = positions[..., :, None, :] - positions[..., None, :, :]
    distances = jnp.linalg.norm(relative, axis=-1)
    upper = jnp.triu(jnp.ones((num_agents, num_agents), dtype=jnp.bool_), k=1)
    colliding = (distances < collision_distance) & upper
    pair_count = num_agents * (num_agents - 1) / 2
    min_distances = jnp.where(
        jnp.eye(num_agents, dtype=jnp.bool_), jnp.inf, distances
    )
    return {
        'collision_count': jnp.sum(colliding, axis=(-2, -1)),
        'pair_collision_rate': jnp.sum(colliding, axis=(-2, -1)) / pair_count,
        'any_collision': jnp.any(colliding, axis=(-2, -1)),
        'min_pair_distance': jnp.min(min_distances, axis=(-2, -1)),
    }


def simple_spread_rewards_from_model_state(model_states, *, num_agents: int=3,
                                           num_landmarks: int=3,
                                           local_ratio: float=0.5,
                                           collision_distance: float=0.30):
    """Reconstruct JaxMARL Simple-Spread rewards from the resulting state."""
    local, landmarks = split_model_state(model_states, num_agents=num_agents, num_landmarks=num_landmarks)
    positions = local[..., :, :2]
    agent_landmark = positions[..., :, None, :] - landmarks[..., None, :, :]
    distances = jnp.linalg.norm(agent_landmark, axis=-1)
    global_reward = -jnp.sum(jnp.min(distances, axis=-2), axis=-1)

    relative = positions[..., :, None, :] - positions[..., None, :, :]
    pair_distances = jnp.linalg.norm(relative, axis=-1)
    nonself = ~jnp.eye(num_agents, dtype=jnp.bool_)
    collisions = (pair_distances < collision_distance) & nonself
    local_rewards = -jnp.sum(collisions, axis=-1).astype(jnp.float32)
    return local_ratio * local_rewards + (1.0-local_ratio) * global_reward[..., None]
