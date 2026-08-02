"""Factory and shape validation for the existing Simple Spread adapter."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def make_private_mb_sac_adapter(config: Any):
    """Create the existing project adapter; do not duplicate environment code."""
    try:
        from sender_marl.envs import make_env_adapter
    except ImportError as exc:
        raise RuntimeError(
            "Could not import sender_marl. Set PYTHONPATH to include the original "
            "project's `src` directory."
        ) from exc

    adapter = make_env_adapter(
        "jaxmarl_simple_spread",
        num_agents=int(config.experiment.num_agents),
        num_landmarks=int(config.experiment.num_landmarks),
        max_steps=int(config.experiment.max_steps),
        action_mode="force_2d",
        remote_feature_mode="relative_state",
    )
    validate_adapter_spec(adapter, config)
    return adapter


def validate_adapter_spec(adapter: Any, config: Any) -> None:
    expected = {
        "num_agents": int(config.experiment.num_agents),
        "actor_obs_dim": int(config.actor.observation_dim),
        "model_state_dim": int(config.critic.state_dim),
        "policy_action_dim": int(config.actor.action_dim),
    }
    actual = {
        "num_agents": int(adapter.spec.num_agents),
        "actor_obs_dim": int(adapter.spec.actor_obs_dim),
        "model_state_dim": int(adapter.spec.model_state_dim),
        "policy_action_dim": int(adapter.spec.policy_action_dim),
    }
    if actual != expected:
        raise ValueError(
            f"Environment specification mismatch: expected={expected}, actual={actual}."
        )


def validate_reset_shapes(adapter: Any, env_state: Any, features: Any) -> None:
    del env_state
    observations = adapter.build_actor_observations(
        features,
        features.local_states,
    )
    expected_obs = (adapter.spec.num_agents, adapter.spec.actor_obs_dim)
    if observations.shape != expected_obs:
        raise ValueError(
            f"Actor observation shape must be {expected_obs}, got {observations.shape}."
        )
    if features.model_state.shape != (adapter.spec.model_state_dim,):
        raise ValueError(
            "Model state shape mismatch: "
            f"expected {(adapter.spec.model_state_dim,)}, "
            f"got {features.model_state.shape}."
        )


def identity_actor_normalizer(adapter: Any, *, clip: float = 10.0):
    """Identity normalizer used only until calibration is implemented."""
    from private_mb_sac.core.types import ObservationNormalizer

    dimension = adapter.spec.actor_obs_dim
    return ObservationNormalizer(
        mean=jnp.zeros((dimension,), dtype=jnp.float32),
        std=jnp.ones((dimension,), dtype=jnp.float32),
        clip=jnp.asarray(float(clip), dtype=jnp.float32),
    )
