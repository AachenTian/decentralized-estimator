"""Environment-agnostic JAX MARL building blocks."""

from sender_marl.envs import make_env_adapter, registered_envs

__all__ = ["make_env_adapter", "registered_envs"]
