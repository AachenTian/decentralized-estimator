from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sender_marl.envs.base import MultiAgentEnvAdapter


EnvFactory = Callable[..., MultiAgentEnvAdapter]
_ENV_REGISTRY: dict[str, EnvFactory] = {}


def register_env(name: str) -> Callable[[EnvFactory], EnvFactory]:
    """Register an environment adapter factory under a stable config name."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Environment name cannot be empty.")

    def decorator(factory: EnvFactory) -> EnvFactory:
        if normalized in _ENV_REGISTRY:
            raise ValueError(f"Environment {normalized!r} is already registered.")
        _ENV_REGISTRY[normalized] = factory
        return factory

    return decorator


def make_env_adapter(name: str, **kwargs: Any) -> MultiAgentEnvAdapter:
    normalized = name.strip().lower()
    try:
        factory = _ENV_REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_ENV_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown environment {name!r}. Registered environments: {available}."
        ) from exc
    return factory(**kwargs)


def registered_envs() -> tuple[str, ...]:
    return tuple(sorted(_ENV_REGISTRY))
