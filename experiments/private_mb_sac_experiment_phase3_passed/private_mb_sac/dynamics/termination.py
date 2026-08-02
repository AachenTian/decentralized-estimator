"""Synthetic rollout termination codes and precedence."""

from __future__ import annotations

import jax.numpy as jnp

TERMINATION_NONE = 0
TERMINATION_HORIZON = 1
TERMINATION_DONE = 2
TERMINATION_COLLISION = 3
TERMINATION_UNCERTAINTY = 4


def step_termination_reason(done, collision, uncertainty):
    """Return codes with precedence: done, collision, uncertainty."""
    reason = jnp.zeros_like(jnp.asarray(done), dtype=jnp.int32)
    reason = jnp.where(uncertainty, TERMINATION_UNCERTAINTY, reason)
    reason = jnp.where(collision, TERMINATION_COLLISION, reason)
    reason = jnp.where(done, TERMINATION_DONE, reason)
    return reason
