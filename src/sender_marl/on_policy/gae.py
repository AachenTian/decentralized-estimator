from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


Array = jax.Array


@struct.dataclass
class AdvantageTargets:
    """Generalized-advantage estimates and value-function targets."""

    advantages: Array
    returns: Array


def compute_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    final_values: Array,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> AdvantageTargets:
    """Compute time-major generalized advantage estimation.

    Expected shapes:
        rewards, values, dones: (T, ...)
        final_values:           (...)

    ``dones[t]`` marks whether transition ``t`` ends the episode. Therefore
    both bootstrapping and recursive advantage propagation are cut at that
    transition.
    """

    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    dones = jnp.asarray(dones, dtype=jnp.float32)
    final_values = jnp.asarray(final_values, dtype=jnp.float32)

    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            "rewards, values, and dones must have identical shapes; got "
            f"{rewards.shape}, {values.shape}, and {dones.shape}."
        )
    if rewards.shape[1:] != final_values.shape:
        raise ValueError(
            "final_values must match the non-time dimensions of rewards; got "
            f"rewards={rewards.shape}, final_values={final_values.shape}."
        )

    def reverse_step(carry, transition):
        next_advantage, next_value = carry
        reward, value, done = transition
        nonterminal = 1.0 - done
        delta = reward + gamma * nonterminal * next_value - value
        advantage = (
            delta
            + gamma * gae_lambda * nonterminal * next_advantage
        )
        return (advantage, value), advantage

    initial_carry = (
        jnp.zeros_like(final_values),
        final_values,
    )
    (_, _), advantages = jax.lax.scan(
        reverse_step,
        initial_carry,
        (rewards, values, dones),
        reverse=True,
    )
    returns = advantages + values
    return AdvantageTargets(advantages=advantages, returns=returns)
