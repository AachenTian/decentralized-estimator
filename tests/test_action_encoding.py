from __future__ import annotations

import jax.numpy as jnp

from sender_marl.envs.jaxmarl_simple_spread import (
    JaxMARLSimpleSpreadAdapter,
)


def test_force_2d_to_mpe_5d() -> None:
    actions = jnp.array(
        [
            [1.0, -0.5],
            [-0.25, 0.75],
        ],
        dtype=jnp.float32,
    )
    encoded = JaxMARLSimpleSpreadAdapter._force_2d_to_mpe_5d(actions)
    expected = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.5, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.75],
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(encoded, expected)
