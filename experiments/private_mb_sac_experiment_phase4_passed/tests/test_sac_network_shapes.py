import jax
import jax.numpy as jnp

from private_mb_sac.agents.networks import (
    SACActor,
    TwinQNetwork,
)


def test_actor_and_twin_critic_shapes():
    actor = SACActor(action_dim=2, hidden_dims=(32, 32))
    actor_params = actor.init(
        jax.random.PRNGKey(0),
        jnp.zeros((7, 18), dtype=jnp.float32),
    )["params"]
    mean, log_std = actor.apply(
        {"params": actor_params},
        jnp.zeros((7, 18), dtype=jnp.float32),
    )

    assert mean.shape == (7, 2)
    assert log_std.shape == (7, 2)

    critic = TwinQNetwork(hidden_dims=(32, 32))
    critic_params = critic.init(
        jax.random.PRNGKey(1),
        jnp.zeros((7, 18), dtype=jnp.float32),
        jnp.zeros((7, 3, 2), dtype=jnp.float32),
    )["params"]
    q1, q2 = critic.apply(
        {"params": critic_params},
        jnp.zeros((7, 18), dtype=jnp.float32),
        jnp.zeros((7, 3, 2), dtype=jnp.float32),
    )

    assert q1.shape == (7,)
    assert q2.shape == (7,)
