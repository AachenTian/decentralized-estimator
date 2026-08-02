import jax.numpy as jnp

from sender_marl.on_policy.gae import compute_gae


def test_gae_cuts_bootstrap_at_terminal_transition():
    rewards = jnp.array([[[1.0]], [[1.0]]])
    values = jnp.zeros_like(rewards)
    dones = jnp.array([[[0.0]], [[1.0]]])
    final_values = jnp.array([[123.0]])

    targets = compute_gae(
        rewards,
        values,
        dones,
        final_values,
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected = jnp.array([[[2.0]], [[1.0]]])
    assert jnp.allclose(targets.advantages, expected)
    assert jnp.allclose(targets.returns, expected)
