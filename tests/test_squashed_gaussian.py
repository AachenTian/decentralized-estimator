import jax
import jax.numpy as jnp

from sender_marl.core.distributions import (
    deterministic_squashed_action,
    sample_squashed_gaussian,
    squashed_gaussian_log_prob,
)


def test_squashed_gaussian_is_bounded_and_finite():
    mean = jnp.zeros((4, 3, 2), dtype=jnp.float32)
    log_std = jnp.full_like(mean, -0.5)
    sample = sample_squashed_gaussian(
        jax.random.PRNGKey(0),
        mean,
        log_std,
    )

    assert sample.action.shape == mean.shape
    assert sample.log_prob.shape == mean.shape[:-1]
    assert bool(jnp.all(sample.action <= 1.0))
    assert bool(jnp.all(sample.action >= -1.0))
    assert bool(jnp.all(jnp.isfinite(sample.log_prob)))

    reconstructed = squashed_gaussian_log_prob(
        sample.action,
        mean,
        log_std,
    )
    assert bool(jnp.allclose(sample.log_prob, reconstructed, atol=2e-4))


def test_deterministic_action_is_tanh_mean():
    mean = jnp.array([[-2.0, 0.0, 2.0]], dtype=jnp.float32)
    assert bool(
        jnp.allclose(
            deterministic_squashed_action(mean),
            jnp.tanh(mean),
        )
    )
