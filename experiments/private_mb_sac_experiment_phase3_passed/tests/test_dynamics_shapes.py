import jax
import jax.numpy as jnp

from private_mb_sac.dynamics.ensemble import EnsembleJointDynamics


def test_five_member_dynamics_shapes():
    model = EnsembleJointDynamics(
        ensemble_size=5,
        hidden_dims=(16, 16),
        output_dim=12,
        minimum_logvar=-6.0,
        maximum_logvar=0.5,
    )
    inputs = jnp.zeros((5, 7, 24), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), inputs)["params"]
    mean, logvar = model.apply({"params": params}, inputs)

    assert mean.shape == (5, 7, 12)
    assert logvar.shape == (5, 7, 12)
    assert jnp.all(logvar >= -6.0)
    assert jnp.all(logvar <= 0.5)
