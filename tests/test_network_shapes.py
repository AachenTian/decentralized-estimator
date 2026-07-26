import jax
import jax.numpy as jnp

from sender_marl.core.networks import (
    CentralizedValueNetwork,
    ContinuousActor,
    LocalValueNetwork,
)


def test_networks_preserve_leading_dimensions():
    observations = jnp.zeros((5, 3, 18), dtype=jnp.float32)
    actor = ContinuousActor(action_dim=2, hidden_dims=(32, 32))
    local_value = LocalValueNetwork(hidden_dims=(32, 32))
    central_value = CentralizedValueNetwork(hidden_dims=(32, 32))

    actor_variables = actor.init(jax.random.PRNGKey(0), observations)
    mean, log_std = actor.apply(actor_variables, observations)
    local_variables = local_value.init(jax.random.PRNGKey(1), observations)
    local_values = local_value.apply(local_variables, observations)
    central_variables = central_value.init(jax.random.PRNGKey(2), observations)
    central_values = central_value.apply(central_variables, observations)

    assert mean.shape == (5, 3, 2)
    assert log_std.shape == (5, 3, 2)
    assert local_values.shape == (5, 3)
    assert central_values.shape == (5, 3)
