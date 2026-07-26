import jax.numpy as jnp

from sender_marl.envs.jaxmarl_simple_spread import (
    JaxMARLSimpleSpreadAdapter,
    SimpleSpreadConfig,
)


def test_collision_metrics_count_unordered_pairs_once():
    adapter = JaxMARLSimpleSpreadAdapter(
        SimpleSpreadConfig(
            num_agents=3,
            num_landmarks=3,
            collision_distance=0.30,
        )
    )

    # Agents 0 and 1 collide; agent 2 is far away.
    local_states = jnp.array(
        [
            [0.00, 0.00, 0.0, 0.0],
            [0.20, 0.00, 0.0, 0.0],
            [1.00, 1.00, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    metrics = adapter.compute_collision_metrics(local_states)

    assert int(metrics["collision_count"]) == 1
    assert bool(metrics["any_collision"])
    assert jnp.isclose(metrics["pair_collision_rate"], 1.0 / 3.0)
    assert jnp.isclose(metrics["min_pair_distance"], 0.20)
