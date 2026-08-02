import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from private_mb_sac.core.types import DynamicsNormalizer, ReplayBatch
from private_mb_sac.dynamics.ensemble import EnsembleJointDynamics
from private_mb_sac.dynamics.loss import dynamics_train_step


def _normalizer():
    return DynamicsNormalizer(
        state_mean=jnp.zeros((18,), jnp.float32),
        state_std=jnp.ones((18,), jnp.float32),
        action_mean=jnp.zeros((6,), jnp.float32),
        action_std=jnp.ones((6,), jnp.float32),
        delta_mean=jnp.zeros((12,), jnp.float32),
        delta_std=jnp.ones((12,), jnp.float32),
        clip=jnp.asarray(10.0, jnp.float32),
        sample_count=jnp.asarray(16, jnp.int32),
    )


def _batch(batch_size=8):
    states = jnp.zeros((batch_size, 18), jnp.float32)
    next_states = states.at[:, :12].set(0.1)
    return ReplayBatch(
        observations=jnp.zeros((batch_size, 3, 18), jnp.float32),
        actions=jnp.zeros((batch_size, 3, 2), jnp.float32),
        rewards=jnp.zeros((batch_size, 3), jnp.float32),
        next_observations=jnp.zeros((batch_size, 3, 18), jnp.float32),
        dones=jnp.zeros((batch_size, 3), jnp.float32),
        model_states=states,
        next_model_states=next_states,
        pair_collision_rates=jnp.zeros((batch_size,), jnp.float32),
        min_pair_distances=jnp.ones((batch_size,), jnp.float32),
    )


def test_one_dynamics_update_changes_only_updated_state():
    model = EnsembleJointDynamics(
        ensemble_size=5,
        hidden_dims=(16,),
        output_dim=12,
    )
    dummy = jnp.zeros((5, 1, 24), jnp.float32)
    params0 = model.init(jax.random.PRNGKey(0), dummy)["params"]
    params1 = model.init(jax.random.PRNGKey(1), dummy)["params"]
    tx = optax.adam(1.0e-3)
    state0 = TrainState.create(apply_fn=model.apply, params=params0, tx=tx)
    state1 = TrainState.create(apply_fn=model.apply, params=params1, tx=tx)
    state1_before = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), state1.params)

    indices = jnp.tile(jnp.arange(8)[None, :], (5, 1))
    state0_after, metrics = dynamics_train_step(
        state0,
        _normalizer(),
        _batch(),
        indices,
    )

    assert np.isfinite(float(metrics["loss"]))
    assert int(state0_after.step) == 1
    assert int(state1.step) == 0
    for before, after in zip(
        jax.tree_util.tree_leaves(state1_before),
        jax.tree_util.tree_leaves(state1.params),
    ):
        np.testing.assert_array_equal(before, np.asarray(after))
