import jax.numpy as jnp
import numpy as np
from private_mb_sac.envs.model_state import build_actor_observations_from_model_state, collision_metrics_from_model_state, simple_spread_rewards_from_model_state


def test_model_state_builds_expected_actor_observation_shape_and_order():
    local=jnp.asarray([[0.,0.,1.,2.],[1.,0.,3.,4.],[0.,1.,5.,6.]])
    landmarks=jnp.asarray([[2.,0.],[0.,2.],[-1.,-1.]])
    state=jnp.concatenate([local.reshape(-1),landmarks.reshape(-1)])
    obs=build_actor_observations_from_model_state(state)
    assert obs.shape==(3,18)
    np.testing.assert_allclose(np.asarray(obs[0,:4]),np.asarray(local[0]))
    np.testing.assert_allclose(np.asarray(obs[0,10:14]),[1.,0.,3.,4.])
    np.testing.assert_allclose(np.asarray(obs[0,14:18]),[0.,1.,5.,6.])


def test_reward_and_collision_reconstruction():
    local=jnp.asarray([[0.,0.,0.,0.],[0.1,0.,0.,0.],[1.,1.,0.,0.]])
    landmarks=jnp.asarray([[0.,0.],[1.,1.],[-1.,-1.]])
    state=jnp.concatenate([local.reshape(-1),landmarks.reshape(-1)])
    metrics=collision_metrics_from_model_state(state)
    assert bool(metrics['any_collision'])
    rewards=simple_spread_rewards_from_model_state(state)
    assert rewards.shape==(3,)
    assert rewards[0] < rewards[2]
