import jax.numpy as jnp
import numpy as np
from private_mb_sac.dynamics.prediction import apply_agent_delta


def test_apply_delta_keeps_landmarks_static():
    state=jnp.arange(18,dtype=jnp.float32)[None,:]
    delta=jnp.ones((1,12),dtype=jnp.float32)
    next_state=apply_agent_delta(state,delta)
    np.testing.assert_allclose(np.asarray(next_state[:,:12]),np.asarray(state[:,:12]+1))
    np.testing.assert_array_equal(np.asarray(next_state[:,12:]),np.asarray(state[:,12:]))
