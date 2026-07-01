# aorpo/envs/jax_toy_env.py
import jax
import jax.numpy as jnp

def env_reset(key, obs_dim):
    key, sub = jax.random.split(key)
    state = 0.5 * jax.random.normal(sub, (obs_dim,), dtype=jnp.float32)
    return state, key

def env_step(state, a_ego, a_opps, key):
    """JAX functional version of ToyMultiAgentEnv.step"""
    obs_dim = state.shape[0]
    a_ego_dim = a_ego.shape[0]
    num_opponents = a_opps.shape[0] / a_ego_dim
    if num_opponents == 1:
        u = a_ego + a_opps
    else:
        a_opps = a_opps.reshape(-1, a_ego_dim)
        u = a_ego + jnp.sum(a_opps, axis=0)[: a_ego.shape[0]]
    key, sub = jax.random.split(key)
    noise = 0.01 * jax.random.normal(sub, (obs_dim,))
    next_state = state + 0.1 * jnp.pad(u, (0, obs_dim - u.shape[0])) + noise
    reward = -jnp.sum(next_state ** 2)
    done = False
    return next_state, reward, done, key
