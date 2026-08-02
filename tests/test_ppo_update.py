import jax
import jax.numpy as jnp

from sender_marl.core.distributions import (
    squashed_gaussian_log_prob_from_pre_tanh,
)
from sender_marl.core.networks import ContinuousActor, LocalValueNetwork
from sender_marl.on_policy.train_state import (
    PPOConfig,
    create_ppo_train_states,
)
from sender_marl.on_policy.types import RolloutBatch, RolloutOutput
from sender_marl.on_policy.update import make_ppo_update


def test_jitted_ppo_update_is_finite_and_changes_parameters():
    t, e, n, obs_dim, action_dim = 4, 2, 3, 7, 2
    actor = ContinuousActor(action_dim=action_dim, hidden_dims=(16, 16))
    value_network = LocalValueNetwork(hidden_dims=(16, 16))
    config = PPOConfig(update_epochs=2, num_minibatches=2)

    key = jax.random.PRNGKey(0)
    key, init_key, data_key, update_key = jax.random.split(key, 4)
    dummy_obs = jnp.zeros((n, obs_dim), dtype=jnp.float32)
    actor_state, value_state = create_ppo_train_states(
        init_key,
        actor,
        value_network,
        dummy_obs,
        dummy_obs,
        config,
    )

    actor_obs = jax.random.normal(data_key, (t, e, n, obs_dim))
    critic_obs = actor_obs
    mean, log_std = actor.apply({"params": actor_state.params}, actor_obs)
    pre_tanh = mean + 0.1
    old_log_probs = squashed_gaussian_log_prob_from_pre_tanh(
        pre_tanh,
        mean,
        log_std,
    )
    old_values = value_network.apply(
        {"params": value_state.params},
        critic_obs,
    )
    rewards = jnp.ones((t, e, n), dtype=jnp.float32) * 0.1
    dones = jnp.zeros((t, e, n), dtype=jnp.bool_).at[-1].set(True)

    batch = RolloutBatch(
        actor_obs=actor_obs,
        critic_obs=critic_obs,
        actions=jnp.tanh(pre_tanh),
        pre_tanh_actions=pre_tanh,
        log_probs=old_log_probs,
        values=old_values,
        rewards=rewards,
        dones=dones,
        episode_dones=jnp.zeros((t, e), dtype=jnp.bool_),
        local_states=jnp.zeros((t, e, n, 4)),
        next_local_states=jnp.zeros((t, e, n, 4)),
        alive=jnp.ones((t, e, n)),
        metrics={},
    )
    rollout = RolloutOutput(
        batch=batch,
        final_env_state=None,
        final_features=None,
        final_common_state_source_state=None,
        final_actor_obs=jnp.zeros((e, n, obs_dim)),
        final_critic_obs=jnp.zeros((e, n, obs_dim)),
        final_values=jnp.zeros((e, n)),
    )

    old_actor_params = actor_state.params
    output = make_ppo_update(config, jit=True)(
        update_key,
        actor_state,
        value_state,
        rollout,
    )

    for value in output.metrics.values():
        assert jnp.all(jnp.isfinite(value))

    changed = any(
        not bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(old_actor_params),
            jax.tree_util.tree_leaves(output.actor_state.params),
        )
    )
    assert changed
