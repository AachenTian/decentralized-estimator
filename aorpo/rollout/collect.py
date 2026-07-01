# aorpo/rollout/collect.py
import jax
import jax.numpy as jnp
from aorpo.agents.policy import PolicyNet
from aorpo.envs.jaxmarl_simple_spread_v3_env_wrapper import make_mpe_env, env_step, env_reset
from aorpo.agents.model_dynamics import predict_next
from jax.flatten_util import ravel_pytree
from omegaconf import DictConfig

def collect_real_data(policy_fn, opp_fn, obs_dim, act_dim, opp_num, opp_dim, key, cfg: DictConfig):
    """Collect real environment data using JAX scan."""
    def rollout(carry, _):
        state, obs, key, dones = carry
        # ❗ 如果 done=True，则 reset 环境
        state, obs = jax.lax.cond(
            state.step == 25,
            lambda _: env_reset(env, key)[:2],  # reset 给新的 state, obs
            lambda _: (state, obs),
            operand=None
        )
        key, sub1, sub2 = jax.random.split(key, 3)
        a_ego, sub1 = policy_fn(obs, sub1)
        a_opps, sub2 = opp_fn(obs, sub2)
        state2, obs2, r, dones, key = env_step(env, state, a_ego, a_opps, key)
        dones_True = {
            "__all__": jnp.array(True),
            "agent_0": jnp.array(True),
            "agent_1": jnp.array(True),
            "agent_2": jnp.array(True),
        }
        dones = jax.lax.cond(
            state2.step == 25,
            lambda _: dones_True,  # reset 给新的 state, obs
            lambda _: dones,
            operand=None
        )

        # joint_act = jnp.concatenate([a_ego, a_opps], axis=-1)
        return (state2, obs2, key, dones), (state2, obs2, state, obs, a_ego, a_opps, r, dones)

    # 初始化环境状态
    env = make_mpe_env(cfg)
    state, obs, key = env_reset(env, key)
    dones = {
        "__all__": jnp.array(False),
        "agent_0": jnp.array(False),
        "agent_1": jnp.array(False),
        "agent_2": jnp.array(False),
    }
    (final_state, final_obs, _, _), (next_state, next_obs, state, obs, a_ego, a_opp, rew, dones) = jax.lax.scan(
        rollout, (state, obs, key, dones), None, length=cfg.collect.steps_per_epoch
    )
    batch = dict(state= state, obs=obs, a_ego=a_ego, a_opp=a_opp,  next_obs=next_obs, next_state=next_state, rew=rew, dones=dones)
    return batch, final_state, key

def episode_reward(policy_fn, opp_fn, num_agents, key, cfg):
    env = make_mpe_env(cfg)
    state, obs, key = env_reset(env, key)

    total_reward = 0.
    total_reward_agent_0 = 0.
    step_reward_list = []  # record the step reward in one rollout
    traj_agents = []  # shape (T, 3, 2)
    traj_landmarks = []  # shape (3, 2) (静态)

    for t in range(cfg.train.max_steps):
        key, sub1 = jax.random.split(key, 2)
        a_ego, sub1 = policy_fn(obs, sub1)

        key, sub2 = jax.random.split(key, 2)
        a_opp, sub2 = opp_fn(obs, sub2)


        key, sub3 = jax.random.split(key,2)
        state, obs, rewards, dones, sub3 = env_step(env, state, a_ego, a_opp, sub3)
        # if t == 0:
        #     print("action of ego:", a_ego)
        #     print("action of opp:", a_opp)
        #     print("episode_reward_step_0:", rewards)
        total_reward_agent_0 += float(rewards["agent_0"])
        total_reward += sum(float(rewards[f"agent_{i}"]) for i in range(num_agents))
        step_reward_list.append(total_reward)

        p = state.p_pos
        traj_agents.append(p[:3])
        traj_landmarks.append(p[3:6])

        if dones["agent_0"]:
            print("break time of epi_reward:{}", t)
            break

    traj = {
        "agents": jnp.array(traj_agents),  # (T, 3, 2)
        "landmarks": jnp.array(traj_landmarks),  # (3, 2)
        "rewards": jnp.array(step_reward_list),  # (T,)
    }
    return float(total_reward), float(total_reward_agent_0), traj


def rollout_env(policy_fn, opp_fn, init_state, init_obs, key, horizon, cfg: DictConfig):
    def rollout_scan(carry, _):
        state, obs, key_env = carry
        key_env, k1, k2, k3 = jax.random.split(key_env, 4)

        a_ego, k1 = policy_fn(obs, k1)
        a_opp, k2 = opp_fn(obs, k2)

        next_state, next_obs, rewards, _, k3 = env_step(env, state, a_ego, a_opp, k3)
        return (next_state, next_obs, key_env), (next_state, rewards)
    env = make_mpe_env(cfg)
    (final_state, final_obs, _), (state, reward) = jax.lax.scan(rollout_scan, (init_state, init_obs, key), None, length=horizon)
    return state, reward

def rollout_dynamics(policy_fn, opp_fn, transition_state, reward_state, std, init_state, init_obs, key, horizon, cfg):
    def rollout_scan_dyna(carry,_):
        state, obs, key_dyna = carry
        key_dyna, k1, k2, k3 = jax.random.split(key_dyna, 4)

        a_ego, k1 = policy_fn(obs, k1)
        a_opp, k2 = opp_fn(obs, k2)
        a_ego = jnp.expand_dims(a_ego, axis=0)
        a_opp = jnp.expand_dims(a_opp, axis=0)
        next_state, next_obs, reward_dict, dones_dict, _, _ = predict_next(
            transition_state=transition_state,
            reward_state=reward_state,
            std=std,
            state_agent=state,
            a_ego=a_ego,
            a_opp=a_opp,
            cfg=cfg,
            rng=k3,
            deterministic=True,
        )
        next_obs = jax.tree_util.tree_map(lambda x: x.squeeze(0), next_obs)
        return (next_state, next_obs, key_dyna), (next_state, reward_dict)
    (final_state, final_obs, _), (state, reward) = jax.lax.scan(rollout_scan_dyna, (init_state, init_obs, key), None, length=horizon)
    return state, reward

def rollout_compare(policy_fn, opp_fn,  transition_state, reward_state, std, key, horizon, cfg: DictConfig):
    key, k1, k2 = jax.random.split(key, 3)
    env = make_mpe_env(cfg)
    init_state, init_obs, key = env_reset(env, k1)
    state_env, reward_env = rollout_env(policy_fn, opp_fn, init_state, init_obs, k2, horizon, cfg)
    state_t = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bool_ else x,
        init_state
    )
    flat_init_state, _ = ravel_pytree(state_t)
    flat_init_state = jnp.expand_dims(flat_init_state, axis=0)
    state_dyna, reward_dyna = rollout_dynamics(policy_fn, opp_fn,  transition_state, reward_state, std, flat_init_state, init_obs, k2, horizon, cfg)
    return state_env, reward_env, state_dyna, reward_dyna
























