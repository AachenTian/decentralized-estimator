import jax
import jax.numpy as jnp
from jaxmarl import make
import os
import hydra
from omegaconf import DictConfig
from aorpo.visualiztion.make_animation import animate_episode

def make_mpe_env(cfg: DictConfig):
    env = make(cfg.env.ENV_NAME, action_type="Continuous", u_noise=jnp.array([5.00, 5.00, 5.00]),)    #"MPE_simple_v3"
    return env

def env_reset(env, key):
    key_reset = jax.random.PRNGKey(40)
    obs, state = env.reset(key_reset)
    return state, obs, key

def env_step(env, state, a_ego, a_opps, key):
    a_ego = a_ego.reshape(1,-1)
    a_opps = a_opps.reshape(2, -1)
    actions = jnp.concatenate([a_ego, a_opps], axis=0)
    actions = {agent: actions[i] for i,agent in enumerate(env.agents)}
    obs, next_state, rewards, dones, infos = env.step(key, state, actions)
    return next_state, obs, rewards, dones, key


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    rng = jax.random.PRNGKey(cfg.seed)

    env = make_mpe_env(cfg)
    print(f"✅ Env initialized: {cfg.env.ENV_NAME}")
    rng, subrng = jax.random.split(rng,2)
    eval_rng = jax.random.PRNGKey(0)
    state, obs, key = env_reset(env, eval_rng)
    print("obs keys:", obs.keys())
    print("obs:", obs)
    print("state:", state)

    total_reward = 0.
    total_reward_agent_0 = 0.
    step_reward_list = []  # record the step reward in one rollout
    traj_agents = []  # shape (T, 3, 2)
    traj_landmarks = []  # shape (3, 2) (静态)

    # a_ego = jnp.ones((cfg.env.act_dim,))
    # a_opps = jnp.zeros((cfg.train.num_opponents * cfg.env.act_dim))
    # a_ego = jnp.array([-0.9009457, -0.19701889, 0.40034992, -0.6112185,  0.0424891 ]) ##(-0.6537076, -0.59736881)
    a_ego = jnp.array([-0.82266784, -0.18221606, 0.31996903, -1.6457553, 0.05335886]) ##(-0.69911416, -0.50218509)

    a_opps = jnp.array([
        -0.18004934, 0.10922899, 0.38247943, -0.97298497, -0.8712533,
        -0.2582537, 0.2940862, -0.98897564, -0.9850584, 0.343176
    ])

    for i in range(50):
        if i == 0:
            a_ego = a_ego
            a_opps = a_opps
        else:
            a_ego = jnp.array([-0.82266784, -10.18221606, 0.31996903, -10.6457553, 0.05335886])
            a_opps = jnp.array([
                -0.18004934, 0.10922899, 0.38247943, -0.97298497, -0.8712533,
                -0.2582537, 0.2940862, -0.98897564, -0.9850584, 0.343176
            ])
        state, obs, rewards, dones, key = env_step(env, state, a_ego, a_opps, key)
        # print("landmarks:", next_state.p_pos[3:])
        # print("episode done flags:", dones)
        # print("next_state:", next_state.step)
        # print("next_state:",next_state)
        # print("obs:", obs)
        print("rewards:", rewards)
        # print("dones:", dones)
        total_reward_agent_0 += float(rewards["agent_0"])
        total_reward += sum(float(rewards[f"agent_{i}"]) for i in range(3))
        step_reward_list.append(total_reward)

        p = state.p_pos
        traj_agents.append(p[:3])
        traj_landmarks.append(p[3:6])

    traj = {
        "agents": jnp.array(traj_agents),  # (T, 3, 2)
        "landmarks": jnp.array(traj_landmarks),  # (3, 2)
        "rewards": jnp.array(step_reward_list),  # (T,)
    }
    episode_reward_history = [1, 2, 3, 4, 5]
    animate_episode(traj, episode_reward_history, save_path=f"episode_epoch_test.mp4")

if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
