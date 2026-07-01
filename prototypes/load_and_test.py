# prototypes/load_and_test.py

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from aorpo.agents.model_dynamics import init_model
from aorpo.agents.policy import init_policy_model, PolicyNet


ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = ROOT / "exported_ckpts" / "final_execution_ckpt.pkl"


def main():
    with open(CKPT_PATH, "rb") as f:
        ckpt = pickle.load(f)

    cfg = OmegaConf.create(ckpt["cfg"])
    meta = ckpt["meta"]

    num_agents = meta["num_agents"]
    num_opponents = meta["num_opponents"]
    obs_dim = meta["obs_dim"]
    act_dim = meta["act_dim"]
    opp_dim = meta["opp_dim"]

    rng = jax.random.PRNGKey(0)

    # 重建 dynamics 与 reward 的网络结构
    rng, km = jax.random.split(rng)
    _, _, transition_state, reward_state = init_model(
        km,
        num_agents,
        act_dim,
        opp_dim,
        cfg,
    )

    # 加载训练后的参数
    transition_state = transition_state.replace(
        params=ckpt["transition_params"]
    )
    reward_state = reward_state.replace(
        params=ckpt["reward_params"]
    )

    # 重建 ego policy
    rng, kp = jax.random.split(rng)
    _, policy_state = init_policy_model(
        kp,
        obs_dim,
        act_dim,
        cfg.policy,
        "agent_0",
    )
    policy_state = policy_state.replace(
        params=ckpt["ego_policy_params"]
    )

    # 重建真实 opponent policies
    real_opponent_states = []

    for i in range(num_opponents):
        agent_name = f"agent_{i + 1}"

        rng, ko = jax.random.split(rng)
        _, opponent_state = init_policy_model(
            ko,
            obs_dim,
            act_dim,
            cfg.policy,
            agent_name,
        )

        opponent_state = opponent_state.replace(
            params=ckpt["opponent_policy_params"][agent_name]
        )
        real_opponent_states.append(opponent_state)

    std = ckpt["std"]

    print("✅ Checkpoint loaded successfully.")
    print("transition step:", ckpt["steps"]["transition_step"])
    print("reward step:", ckpt["steps"]["reward_step"])
    print("ego policy step:", ckpt["steps"]["ego_policy_step"])
    print("number of opponent policies:", len(real_opponent_states))
    print("standardizer:", type(std).__name__)

    fake_obs = jnp.zeros((1, obs_dim))

    # ---- 1. ego policy: obs -> ego action ----
    fake_obs_single = jnp.zeros((obs_dim,))
    ego_action, _, rng = PolicyNet.sample_action(
        policy_state.params,
        policy_state.apply_fn,
        rng,
        fake_obs_single,
    )
    ego_action = jnp.atleast_2d(ego_action)

    print("\n===== Ego policy test =====")
    print("ego action shape:", ego_action.shape)
    print("ego action:", ego_action)

    # ---- 2. opponent policies: obs -> opponent actions ----
    print("\n===== Opponent policy test =====")
    opponent_actions = []
    for i, opponent_state in enumerate(real_opponent_states):
        action_i, _, rng = PolicyNet.sample_action(
            opponent_state.params,
            opponent_state.apply_fn,
            rng,
            fake_obs_single,
        )
        action_i = jnp.atleast_2d(action_i)
        opponent_actions.append(action_i)
        print(f"agent_{i + 1} action shape:", action_i.shape)
        print(f"agent_{i + 1} action:", action_i)

    joint_opp_action = jnp.concatenate(opponent_actions, axis=-1)
    print("opponent action shape:", joint_opp_action.shape)
    print("joint action:", joint_opp_action)

    # ---- 3. dynamics input: core_state + ego action + opp action ----
    fake_core_state = jnp.zeros((1, meta["core_state_dim"]))
    a_ego_n = std.norm_a_ego(ego_action)
    a_opp_n = std.norm_a_opp(joint_opp_action)
    dynamics_input = jnp.concatenate(
        [
            fake_core_state,
            a_ego_n,
            a_opp_n,
        ],
        axis=-1,
    )
    print("\n===== Dynamics input test =====")
    print("dynamics input shape:", dynamics_input.shape)  #expected：(1, 18 + 5 + 10) = (1, 33)

    # ---- 4. transition model: input -> predicted next state ----
    mu, logvar = transition_state.apply_fn(
        {"params": transition_state.params},
        dynamics_input,
    )
    print("\n===== Transition model test =====")
    print("mu:", mu[0])
    print("logvar:", logvar[0])

    # ---- 5. reward model: input -> predicted reward ----
    reward_output = reward_state.apply_fn(
        {"params": reward_state.params},
        dynamics_input,
    )
    print("\n===== Reward model test =====")
    print("reward_output:", reward_output)

if __name__ == "__main__":
    main()




























