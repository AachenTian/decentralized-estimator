from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from omegaconf import OmegaConf

from aorpo.agents.model_dynamics import (
    init_model,
    predict_next,
    manual_unflatten_state,
    get_obs,
)
from aorpo.agents.policy import init_policy_model, PolicyNet
from aorpo.envs.jaxmarl_simple_spread_v3_env_wrapper import make_mpe_env
from aorpo.utils.replay import manual_flatten_state


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = ROOT / "exported_ckpts" / "final_execution_ckpt.pkl"
OUTPUT_DIR = ROOT / "demo_outputs"


# ============================================================
# Load trained checkpoint
# ============================================================

def load_models(ckpt_path: Path):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    cfg = OmegaConf.create(ckpt["cfg"])
    meta = ckpt["meta"]

    num_agents = int(meta["num_agents"])
    num_opponents = int(meta["num_opponents"])
    num_landmarks = int(cfg.train.num_landmark)

    obs_dim = int(meta["obs_dim"])
    act_dim = int(meta["act_dim"])
    opp_dim = int(meta["opp_dim"])

    # 当前 model_dynamics.py 里的 state parser 默认写死为 3 agents + 3 landmarks
    if num_agents != 3 or num_landmarks != 3:
        raise ValueError(
            "当前 demo 假设 num_agents=3 且 num_landmark=3。"
            f"现在得到 num_agents={num_agents}, num_landmark={num_landmarks}"
        )

    rng = jax.random.PRNGKey(0)

    # ---------- Rebuild dynamics and reward ----------
    rng, model_key = jax.random.split(rng)

    _, _, transition_state, reward_state = init_model(
        model_key,
        num_agents,
        act_dim,
        opp_dim,
        cfg,
    )

    transition_state = transition_state.replace(
        params=ckpt["transition_params"]
    )

    reward_state = reward_state.replace(
        params=ckpt["reward_params"]
    )

    # ---------- Rebuild ego policy ----------
    rng, ego_key = jax.random.split(rng)

    _, policy_state = init_policy_model(
        ego_key,
        obs_dim,
        act_dim,
        cfg.policy,
        "agent_0",
    )

    policy_state = policy_state.replace(
        params=ckpt["ego_policy_params"]
    )

    # ---------- Rebuild real opponent policies ----------
    real_opponent_states = []

    for i in range(num_opponents):
        agent_name = f"agent_{i + 1}"

        rng, opp_key = jax.random.split(rng)

        _, opponent_state = init_policy_model(
            opp_key,
            obs_dim,
            act_dim,
            cfg.policy,
            agent_name,
        )

        opponent_state = opponent_state.replace(
            params=ckpt["opponent_policy_params"][agent_name]
        )

        real_opponent_states.append(opponent_state)

    return {
        "cfg": cfg,
        "std": ckpt["std"],
        "transition_state": transition_state,
        "reward_state": reward_state,
        "policy_state": policy_state,
        "real_opponent_states": real_opponent_states,
        "num_agents": num_agents,
        "num_opponents": num_opponents,
        "num_landmarks": num_landmarks,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "opp_dim": opp_dim,
    }


# ============================================================
# Policy actions
# ============================================================

def sample_all_actions(policy_state, opponent_states, obs, rng):
    """
    输入:
        obs:
            real env rollout 时: (obs_dim,)
            model rollout 时:    (1, obs_dim)

    输出:
        ego_action
        opponent_actions: list，每个 agent 一个 action
        updated rng
    """

    ego_action, _, rng = PolicyNet.sample_action(
        policy_state.params,
        policy_state.apply_fn,
        rng,
        obs["agent_0"],
    )

    opponent_actions = []

    for i, opponent_state in enumerate(opponent_states):
        action_i, _, rng = PolicyNet.sample_action(
            opponent_state.params,
            opponent_state.apply_fn,
            rng,
            obs[f"agent_{i + 1}"],
        )
        opponent_actions.append(action_i)

    return ego_action, opponent_actions, rng


def to_env_action(action):
    """
    把 (5,) 或 (1, 5) 统一变为环境需要的 (5,)。
    """
    return jnp.asarray(action).reshape(-1)


def to_model_action(action):
    """
    dynamics model 需要 batch 维：
    (5,) -> (1, 5)
    (1, 5) -> (1, 5)
    """
    return jnp.atleast_2d(jnp.asarray(action))


# ============================================================
# entropy of aleatoric and epistemic uncertainty
# ============================================================
def dynamics_total_uncertainty(mu, logvar):
    eps = 1e-6

    # 每个 ensemble member 的 aleatoric variance
    var_e = jnp.exp(logvar)

    # covariance-intersection 风格融合
    prec_e = 1.0 / (var_e + eps)
    aleatoric_var = 1.0 / jnp.mean(prec_e, axis=0)

    # precision-weighted mean
    mu_bar = aleatoric_var * jnp.mean(prec_e * mu, axis=0)

    # ensemble disagreement
    epistemic_var = jnp.mean(
        (mu - mu_bar[None, ...]) ** 2,
        axis=0,
    )

    # 你需要的总不确定性
    total_var = aleatoric_var + epistemic_var

    # 可选：由总方差得到 predictive entropy
    total_entropy_dim = 0.5 * jnp.log(
        2.0 * jnp.pi * jnp.e * (total_var + eps)
    )

    total_entropy = jnp.sum(total_entropy_dim, axis=-1)

    return (
        total_var,
        aleatoric_var,
        epistemic_var,
        total_entropy_dim,
        total_entropy,
    )


# ============================================================
# Trajectory utilities
# ============================================================

def create_trajectory():
    return {
        "agents": [],
        "landmarks": [],
        "rewards": [],
        "agent_0_rewards": [],
        "ego_actions": [],
        "opp_actions": [],
        "flat_states": [],
    }


def append_frame(
    traj,
    p_pos,
    reward_vec,
    ego_action,
    opp_action,
    flat_state,
    num_agents,
    num_landmarks,
):
    """
    p_pos:
        (6, 2) = 3 agents + 3 landmarks

    reward_vec:
        (3,)
    """

    p_pos = np.asarray(jax.device_get(p_pos))
    reward_vec = np.asarray(jax.device_get(reward_vec), dtype=np.float32)

    traj["agents"].append(
        p_pos[:num_agents].copy()
    )

    traj["landmarks"].append(
        p_pos[num_agents:num_agents + num_landmarks].copy()
    )

    # 用全体 agent reward 之和当作 team reward
    traj["rewards"].append(
        float(np.sum(reward_vec))
    )

    traj["agent_0_rewards"].append(
        float(reward_vec[0])
    )

    traj["ego_actions"].append(
        np.asarray(jax.device_get(ego_action)).reshape(-1).copy()
    )

    traj["opp_actions"].append(
        np.asarray(jax.device_get(opp_action)).reshape(-1).copy()
    )

    traj["flat_states"].append(
        np.asarray(jax.device_get(flat_state)).reshape(-1).copy()
    )


def finalize_trajectory(traj):
    return {
        "agents": np.asarray(traj["agents"], dtype=np.float32),
        "landmarks": np.asarray(traj["landmarks"], dtype=np.float32),
        "rewards": np.asarray(traj["rewards"], dtype=np.float32),
        "agent_0_rewards": np.asarray(
                                        traj["agent_0_rewards"],
                                        dtype=np.float32,
                                    ),
        "ego_actions": np.asarray(traj["ego_actions"], dtype=np.float32),
        "opp_actions": np.asarray(traj["opp_actions"], dtype=np.float32),
        "flat_states": np.asarray(traj["flat_states"], dtype=np.float32),
    }


def save_trajectory(traj, save_path: Path):
    np.savez_compressed(save_path, **traj)
    print(f"✅ Trajectory saved: {save_path}")


# ============================================================
# Group A: real environment rollout
# ============================================================

def rollout_real_env(
    env,
    initial_state,
    initial_obs,
    models,
    horizon,
    policy_rng,
    env_rng,
):
    """
    每一步:
        true obs_t
        -> policies
        -> env.step
        -> true state_{t+1}, true obs_{t+1}
    """

    state = initial_state
    obs = initial_obs

    traj = create_trajectory()

    for step in range(horizon):
        ego_action, opponent_actions, policy_rng = sample_all_actions(
            models["policy_state"],
            models["real_opponent_states"],
            obs,
            policy_rng,
        )

        # 变成 env 所需的 action dict
        all_actions = [to_env_action(ego_action)]
        all_actions += [to_env_action(a) for a in opponent_actions]

        actions_dict = {
            agent_name: all_actions[i]
            for i, agent_name in enumerate(env.agents)
        }

        env_rng, step_key = jax.random.split(env_rng)

        next_obs, next_state, rewards, dones, _ = env.step(
            step_key,
            state,
            actions_dict,
        )

        reward_vec = jnp.asarray([
            rewards[f"agent_{i}"]
            for i in range(models["num_agents"])
        ])

        opp_action_flat = jnp.concatenate(
            [to_env_action(a) for a in opponent_actions],
            axis=-1,
        )

        flat_next_state = manual_flatten_state(next_state)

        append_frame(
            traj=traj,
            p_pos=next_state.p_pos,
            reward_vec=reward_vec,
            ego_action=to_env_action(ego_action),
            opp_action=opp_action_flat,
            flat_state=flat_next_state,
            num_agents=models["num_agents"],
            num_landmarks=models["num_landmarks"],
        )

        state = next_state
        obs = next_obs

        done_vec = jnp.asarray([
            dones[f"agent_{i}"]
            for i in range(models["num_agents"])
        ])

        if bool(jnp.all(done_vec)):
            print(f"Real environment finished at step {step + 1}.")
            break

    return finalize_trajectory(traj)


# ============================================================
# Group B: dynamics model rollout
# ============================================================

def rollout_dynamics_model(
    initial_state,
    initial_obs,
    models,
    horizon,
    policy_rng,
    model_rng,
):
    """
    只使用真实环境的 state_0 / obs_0。

    t >= 1 后:
        不调用 env.step()
        next_state / next_obs 全部来自 dynamics model
    """

    # 真实 State -> flatten -> 加 batch 维
    flat_state = manual_flatten_state(initial_state)[None, :]

    # 从真实 state_0 重建 model observation，验证它与 env obs_0 一致
    initial_model_state = manual_unflatten_state(flat_state)
    obs = get_obs(initial_model_state)

    print("\n===== Initial observation verification =====")

    for i in range(models["num_agents"]):
        err = jnp.max(
            jnp.abs(
                obs[f"agent_{i}"][0]
                - initial_obs[f"agent_{i}"]
            )
        )

        print(f"agent_{i} observation max error: {float(err):.8f}")

        if float(err) > 1e-5:
            raise RuntimeError(
                f"agent_{i} 的 reconstructed observation 和真实 env observation 不一致。"
            )

    traj = create_trajectory()

    for step in range(horizon):
        ego_action, opponent_actions, policy_rng = sample_all_actions(
            models["policy_state"],
            models["real_opponent_states"],
            obs,
            policy_rng,
        )

        # dynamics input 需要 batch 维
        ego_action_batch = to_model_action(ego_action)

        opponent_action_batches = [
            to_model_action(a)
            for a in opponent_actions
        ]

        opp_action_batch = jnp.concatenate(
            opponent_action_batches,
            axis=-1,
        )

        model_rng, prediction_key = jax.random.split(model_rng)

        # next_state / next_obs / reward 完全由 model 产生
        next_flat_state, next_obs, reward_dict, dones_dict, mu, logvar = predict_next(
            transition_state=models["transition_state"],
            reward_state=models["reward_state"],
            std=models["std"],
            state_agent=flat_state,
            a_ego=ego_action_batch,
            a_opp=opp_action_batch,
            cfg=models["cfg"],
            rng=prediction_key,
            deterministic=True,
            member_idx=0,
        )

        total_var, aleatoric_var, epistemic_var, total_entropy_dim, total_entropy = dynamics_total_uncertainty(mu, logvar)

        reward_vec = jnp.concatenate(
            [
                reward_dict[f"agent_{i}"]
                for i in range(models["num_agents"])
            ],
            axis=-1,
        )[0]

        restored_next_state = manual_unflatten_state(next_flat_state)

        append_frame(
            traj=traj,
            p_pos=restored_next_state.p_pos[0],
            reward_vec=reward_vec,
            ego_action=ego_action_batch[0],
            opp_action=opp_action_batch[0],
            flat_state=next_flat_state[0],
            num_agents=models["num_agents"],
            num_landmarks=models["num_landmarks"],
        )

        flat_state = next_flat_state
        obs = next_obs

        done_vec = jnp.concatenate(
            [
                dones_dict[f"agent_{i}"]
                for i in range(models["num_agents"])
            ],
            axis=-1,
        )[0]

        if bool(jnp.all(done_vec)):
            print(f"Dynamics rollout finished at step {step + 1}.")
            break

    return finalize_trajectory(traj)


# ============================================================
# Side-by-side animation
# ============================================================

def animate_comparison(real_traj, model_traj, save_path: Path):
    """
    左边: cumulative team reward
    中间: real environment trajectory
    右边: model rollout trajectory
    """

    T = min(
        len(real_traj["agents"]),
        len(model_traj["agents"]),
    )

    if T == 0:
        raise RuntimeError("Trajectory is empty; cannot create animation.")

    real_agents = real_traj["agents"][:T]
    real_landmarks = real_traj["landmarks"][:T]
    real_rewards = real_traj["rewards"][:T]

    model_agents = model_traj["agents"][:T]
    model_landmarks = model_traj["landmarks"][:T]
    model_rewards = model_traj["rewards"][:T]

    real_cumulative_reward = np.cumsum(real_rewards)
    model_cumulative_reward = np.cumsum(model_rewards)

    all_positions = np.concatenate(
        [
            real_agents.reshape(-1, 2),
            real_landmarks.reshape(-1, 2),
            model_agents.reshape(-1, 2),
            model_landmarks.reshape(-1, 2),
        ],
        axis=0,
    )

    axis_limit = max(
        1.5,
        float(np.max(np.abs(all_positions))) + 0.15,
    )

    fig, (ax_reward, ax_real, ax_model) = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    # ---------- reward panel ----------
    ax_reward.set_title("Cumulative Team Reward")
    ax_reward.set_xlabel("Step")
    ax_reward.set_ylabel("Reward")
    ax_reward.set_xlim(0, max(T - 1, 1))

    reward_min = min(
        float(np.min(real_cumulative_reward)),
        float(np.min(model_cumulative_reward)),
    )
    reward_max = max(
        float(np.max(real_cumulative_reward)),
        float(np.max(model_cumulative_reward)),
    )

    margin = max(1.0, 0.1 * (reward_max - reward_min + 1e-6))

    ax_reward.set_ylim(
        reward_min - margin,
        reward_max + margin,
    )

    line_real, = ax_reward.plot(
        [],
        [],
        label="Real environment",
    )

    line_model, = ax_reward.plot(
        [],
        [],
        label="Dynamics model",
    )

    ax_reward.legend()

    # ---------- environment panels ----------
    for ax, title in [
        (ax_real, "Group A: Real Environment"),
        (ax_model, "Group B: Dynamics Rollout"),
    ]:
        ax.set_title(title)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal")

    real_agents_scatter = ax_real.scatter([], [], s=80)
    real_landmarks_scatter = ax_real.scatter(
        [],
        [],
        s=120,
        marker="X",
        color="black",
    )

    model_agents_scatter = ax_model.scatter([], [], s=80)
    model_landmarks_scatter = ax_model.scatter(
        [],
        [],
        s=120,
        marker="X",
        color="black",
    )

    agent_colors = ["red", "blue", "green"]

    def update(frame):
        steps = np.arange(frame + 1)

        line_real.set_data(
            steps,
            real_cumulative_reward[:frame + 1],
        )

        line_model.set_data(
            steps,
            model_cumulative_reward[:frame + 1],
        )

        real_agents_scatter.set_offsets(
            real_agents[frame]
        )

        real_agents_scatter.set_color(
            agent_colors
        )

        real_landmarks_scatter.set_offsets(
            real_landmarks[frame]
        )

        model_agents_scatter.set_offsets(
            model_agents[frame]
        )

        model_agents_scatter.set_color(
            agent_colors
        )

        model_landmarks_scatter.set_offsets(
            model_landmarks[frame]
        )

        return (
            line_real,
            line_model,
            real_agents_scatter,
            real_landmarks_scatter,
            model_agents_scatter,
            model_landmarks_scatter,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=150,
        blit=False,
    )

    ani.save(
        save_path,
        fps=10,
        dpi=120,
    )

    plt.close(fig)

    print(f"✅ Comparison animation saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = load_models(CKPT_PATH)

    env = make_mpe_env(models["cfg"])

    # 两组必须从同一个真实初始状态开始
    reset_key = jax.random.PRNGKey(40)
    initial_obs, initial_state = env.reset(reset_key)

    horizon = min(
        50,
        int(models["cfg"].train.max_steps),
    )

    print("\n===== Demo settings =====")
    print("horizon:", horizon)
    print("num_agents:", models["num_agents"])
    print("num_opponents:", models["num_opponents"])
    print("obs_dim:", models["obs_dim"])
    print("act_dim:", models["act_dim"])
    print("opp_dim:", models["opp_dim"])

    # 使用相同的 policy RNG，使两组初始动作一致；
    # 后续动作不同则是因为 model state 与真实 state 开始偏离。
    policy_seed = 123

    real_traj = rollout_real_env(
        env=env,
        initial_state=initial_state,
        initial_obs=initial_obs,
        models=models,
        horizon=horizon,
        policy_rng=jax.random.PRNGKey(policy_seed),
        env_rng=jax.random.PRNGKey(999),
    )

    model_traj = rollout_dynamics_model(
        initial_state=initial_state,
        initial_obs=initial_obs,
        models=models,
        horizon=horizon,
        policy_rng=jax.random.PRNGKey(policy_seed),
        model_rng=jax.random.PRNGKey(999),
    )

    # ---------- Save trajectories ----------
    save_trajectory(
        real_traj,
        OUTPUT_DIR / "real_env_traj.npz",
    )

    save_trajectory(
        model_traj,
        OUTPUT_DIR / "dynamics_model_traj.npz",
    )

    # ---------- Compare metric ----------
    common_T = min(
        len(real_traj["agents"]),
        len(model_traj["agents"]),
    )

    position_rmse = np.sqrt(
        np.mean(
            (
                real_traj["agents"][:common_T]
                - model_traj["agents"][:common_T]
            ) ** 2
        )
    )

    print("\n===== Demo summary =====")
    print(
        "real cumulative team reward:",
        float(np.sum(real_traj["rewards"])),
    )

    print(
        "model cumulative predicted reward:",
        float(np.sum(model_traj["rewards"])),
    )

    print(
        "agent position RMSE:",
        float(position_rmse),
    )

    # ---------- Animation ----------
    animate_comparison(
        real_traj,
        model_traj,
        OUTPUT_DIR / "real_vs_dynamics.mp4",
    )


if __name__ == "__main__":
    main()