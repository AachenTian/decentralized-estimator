# aorpo/train.py
from __future__ import annotations
import os
import pickle
import jax
import jax.numpy as jnp
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb, random
import copy

# ===== 你项目里的模块 =====
from aorpo.utils.replay import ReplayBuffer, manual_flatten_dict
from aorpo.utils.export_checkpoint import save_execution_checkpoint
from aorpo.rollout.collect import collect_real_data, episode_reward, rollout_compare
from aorpo.rollout.rollout import rollout_model, compute_rollout_lengths


from aorpo.agents.policy import init_policy_model, PolicyNet, init_policy_ensemble
from aorpo.agents.q_function import init_q_function

from aorpo.agents.update_q_function import update_q_function, evaluate_fixed_q_loss
from aorpo.agents.update_policy import update_policy, update_opponent_policy
from aorpo.agents.update_opponents_model import update_opponent_model
from aorpo.visualiztion.make_animation import animate_episode

from aorpo.agents.model_dynamics import (
    init_model,
    train_transition_step,
    train_reward_step,
    StandardizerRS,
    Standardizer
)


wandb.login()
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="yachen-tian-rwth-aachen-university",
    # Set the wandb project where this run will be logged.
    project="AORPO-dynamics model",
    # mode="offline",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 3e-4,
        "architecture": "AORPO",
        "Environment": "mpe_spread_v3",
        "epochs": 10,
    },
)

# -------------------------------------------------
# 辅助：软更新 target Q
# -------------------------------------------------
def soft_update(target_state, source_state, tau: float):
    new_params = jax.tree_util.tree_map(
        lambda t, s: (1.0 - tau) * t + tau * s, target_state.params, source_state.params
    )
    return target_state.replace(params=new_params)

# -------------------------------------------------
# 辅助：把一批 dict(jnp arrays) 加入 replay
# -------------------------------------------------
def add_batch_env_to_replay(replay: ReplayBuffer, batch: dict, cfg:DictConfig) -> ReplayBuffer:
    return replay.add_batch(batch, cfg)


# -------------------------------------------------
# JAX 风格 policy / opponent 的“可调用函数”（供 collect 使用）
#   collect_real_data(policy_fn, opp_fn, ...) 期望：
#   - policy_fn(s, key) -> ego 动作 a_i
#   - opp_fn(s, key)    -> 拼好的对手动作向量 a_-i
# -------------------------------------------------
def make_policy_fn(policy_state):
    def policy_fn(obs, key):
        act, _, new_key = PolicyNet.sample_action(
            policy_state.params,
            policy_state.apply_fn,
            key,
            obs["agent_0"],
        )
        return act, new_key
    return policy_fn



def make_opp_fn(opponent_states):
    def opp_fn(obs, key):
        acts = []
        key, sub = jax.random.split(key)
        for i, state in enumerate(opponent_states):
            a_j, _, sub = PolicyNet.sample_action(
                state.params,
                state.apply_fn,
                sub,
                obs[f"agent_{i+1}"]
            )
            acts.append(a_j)
        return jnp.concatenate(acts, -1), sub
    return opp_fn

# -------------------------------------------------
# 主训练流程（Hydra）
# -------------------------------------------------
@hydra.main(config_path="aorpo/configs", config_name="train", version_base=None)
def main(cfg: DictConfig):

    print("\n===== Config =====")
    print(OmegaConf.to_yaml(cfg))

    rng = jax.random.PRNGKey(cfg.seed)

    # 维度
    state_dim = cfg.env.state_dim
    num_opponents = cfg.train.num_opponents
    num_agents = num_opponents + 1
    obs_dim = cfg.env.obs_dim
    act_dim = cfg.env.act_dim
    opp_num = getattr(cfg.train, "num_opponents", 0)
    opp_dim = act_dim * opp_num  # 简单假设每个对手动作维度与 ego 相同



    # --- Replay Buffers
    replay_env = ReplayBuffer.create(cfg.replay.capacity, obs_dim, act_dim, opp_num, state_dim)
    replay_model = ReplayBuffer.create(cfg.replay.capacity, obs_dim, act_dim, opp_num, state_dim)
    replay_env_fix = ReplayBuffer.create(cfg.replay.capacity, obs_dim, act_dim, opp_num, state_dim)


    # --- 初始化网络
    rng, k1 = jax.random.split(rng)
    rng, k11 = jax.random.split(rng)
    _, policy_state = init_policy_model(k1, obs_dim, act_dim, cfg.policy, "agent_0")

    rng, kq1 = jax.random.split(rng)
    q1_net, q1_state = init_q_function(kq1, state_dim, act_dim, cfg.q_function)
    rng, kq2 = jax.random.split(rng)
    q2_net, q2_state = init_q_function(kq2, state_dim, act_dim, cfg.q_function)

    # target Q
    _, target_q1_state = init_q_function(kq1, state_dim, act_dim, cfg.q_function)
    _, target_q2_state = init_q_function(kq2, state_dim, act_dim, cfg.q_function)

    # dynamics model
    rng, km = jax.random.split(rng)
    transition_model, reward_model, transition_state, reward_state = init_model(
        km, num_agents, act_dim, opp_dim, cfg
    )

    # opponent
    opponent_states = []
    for i in range(opp_num):
        rng, ko = jax.random.split(rng)
        j = i+1
        _, opp_state = init_policy_ensemble(ko, obs_dim, act_dim, cfg.policy, f"agent_{j}")
        opponent_states.append(opp_state)

    # real opponent
    real_opponent_states = []
    for i in range(opp_num):
        rng, ko = jax.random.split(rng)
        _, real_opp_state = init_policy_model(ko, obs_dim, act_dim, cfg.policy, f"agent_{i+1}")
        real_opponent_states.append(real_opp_state)

    # init std
    core_state_dim = cfg.train.core_state_dim
    std = StandardizerRS.create(
        core_state_dim=core_state_dim,
        act_dim_ego=act_dim,
        act_dim_opp=act_dim * opp_num,
    )

    # init animation parameters
    episode_reward_history = []
    epochs_to_render = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 130, 160, 200]
    total_comm_count = 0

    # prepare fixed_batch
    rng, k_fix = jax.random.split(rng)
    policy_fn = make_policy_fn(policy_state)
    real_opp_fn = make_opp_fn(real_opponent_states)
    batch_env_fix, final_state_fix, rng = collect_real_data(
        policy_fn=policy_fn,
        opp_fn=real_opp_fn,
        obs_dim=obs_dim,
        act_dim=act_dim,
        opp_num=opp_num,
        opp_dim=act_dim,  # 如果每个对手与 ego 维度不同，这里改成对应维度
        key=k_fix,
        cfg=cfg
    )
    rng, k_fix_sample = jax.random.split(rng)
    replay_env_fix = add_batch_env_to_replay(replay_env_fix, batch_env_fix, cfg)
    batch_env_fix_sample = replay_env_fix.sample(k_fix_sample, batch_size=cfg.train.batch_size, opp_num=opp_num)

    print("✅ Init done.")

    for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="Training Epochs"):
        print(f"\n===== Epoch {epoch}/{cfg.train.epochs} =====")
        if cfg.train.reinit_opp_model:
            # -------------------------------------------------
            # 0) REINIT opponent model
            # ---------------------------------——--------------
            opponent_states = []
            for i in range(opp_num):
                rng, ko = jax.random.split(rng)
                j = i + 1
                _, opp_state = init_policy_ensemble(ko, obs_dim, act_dim, cfg.policy, f"agent_{j}")
                opponent_states.append(opp_state)

        # -------------------------------------------------
        # 1) 真实环境采样 D_env
        # ---------------------------------——--------------
        policy_fn = make_policy_fn(policy_state)
        opp_fn = make_opp_fn(opponent_states)
        real_opp_fn = make_opp_fn(real_opponent_states)

        rng, kc = jax.random.split(rng)
        batch_env, final_state, rng = collect_real_data(
            policy_fn=policy_fn,
            opp_fn=real_opp_fn,
            obs_dim=obs_dim,
            act_dim=act_dim,
            opp_num=opp_num,
            opp_dim=act_dim,   # 如果每个对手与 ego 维度不同，这里改成对应维度
            key=kc,
            cfg=cfg
        )

        replay_env = add_batch_env_to_replay(replay_env, batch_env, cfg)
        replay_dyna = ReplayBuffer.create(cfg.collect.steps_per_epoch, obs_dim, act_dim, opp_num, state_dim)
        replay_dyna = add_batch_env_to_replay(replay_dyna, batch_env, cfg)
        # -------------------------------------------------
        # 2) 基于 D_env 拟合 Standardizer，并训练 dynamics
        # -------------------------------------------------
        # 用一批 env 数据估计均值方差
        rng, ks = jax.random.split(rng)
        # boot = replay_env.sample(ks, batch_size=epoch*cfg.train.batch_size_std, opp_num=opp_num)
        # std = Standardizer.fit(boot["state"], boot["a_ego"], boot["a_opp"], boot["next_state"])
        boot = replay_env.sample(ks, batch_size=len(replay_dyna), opp_num=opp_num)
        std = std.update(boot)

        # 训练 dynamics
        for i in range(cfg.train.model_updates):
            rng, kb = jax.random.split(rng)
            b = replay_env.sample(kb, batch_size=cfg.train.batch_size, opp_num=opp_num)
            transition_state, trans_metrics = train_transition_step(transition_state, b, std)
            reward_state, rew_metrics = train_reward_step(reward_state, b, std)
            wandb.log({
                "transition_nll": trans_metrics["transition_nll"],
                "transition_mse": trans_metrics["transition_mse"],
                "reward_mse": rew_metrics["reward_mse"],
            })
        print(f"Model NLL: {float(trans_metrics['transition_nll']):.4f}")
        print(f"[Epoch {epoch}] Model NLL: {float(trans_metrics['transition_nll']):.4f}")
        # -------------------------------------------------
        # 3) update opponent model by clone learning
        # -------------------------------------------------
        for i in range(cfg.train.opp_model_updates):
            opponent_states, opp_metrics = update_opponent_model(
                opponent_states,
                batch_env,
            )
        # -------------------------------------------------
        # 4) 模型 rollout 生成 D_model
        #    （rollout.py 内部已包含 adaptive n^j 逻辑）
        # -------------------------------------------------
        rng, kr = jax.random.split(rng)
        replay_model, comm_count_cum = rollout_model(
            rng=kr,
            transition_state=transition_state,
            reward_state=reward_state,
            std=std,
            policy_state=policy_state,
            opponent_policies=opponent_states,
            replay_env=replay_env,
            replay_model=replay_model,
            cfg=cfg,
            epoch=epoch,
        )

        total_comm_count += comm_count_cum
        wandb.log({
            "total_comm_count": total_comm_count,
        })

        # -------------------------------------------------
        # 5) 用 D_model 更新 Q & Policy
        # -------------------------------------------------

        for i in range(cfg.train.gradient_updates):
            rng, subkey = jax.random.split(rng)
            # if epoch > 15:
            batch = replay_model.sample(subkey, batch_size=cfg.train.batch_size, opp_num=opp_num)
            # else:
            #     batch = replay_env.sample(subkey, batch_size=cfg.train.batch_size, opp_num=opp_num)

            # 先更新两个 Q（update_q_function 里已做最小化目标）
            q1_state, q2_state, q_metrics, rng = update_q_function(
                q1_state=q1_state,
                q2_state=q2_state,
                target_q1_state=target_q1_state,
                target_q2_state=target_q2_state,
                policy_state=policy_state,
                opponent_policies=opponent_states,
                batch=batch,
                cfg=cfg.q_function,
                rng=rng,
            )
            joint_act = jnp.concatenate([batch["a_ego"], batch["a_opp"]], axis=-1)
            q1_pred = q1_state.apply_fn({"params": q1_state.params}, batch["state"], joint_act)
            q2_pred = q2_state.apply_fn({"params": q2_state.params}, batch["state"], joint_act)
            mean_q1 = jnp.mean(q1_pred)
            mean_q2 = jnp.mean(q2_pred)
            std_q1 = jnp.std(q1_pred)
            if mean_q1 < mean_q2:
                smaller_q_state = q1_state
                # print("Q1 is smaller, mean:", float(mean_q1))
            else:
                smaller_q_state = q2_state
                # print("Q2 is smaller, mean:", float(mean_q2))




            #update ego policy
            policy_state, pi_metrics = update_policy(
                policy_state=policy_state,
                q_state=smaller_q_state,  # 如果你在 update_policy 里使用 min(Q1,Q2)，这里传个结构或改函数
                batch=batch,
                cfg=cfg.policy,
                rng=rng,
                opponent_policies=opponent_states,
            )


            # if epoch % 10 == 0:
            # update real opponents policy
            new_opponent_states = []

            for i in range(len(real_opponent_states)):
                update_opp = real_opponent_states[i]  # 当前的 opponent_policy_state

                # 更新该 opponent 的 policy
                new_state, metrics = update_opponent_policy(
                    opponent_state=update_opp,
                    q_state=smaller_q_state,
                    batch=batch,
                    cfg=cfg.policy,
                    rng=rng,
                    ego_policy_state=policy_state,
                    all_opponent_states=real_opponent_states,
                )

                new_opponent_states.append(new_state)
            real_opponent_states = new_opponent_states

            # 软更新 target Q
            target_q1_state = soft_update(target_q1_state, q1_state, cfg.q_function.tau)
            target_q2_state = soft_update(target_q2_state, q2_state, cfg.q_function.tau)
            wandb.log({
                "q1_loss": q_metrics["q1_loss"],
                "q2_loss": q_metrics["q2_loss"],
                "q1_pred": q_metrics["q1_pred"],
                "q2_pred": q_metrics["q2_pred"],
                # "policy_loss": pi_metrics["policy_loss"],
                "sac_step": i
            })

            # update_real_opp_state = []
            # for state in real_opponent_states:
            #     opp_state = state.replace(
            #         params=copy.deepcopy(policy_state.params),
            #         opt_state=copy.deepcopy(policy_state.opt_state),
            #         step=policy_state.step,
            #     )
            #     update_real_opp_state.append(opp_state)
            # real_opponent_states = update_real_opp_state

        eval_rng = jax.random.PRNGKey(0)
        policy_fn = make_policy_fn(policy_state)
        opp_fn = make_opp_fn(real_opponent_states)
        epi_reward, epi_reward_0, _ = episode_reward(policy_fn, opp_fn, num_agents, eval_rng, cfg)
        episode_reward_history.append(epi_reward)
        wandb.define_metric("episode_reward", step_metric="total_comm_count_x")
        wandb.log({
            "total_comm_count_x": total_comm_count,
            "episode_reward": epi_reward,
        })
        wandb.log({
            "episode_reward": epi_reward,
            "epi_reward_agent-0": epi_reward_0,
        })

        # 简单日志
        q1l = float(q_metrics.get("q1_loss", 0.0))
        q2l = float(q_metrics.get("q2_loss", 0.0))
        # pil = float(pi_metrics.get("policy_loss", 0.0))
        print(f"[Epoch {epoch}] Q1 {q1l:.4f} | Q2 {q2l:.4f} | Policy")  # {pil:.4f}

        # -------------------------------------------------
        # 6) evaluate fixed q loss use fixed batch
        # -------------------------------------------------
        eval_metrics = evaluate_fixed_q_loss(
            q1_state,
            q2_state,
            target_q1_state,
            target_q2_state,
            policy_state,
            real_opponent_states,
            batch_env_fix_sample,
            cfg.q_function,
            rng,
        )
        wandb.log({
            "fixed_q1_loss": float(eval_metrics["q1_eval_loss"]),
            "fixed_q2_loss": float(eval_metrics["q2_eval_loss"]),
            "fixed_q1_pred_mean": float(eval_metrics["q1_pred_mean"]),
            "fixed_q2_pred_mean": float(eval_metrics["q2_pred_mean"]),
            "fixed_target_mean": float(eval_metrics["target_mean"]),
        })
        print("Q eval loss:", eval_metrics["q1_eval_loss"])
        # -------------------------------------------------
        # 7) evaluate episode_reward
        # -------------------------------------------------
        if epoch % cfg.train.eval_interval == 0:
            # rng, compare_key = jax.random.split(rng, 2)
            compare_key = jax.random.PRNGKey(40)
            state_env, reward_env, state_dyna, reward_dyna = rollout_compare(
                policy_fn=policy_fn,
                opp_fn=real_opp_fn,
                transition_state=transition_state,
                reward_state=reward_state,
                std=std,
                key=compare_key,
                horizon=15,
                cfg=cfg
            )
            T = state_dyna.shape[0]
            mse_list = []
            l2_list = []
            episode_reward_env = {f"agent_{i}": 0.0 for i in range(3)}
            episode_reward_dyna = {f"agent_{i}": 0.0 for i in range(3)}
            for t in range(T):   # State to dict
                env_state_t = {
                    "p_pos": state_env.p_pos[t],
                    "p_vel": state_env.p_vel[t],
                    "c": state_env.c[t],
                    "done": state_env.done[t],
                    "step": state_env.step[t]
                }
                flat_env= manual_flatten_dict(env_state_t) # dict to flat
                flat_dyna = state_dyna[t]
                diff = flat_env - flat_dyna
                # print("flat_env:", flat_env)
                # print("flat_dyna:", flat_dyna)
                # print("diff of flat_env and flat_dyna:", diff)
                mse = jnp.mean(diff**2)
                l2 = jnp.linalg.norm(diff)
                mse_list.append(mse)
                l2_list.append(l2)

                # reward error
                reward_mse_per_agent = {}
                reward_abs_per_agent = {}
                reward_errors = []  # MSE

                for agent_i in reward_env.keys():
                    env_r = reward_env[agent_i][t]  # (1,)
                    dyna_r = reward_dyna[agent_i][t]  # (1,)

                    r_diff = env_r - dyna_r  # (1,)
                    mse_i = jnp.mean(r_diff ** 2)

                    reward_mse_per_agent[agent_i] = mse_i
                    reward_abs_per_agent[agent_i] = jnp.abs(r_diff)

                    reward_errors.append(mse_i)

                # === 总 reward MSE ===
                reward_errors = jnp.stack(reward_errors)  # (N,)
                reward_mse_total = jnp.mean(reward_errors)
                wandb.log({
                        "mse" : mse,
                        "l2" : l2,
                        "reward_error": reward_mse_total,
                    })
            for agent_i in reward_env.keys():
                episode_reward_env[agent_i] = float(jnp.sum(reward_env[agent_i]).item())
                episode_reward_dyna[agent_i] = float(jnp.sum(reward_dyna[agent_i]).item())

            log_dict = {}
            log_dict["episode_reward_env"] = (
                                                          episode_reward_env["agent_0"] +
                                                          episode_reward_env["agent_1"] +
                                                          episode_reward_env["agent_2"]
                                                  )

            log_dict["episode_reward_dyna"] = (
                                                           episode_reward_dyna["agent_0"] +
                                                           episode_reward_dyna["agent_1"] +
                                                           episode_reward_dyna["agent_2"]
                                                   )
            wandb.log({
                "episode_reward_env_agent_0": episode_reward_env["agent_0"],
                "episode_reward_dyna_agent_0": episode_reward_dyna["agent_0"],
                "episode_reward_env": log_dict["episode_reward_env"],
                "episode_reward_dyna": log_dict["episode_reward_dyna"],
            })

        if epoch in epochs_to_render:
            rng, k_eval = jax.random.split(rng)
            policy_fn = make_policy_fn(policy_state)
            opp_fn = make_opp_fn(real_opponent_states)

            epi_reward, _, traj = episode_reward(policy_fn, opp_fn, num_agents, k_eval, cfg)
            # episode_reward_history.append(epi_reward)
            animate_episode(traj, episode_reward_history, save_path=f"episode_epoch_{epoch}.mp4")

    save_execution_checkpoint(
        transition_state=transition_state,
        reward_state=reward_state,
        policy_state=policy_state,
        real_opponent_states=real_opponent_states,
        std=std,
        cfg=cfg,
        num_agents=num_agents,
        num_opponents=num_opponents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        act_dim=act_dim,
        opp_dim=opp_dim,
        final_epoch=cfg.train.epochs,
        save_name="final_execution_ckpt.pkl",
    )

    wandb.finish()
    print("\n🎉 Training finished.")

if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()

















