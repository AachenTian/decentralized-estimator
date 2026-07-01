# aorpo/rollout/rollout.py
from __future__ import annotations
from typing import Dict, Any, List
import jax
import jax.numpy as jnp
from flax.nnx import TrainState
from omegaconf import DictConfig, OmegaConf
import wandb, random
from dataclasses import dataclass
from functools import partial

from aorpo.agents.model_dynamics import predict_next, eval_error, unflatten_batch, dynamics_uncertainty_per_dim
from aorpo.agents.policy import PolicyNet, EnsemblePolicyUtils
from aorpo.utils.replay import ReplayBuffer
from aorpo.rollout.uncertainty_threshold import compute_opponent_threshold, compute_entropy_thresholds

@dataclass
class State:
    p_pos: jnp.ndarray
    p_vel: jnp.ndarray
    c: jnp.ndarray
    done: jnp.ndarray
    step: jnp.ndarray

def dict_to_state(d):
    return State(
        p_pos=d["p_pos"],
        p_vel=d["p_vel"],
        c=d["c"],
        done=d["done"],
        step=d["step"],
    )
# -----------------------------------------------------
# communication function
# -----------------------------------------------------
def Comm(policy_state, obs_j, rng):
    action, _, rng = PolicyNet.sample_action(
        policy_state.params,
        policy_state.apply_fn,
        rng,
        obs_j
    )
    return action, rng


# -----------------------------------------------------
# Compute adaptive rollout length for each opponent
# -----------------------------------------------------
def compute_rollout_lengths(errors: List[float], k: int) -> List[int]:
    """
    根据 AORPO 论文公式计算每个对手的 rollout 步数：
        n^j = floor(k * min_j'(ε̂_j') / ε̂_j)
    Args:
        errors: list of opponent model errors ε̂_j
        k: maximum rollout length (超参数)
    Returns:
        n_js: list[int], rollout steps for each opponent
    """
    min_err = min(errors)
    n_js = []
    for e in errors:
        print("errors",errors)
        print(e)
        ratio = k * min_err / float(e)
        print(ratio)
        n_js.append(max(1, int(ratio)))
    return n_js

def add_batch_model_to_replay(replay: ReplayBuffer, batch: dict, cfg:DictConfig) -> ReplayBuffer:
    return replay.add_batch_model(batch, cfg)




# -----------------------------------------------------
# Rollout using learned dynamics + opponent models
# -----------------------------------------------------
def rollout_model(
    rng: Any,
    transition_state: Any,
    reward_state: Any,
    std: Any,
    policy_state: Any,
    opponent_policies: List[Any],
    replay_env: Any,
    replay_model: Any,
    cfg: Any,
    epoch: Any,
):
    """
    Perform adaptive opponent-wise model rollouts.

    Args:
        rng: jax.random.PRNGKey
        transition_state: dynamics model (predict next_state)
        reward_state: reward model (predict reward)
        std: Standardizer (for model_dynamics normalization)
        policy_state: main agent policy (πζ)
        opponent_policies: list of opponent policy dicts [{state, model}, ...]
        replay_env: real environment replay buffer
        replay_model: model replay buffer (to store rollouts)
        cfg: rollout configuration (Hydra)
        epoch: current epoch
    """

    # 1️ 从真实经验池采样初始状态
    key, subkey = jax.random.split(rng)
    batch_env = replay_env.sample(subkey, cfg.rollout.batch_size, cfg.train.num_opponents)
    obs = batch_env["obs"]
    state = batch_env["state"]
    a_opp = batch_env["a_opp"]

    # 2️ 计算lambda
    lambda_opp = compute_opponent_threshold(opponent_policies, replay_env, cfg)
    thresholds = lambda_opp[:, jnp.newaxis, :]

    key, entropy_key = jax.random.split(key)
    lambda1_dyn, lambda2_dyn ,total_var_threshold = compute_entropy_thresholds(
        transition_state=transition_state,
        std=std,
        replay_env=replay_env,
        cfg=cfg,
        rng=entropy_key,
    )   # shape: (state_dim,)

    # # 2️ 计算每个 opponent 模型误差 ε̂_j
    # errors = []
    # for j, opp in enumerate(opponent_policies):
    #     target = a_opp[:, j * cfg.env.act_dim:(j + 1) * cfg.env.act_dim]
    #     pred, _ = opp.apply_fn({"params": opp.params}, obs[f"agent_{j + 1}"])
    #     pred = jnp.tanh(pred)
    #     eps_j = jnp.mean((pred - target) ** 2)
    #     # eps_j = eval_error(
    #     #     real_state=policy_state,
    #     #     opp_state=opp["state"],
    #     #     std=std,
    #     #     batch=batch_env,
    #     #     deterministic=True,
    #     #     member_idx=j,
    #     # )
    #     errors.append(float(eps_j))

    # 3️ 根据公式计算每个对手的 rollout 步数 n^j
    max_n = cfg.rollout.k  # 最大 rollout 步数上限
    def get_k(epoch, max_n):
        epoch = jnp.asarray(epoch)
        start_epoch = 15
        end_epoch = 100
        k_min = 1
        k_max = max_n

        # 线性插值 (浮点)
        k_float = k_min + (k_max - k_min) * ((epoch - start_epoch) / (end_epoch - start_epoch))

        # 四舍五入为整数
        k_int = jnp.round(k_float)

        # 三段逻辑：epoch < 15 → 1；epoch > 100 → max_n；中间 → 插值
        k = jnp.where(epoch < start_epoch, k_min,
                      jnp.where(epoch > end_epoch, k_max, k_int))

        return k.astype(int)
    max_n = get_k(epoch, max_n)
    # n_js = compute_rollout_lengths(errors, int(max_n))
    #
    # print(f"Opponent model errors: {errors}")
    # print(f"Adaptive rollout steps (n^j): {n_js}")

    reward_roll = 0
    #initialize a all True mask
    active_mask = jnp.ones((cfg.rollout.batch_size,), dtype=jnp.bool_)
    active_dyn_mask = jnp.ones((cfg.rollout.batch_size,), dtype=jnp.bool_)

    # 4️ 模型rollout 循环
    step = 0
    comm_count_cum = 0
    entropy_cum = jnp.zeros((cfg.rollout.batch_size,lambda2_dyn.shape[0]), dtype=jnp.float32)
    while True:
        rng, subkey = jax.random.split(rng)

        # 主体动作 a_i
        a_i, _, subkey = PolicyNet.sample_action(
            policy_state.params,
            policy_state.apply_fn,
            subkey,
            obs["agent_0"],
        )

        # 每个对手动作 a_j
        a_js = []
        step_ood_any_opp = jnp.zeros((cfg.rollout.batch_size, cfg.env.act_dim), dtype=jnp.bool_)
        for j, opp in enumerate(opponent_policies):
            # 使用 learned opponent policy
            a_j_ens, mu_j, log_std_j, subkey = EnsemblePolicyUtils.sample_action_ensemble(
                opp.params,
                opp.apply_fn,
                subkey,
                obs[f"agent_{j+1}"],
            )
            current_entropy = EnsemblePolicyUtils.get_infoprop_entropy(mu_j, log_std_j)
            ood_mask_j = current_entropy > thresholds[j]
            opp_j_is_ood = ood_mask_j #jnp.any(ood_mask_j, axis=-1)
            step_ood_any_opp = jnp.logical_or(step_ood_any_opp, opp_j_is_ood)

            a_j_comm, subkey = Comm(policy_state, obs[f"agent_{j+1}"], subkey)
            actual_a_j = jnp.where(opp_j_is_ood, a_j_comm, a_j_ens)
            # if step < n_js[j]:
            #     actual_a_j = a_j_ens
            # else:
            #     actual_a_j = a_j_comm
            a_js.append(actual_a_j)

            comm_count_cum += jnp.sum((opp_j_is_ood).astype(jnp.float32))
            true_ratio = jnp.mean(opp_j_is_ood.astype(jnp.float32))
            print("OOD ratio:", true_ratio)

        # 联合动作
        joint_act = jnp.concatenate([a_i] + a_js, axis=-1)
        a_js = jnp.concatenate(a_js, axis=-1)  # 形状：(batch_size, opp_num * act_dim)
        # 预测下一状态（模型）
        next_state, next_obs, reward_dict, dones_dict, mu, logvar= predict_next(
            transition_state=transition_state,
            reward_state=reward_state,
            std=std,
            state_agent=state,
            a_ego=a_i,
            a_opp=a_js,
            cfg=cfg,
            rng=subkey,
            deterministic=False,
        )
        total_var, _, _, entropy_dim = dynamics_uncertainty_per_dim(mu, logvar)
        entropy_cum += entropy_dim
        # print("entropy_dim:", entropy_dim)
        # print("entropy_cum:", entropy_cum)
        # print("lambda2_dyn:", lambda2_dyn)


        # ood_step = jnp.any(entropy_dim > (lambda1_dyn), axis=-1)  # λ1
        ood_step = jnp.any(total_var > total_var_threshold, axis=-1)  # λ1
        ood_long = jnp.any(entropy_cum > lambda2_dyn, axis=-1)  # λ2
        ood_any = jnp.logical_or(ood_step, ood_long)
        # if step > 0:
        #     if jnp.all(ood_any):
        #         break
            # if step > cfg.rollout.k:
            #     break
        if step >= max_n:
            break
        # state = jax.tree_util.tree_map(lambda x: x[None], state)
        # next_state = jax.tree_util.tree_map(lambda x: x[None], next_state)

        reward_mean = jnp.mean(reward_dict["agent_0"])
        reward_roll += reward_mean
        batch_model = dict(
            state=state,
            obs=obs,
            a_ego=a_i,
            a_opp=a_js,
            next_state=next_state,
            next_obs=next_obs,
            rew=reward_dict,
            dones=dones_dict,
        )

        # 存储到模型经验池
        replay_model = add_batch_model_to_replay(replay_model, batch_model, cfg)
        obs = next_obs
        state = next_state
        step += 1


    jax.debug.print("rollout step ={}",step)
    wandb.log({
        "episode rewards": reward_roll,
        "rollout_step": step
    })

    return replay_model, comm_count_cum
