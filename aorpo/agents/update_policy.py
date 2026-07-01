# aorpo/agents/update_policy.py
from __future__ import annotations
from typing import Dict, Any,  List

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from omegaconf import DictConfig

from aorpo.agents.policy import PolicyNet, TrainStateid, EnsemblePolicyUtils


def update_policy(
    policy_state: TrainStateid,
    q_state: TrainState,
    batch: Dict[str, jnp.ndarray],
    cfg: DictConfig,
    rng: Any,
    opponent_policies: List[TrainStateid],
):
    """
    Update policy parameters using SAC/AORPO objective:
        J(π) = E_s[ α * log π(a|s) - Q(s,a) ]
    """

    def loss_fn(params, rng):
        # --- 采样动作 & 对应 log π(a|s)
        agent_id = policy_state.agent_id
        obs = batch["obs"][agent_id]
        state = batch["state"]
        a_i, log_prob, rng = PolicyNet.sample_action(
            params, policy_state.apply_fn, rng, obs
        )
        a_js = []
        for j, opp in enumerate(opponent_policies):
            # 使用 learned opponent policy
            rng, subkey = jax.random.split(rng)
            # a_j, _, _ = PolicyNet.sample_action(
            #     opp["state"].params,
            #     opp["state"].apply_fn,
            #     subkey,
            #     batch["obs"][f"agent_{j + 1}"],
            # )
            a_j, _ = EnsemblePolicyUtils.sample_deterministic_action_ensemble(
                opp.params,
                opp.apply_fn,
                batch["obs"][f"agent_{j + 1}"],
            )
            a_j = jax.lax.stop_gradient(a_j)
            a_js.append(a_j)

        action = jnp.concatenate([a_i] + a_js, axis=-1)
        # jax.debug.print("log_prob:{}", log_prob)
        # --- 计算 Q(s,a)
        # q_value = q_state.apply_fn({"params": q_state.params}, state, action)
        # q_value = jax.lax.stop_gradient(q_value)  # ❗ policy 不能更新 Q 参数
        q_value = q_state.apply_fn(
            {"params": jax.lax.stop_gradient(q_state.params)},
            state,
            action
        )

        log_prob = log_prob.reshape(-1, 1)
        # --- 策略损失 Eq.(4)
        policy_loss = jnp.mean(cfg.alpha * log_prob - q_value)

        return policy_loss, {"policy_loss": policy_loss}

    grads, metrics = jax.grad(loss_fn, has_aux=True)(policy_state.params, rng)

    # # === 打印梯度大小 ===
    # grad_norm = jax.tree_util.tree_reduce(
    #     lambda a, b: a + jnp.linalg.norm(b), grads, initializer=0.0
    # )
    # jax.debug.print("policy grad_norm_agent_0 = {}", grad_norm)

    # --- 更新参数
    updates, opt_state = policy_state.tx.update(grads, policy_state.opt_state, policy_state.params)
    new_params = optax.apply_updates(policy_state.params, updates)

    new_state = policy_state.replace(
        step=policy_state.step + 1,
        params=new_params,
        opt_state=opt_state,
    )

    return new_state, metrics


update_policy = jax.jit(update_policy, static_argnums=(3,))

def update_opponent_policy(
    opponent_state: TrainStateid,         # 当前 opponent 的 policy_state
    q_state: TrainState,                  # 共享 Q 函数
    batch: Dict[str, jnp.ndarray],
    cfg: DictConfig,
    rng: Any,
    ego_policy_state: TrainStateid,       # ego 的策略
    all_opponent_states: List[TrainStateid],  # 所有 opponent 的策略
):
    """
    只更新指定 opponent_state 的策略。
    对自己采样动作；其他 agent 的动作全 stop_gradient。
    """

    # 当前 opponent 的索引，例如 agent_1 → 1
    agent_id = opponent_state.agent_id
    agent_idx = int(agent_id.split("_")[-1])

    def loss_fn(params, rng):
        """构造 joint action，并计算 opponent policy loss。"""

        # === 1. 当前 agent 的 observation ===
        obs_i = batch["obs"][agent_id]

        # === 2. 对当前 opponent 采样动作（有梯度） ===
        a_i, logp_i, rng = PolicyNet.sample_action(
            params, opponent_state.apply_fn, rng, obs_i
        )

        # === 3. 构造 joint action ===
        actions = []

        # -- 先添加 ego --
        obs_ego = batch["obs"]["agent_0"]
        a_ego = PolicyNet.deterministic_action(
            ego_policy_state.params,
            ego_policy_state.apply_fn,
            obs_ego,
        )
        a_ego = jax.lax.stop_gradient(a_ego)
        actions.append(a_ego)

        # --- 添加所有 opponent 的动作 ---
        for j, opp_state in enumerate(all_opponent_states):
            obs_j = batch["obs"][f"agent_{j+1}"]   # opponent 从 1 开始编号

            if j+1 == agent_idx:
                # 当前正在更新的 opponent
                actions.append(a_i)
            else:
                # 其他 opponents：deterministic, stop_gradient
                a_j = PolicyNet.deterministic_action(
                    opp_state.params,
                    opp_state.apply_fn,
                    obs_j
                )
                a_j = jax.lax.stop_gradient(a_j)
                actions.append(a_j)

        # === 拼 joint action ===
        joint_action = jnp.concatenate(actions, axis=-1)

        # === 4. 获取 Q(s, joint_action) ===
        state = batch["state"]
        q_value = q_state.apply_fn(
            {"params": jax.lax.stop_gradient(q_state.params)},
            state,
            joint_action
        )
        logp_i = logp_i.reshape(-1, 1)
        # === 5. SAC policy objective ===
        policy_loss = jnp.mean(cfg.alpha * logp_i - q_value)

        return policy_loss, {"opponent_policy_loss": policy_loss}

    # === compute grad ===
    grads, metrics = jax.grad(loss_fn, has_aux=True)(opponent_state.params, rng)

    # # === 打印梯度大小 ===
    # grad_norm = jax.tree_util.tree_reduce(
    #     lambda a, b: a + jnp.linalg.norm(b), grads, initializer=0.0
    # )
    # jax.debug.print("policy grad_norm_agent_opp = {}", grad_norm)

    # === update params ===
    updates, opt_state = opponent_state.tx.update(
        grads, opponent_state.opt_state, opponent_state.params
    )
    new_params = optax.apply_updates(opponent_state.params, updates)

    # === 返回新的 state ===
    new_state = opponent_state.replace(
        step=opponent_state.step + 1,
        params=new_params,
        opt_state=opt_state,
    )

    return new_state, metrics
update_opponent_policy = jax.jit(update_opponent_policy, static_argnums=(3,))