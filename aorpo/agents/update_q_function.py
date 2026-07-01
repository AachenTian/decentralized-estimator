# aorpo/train_utils/update_q_function.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from omegaconf import DictConfig

from aorpo.agents.q_function import QNet
from aorpo.agents.policy import PolicyNet, EnsemblePolicyUtils


# --------------------------------
# Config
# --------------------------------
@dataclass
class QFunctionConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    alpha: float = 0.2  # entropy temperature


# --------------------------------
# Training Step
# --------------------------------

def update_q_function(
        q1_state: TrainState,
        q2_state: TrainState,
        target_q1_state: TrainState,
        target_q2_state: TrainState,
        policy_state: TrainState,
        opponent_policies: List[TrainState],
        batch: Dict[str, jnp.ndarray],
        cfg: DictConfig,
        rng: Any,
):
    """
    Update 2 Q-function parameters.
    AORPO Eq.(5):  J(Q) = E[(Q(s,a) - (r + γ(Q' - α log π)))^2]
    batch: dict with keys {obs, act, rew, next_obs, done}
    """
    def loss_fn(params1, params2, rng):
        ego_act = batch["a_ego"]
        opp_act = batch["a_opp"]
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        state = batch["state"]
        next_state = batch["next_state"]
        joint_act = jnp.concatenate([ego_act, opp_act],axis=-1)
        q1_pred = q1_state.apply_fn({"params":params1}, state, joint_act)
        q2_pred = q2_state.apply_fn({"params":params2}, state, joint_act)

        # log_prob_s = []
        rng, subkey = jax.random.split(rng)
        a_i, log_prob_i, key = PolicyNet.sample_action(
            policy_state.params, policy_state.apply_fn, rng, next_obs['agent_0']
        )
        # log_prob_s.append(log_prob_i)
        log_prob = log_prob_i

        a_js = []
        for j, opp in enumerate(opponent_policies):
            # 使用 learned opponent policy
            rng, subkey = jax.random.split(rng)
            a_j, _, _ , _ = EnsemblePolicyUtils.sample_action_ensemble(
                opp.params,
                opp.apply_fn,
                subkey,
                next_obs[f"agent_{j+1}"],
            )
            # a_j = EnsemblePolicyUtils.sample_deterministic_action_ensemble(
            #     opp.params,
            #     opp.apply_fn,
            #     next_obs[f"agent_{j + 1}"],
            # )
            # a_j = jax.lax.stop_gradient(a_j)
            a_js.append(a_j)
            # log_prob_s.append(log_prob_j)

        # log_prob =  sum(log_prob_s) / cfg.agent_num
        next_action = jnp.concatenate([a_i] + a_js, axis=-1)
        q1_target_next = target_q1_state.apply_fn(
            {"params":target_q1_state.params}, next_state, next_action
        )
        q2_target_next = target_q2_state.apply_fn(
            {"params": target_q2_state.params}, next_state, next_action
        )
        q_target_next = jnp.minimum(q1_target_next, q2_target_next)

        # rewards = jnp.stack(
        #     [batch["rew"][f"agent_{i}"] for i in range(cfg.agent_num)], axis=-1
        # )  # (B, num_agents)
        #reward = jnp.sum(rewards, axis=-1)  # (B,)
        # reward = reward / cfg.agent_num
        reward = 3 * batch["rew"]["agent_0"]
        reward = reward / cfg.reward_scale
        target_q = reward.squeeze(-1) + cfg.gamma * (1.0 - batch["dones"]['agent_0'].squeeze(-1)) * (q_target_next - cfg.alpha * log_prob) #* (q_target_next - cfg.alpha * log_prob) delete other agent's entropy
        target_q = jax.lax.stop_gradient(target_q)

        #Q1, Q2 MSE
        loss_q1 = jnp.mean((q1_pred - target_q) **2)
        loss_q2 = jnp.mean((q2_pred - target_q) **2)
        total_loss = 0.5 * (loss_q1 + loss_q2)
        q1_pred_mean = jnp.mean(q1_pred)
        q2_pred_mean = jnp.mean(q2_pred)

        return total_loss, {"q1_loss": loss_q1, "q2_loss": loss_q2, "q1_pred": q1_pred_mean, "q2_pred": q2_pred_mean}

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)
    (loss, metrics), (grads1, grads2) = grad_fn(q1_state.params, q2_state.params, rng)

    # # === 打印梯度大小 ===
    # grad_norm1 = jax.tree_util.tree_reduce(
    #     lambda a, b: a + jnp.linalg.norm(b), grads1, initializer=0.0
    # )
    # grad_norm2 = jax.tree_util.tree_reduce(
    #     lambda a, b: a + jnp.linalg.norm(b), grads2, initializer=0.0
    # )
    # jax.debug.print("q1 function grad_norm_agent_opp = {}", grad_norm1)
    # jax.debug.print("q2 function grad_norm_agent_opp = {}", grad_norm2)

    updates1, opt_state1 = q1_state.tx.update(grads1, q1_state.opt_state, q1_state.params)
    updates2, opt_state2 = q2_state.tx.update(grads2, q2_state.opt_state, q2_state.params)

    new_params1 = optax.apply_updates(q1_state.params, updates1)
    new_params2 = optax.apply_updates(q2_state.params, updates2)

    new_q1_state = q1_state.replace(
        step=q1_state.step + 1, params=new_params1, opt_state=opt_state1
    )
    new_q2_state = q2_state.replace(
        step=q2_state.step + 1, params=new_params2, opt_state=opt_state2
    )

    return new_q1_state, new_q2_state, metrics, rng

update_q_function = jax.jit(update_q_function, static_argnums=(7,))



def evaluate_fixed_q_loss(
    q1_state,
    q2_state,
    target_q1_state,
    target_q2_state,
    policy_state,
    opponent_policies,
    fixed_batch,
    cfg,
    rng,
):
    """
    Evaluate Q-loss on a fixed batch of data (no randomness).
    This allows you to see if Q-function is actually improving.

    fixed_batch keys:
    {
      "obs", "next_obs", "a_ego", "a_opp",
      "state", "next_state",
      "rew", "dones"
    }
    """

    def compute_loss(params1, params2, rng):
        ego_act = fixed_batch["a_ego"]
        opp_act = fixed_batch["a_opp"]
        state = fixed_batch["state"]
        next_state = fixed_batch["next_state"]

        # 1. Current Q(s, a)
        joint_act = jnp.concatenate([ego_act, opp_act], axis=-1)
        q1_pred = q1_state.apply_fn({"params": params1}, state, joint_act)
        q2_pred = q2_state.apply_fn({"params": params2}, state, joint_act)

        # 2. Next actions (we freeze RNG so the evaluation is deterministic)
        rng, subkey = jax.random.split(rng)

        a_i = PolicyNet.deterministic_action(
            policy_state.params,
            policy_state.apply_fn,
            fixed_batch["next_obs"]["agent_0"],
        )
        log_prob = 0
        a_js = []
        for j, opp in enumerate(opponent_policies):
            rng, sub = jax.random.split(rng)
            a_j = PolicyNet.deterministic_action(
                opp.params,
                opp.apply_fn,
                fixed_batch["next_obs"][f"agent_{j+1}"]
            )
            a_js.append(a_j)

        next_action = jnp.concatenate([a_i] + a_js, axis=-1)

        # 3. Compute Q-target
        q1_t = target_q1_state.apply_fn({"params": target_q1_state.params},
                                        next_state, next_action)
        q2_t = target_q2_state.apply_fn({"params": target_q2_state.params},
                                        next_state, next_action)
        q_target_next = jnp.minimum(q1_t, q2_t)

        # reward sum over agents
        # rewards = jnp.stack(
        #     [fixed_batch["rew"][f"agent_{i}"] for i in range(cfg.agent_num)], axis=-1
        # )
        # reward = jnp.sum(rewards, axis=-1)
        # reward = reward / cfg.agent_num
        reward = 3 * fixed_batch["rew"]["agent_0"]
        reward = reward / cfg.reward_scale

        target_q = (
            reward.squeeze(-1)
            + cfg.gamma
            * (1 - fixed_batch["dones"]["agent_0"].squeeze(-1))
            * (q_target_next - cfg.alpha * log_prob)
        )

        # 4. Q-loss (no gradient, one-shot eval)
        loss_q1 = jnp.mean((q1_pred - target_q) ** 2)
        loss_q2 = jnp.mean((q2_pred - target_q) ** 2)

        return {
            "q1_eval_loss": loss_q1,
            "q2_eval_loss": loss_q2,
            "q1_pred_mean": jnp.mean(q1_pred),
            "q2_pred_mean": jnp.mean(q2_pred),
            "target_mean": jnp.mean(target_q),
        }

    compute_loss = jax.jit(compute_loss)

    return compute_loss(q1_state.params, q2_state.params, rng)
