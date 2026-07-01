# aorpo/agents/opponent_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from omegaconf import DictConfig

from aorpo.agents.policy import PolicyNet


# ---------------------------------------------------
# Config
# ---------------------------------------------------
@dataclass
class OpponentPolicyConfig:
    lr: float = 3e-4
    min_logvar: float = -5.0
    max_logvar: float = 2.0
    hidden_dims: tuple = (256, 256)


# ---------------------------------------------------
# Initialize opponent policy model
# ---------------------------------------------------
def init_opponent_policy_model(rng: Any, obs_dim: int, act_dim: int, cfg: DictConfig):
    """初始化对手策略网络"""
    model = PolicyNet(
        action_dim=act_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        min_logvar=cfg.min_logvar,
        max_logvar=cfg.max_logvar,
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(init_rng, dummy_obs)["params"]

    tx = optax.adam(cfg.lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


# ---------------------------------------------------
# Opponent policy update
# ---------------------------------------------------
def update_opponent_policy(
    policy_state: TrainState,
    batch: Dict[str, jnp.ndarray],
    cfg: DictConfig,
):
    """
    更新对手策略 π_φj，使用行为克隆 (Behavior Cloning) 目标：
    最大化 E[ log π_φj(a_j | s) ]
    输入 batch: dict(obs, act)，来自环境数据 D_env
    """

    def loss_fn(params):
        mu, log_std = policy_state.apply_fn({'params': params}, batch["obs"])
        std = jnp.exp(log_std)

        # 重参数化成高斯对数似然 (log prob)
        log_prob = -0.5 * jnp.sum(
            ((batch["act"] - mu) / (std + 1e-8)) ** 2
            + 2 * log_std
            + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        loss = -jnp.mean(log_prob)
        return loss, {"opponent_loss": loss}

    grads, metrics = jax.grad(loss_fn, has_aux=True)(policy_state.params)
    updates, opt_state = policy_state.tx.update(grads, policy_state.opt_state, policy_state.params)
    new_params = optax.apply_updates(policy_state.params, updates)
    new_state = policy_state.replace(
        step=policy_state.step + 1,
        params=new_params,
        opt_state=opt_state,
    )
    return new_state, metrics


# JIT 加速
update_opponent_policy = jax.jit(update_opponent_policy, static_argnums=(2,))
