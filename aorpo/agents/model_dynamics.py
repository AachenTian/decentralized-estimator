# aorpo/agents/model_dynamics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional, Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from omegaconf import DictConfig
from brax.training.acme import running_statistics
from aorpo.utils.replay import ReplayBuffer

# -------------------------------
# Standardization helpers
# -------------------------------
@dataclass
class Standardizer:
    """Hold mean/std for obs, act, delta (next - obs)."""
    state_mean: jnp.ndarray
    state_std: jnp.ndarray
    a_ego_mean: jnp.ndarray
    a_ego_std: jnp.ndarray
    a_opp_mean: jnp.ndarray
    a_opp_std: jnp.ndarray
    delta_mean: jnp.ndarray
    delta_std: jnp.ndarray
    eps: float = 1e-6

    def __hash__(self)-> int:
        return id(self)

    @classmethod
    def fit(cls, state: jnp.ndarray, a_ego: jnp.ndarray, a_opp:jnp.ndarray, next_state: jnp.ndarray) -> "Standardizer":
        delta = next_state - state
        def _ms(x):
            return jnp.mean(x, axis=0), jnp.std(x, axis=0) + 1e-6

        sm, ss = _ms(state)
        aem, aes = _ms(a_ego)
        aom, aos = _ms(a_opp)
        dm, ds = _ms(delta)
        return cls(sm, ss, aem, aes, aom, aos, dm, ds)

    def norm_state(self, x):  return (x - self.state_mean) / (self.state_std + self.eps)
    def denorm_state(self, x): return x * (self.state_std + self.eps) + self.state_mean
    def norm_a_ego(self, x):  return (x - self.a_ego_mean) / (self.a_ego_std + self.eps)
    def norm_a_opp(self, x):  return (x - self.a_opp_mean) / (self.a_opp_std + self.eps)
    def norm_delta(self, x): return (x - self.delta_mean) / (self.delta_std + self.eps)
    def denorm_delta(self, x): return x * (self.delta_std + self.eps) + self.delta_mean
# -------------------------------
# Standardization helpers
# -------------------------------
@struct.dataclass
class StandardizerRS:
    state_stats: running_statistics.RunningStatisticsState
    a_ego_stats: running_statistics.RunningStatisticsState
    a_opp_stats: running_statistics.RunningStatisticsState
    delta_stats: running_statistics.RunningStatisticsState

    @classmethod
    def create(cls, core_state_dim, act_dim_ego, act_dim_opp):
        """Initialize running stats for each component."""
        dummy_state = jnp.zeros((core_state_dim,))
        dummy_ego = jnp.zeros((act_dim_ego,))
        dummy_opp = jnp.zeros((act_dim_opp,))
        dummy_delta = jnp.zeros((core_state_dim,))

        return cls(
            state_stats=running_statistics.init_state(dummy_state),
            a_ego_stats=running_statistics.init_state(dummy_ego),
            a_opp_stats=running_statistics.init_state(dummy_opp),
            delta_stats=running_statistics.init_state(dummy_delta),
        )
    # --------------------------------------------------------------
    # Update running statistics using a batch from replay_env
    # --------------------------------------------------------------
    def update(self, batch):
        """
        batch:
            'state':      (B, state_dim)
            'a_ego':      (B, act_dim)
            'a_opp':      (B, opp_dim)
            'next_state': (B, state_dim)
        """
        core_state = extract_core_state(batch["state"])
        core_next_state = extract_core_state(batch["next_state"])
        delta = core_next_state - core_state

        new_state_stats = running_statistics.update(self.state_stats, core_state)
        new_ego_stats = running_statistics.update(self.a_ego_stats, batch["a_ego"])
        new_opp_stats = running_statistics.update(self.a_opp_stats, batch["a_opp"])
        new_delta_stats = running_statistics.update(self.delta_stats, delta)

        return StandardizerRS(
            state_stats=new_state_stats,
            a_ego_stats=new_ego_stats,
            a_opp_stats=new_opp_stats,
            delta_stats=new_delta_stats,
        )
    # --------------------------------------------------------------
    # Normalization
    # --------------------------------------------------------------

    def norm_state(self, x):
        return running_statistics.normalize(x, self.state_stats)

    def norm_a_ego(self, x):
        return running_statistics.normalize(x, self.a_ego_stats)

    def norm_a_opp(self, x):
        return running_statistics.normalize(x, self.a_opp_stats)

    def norm_delta(self, x):
        return running_statistics.normalize(x, self.delta_stats)

    # --------------------------------------------------------------
    # Denormalization
    # --------------------------------------------------------------

    def denorm_state(self, x):
        return running_statistics.denormalize(x, self.state_stats)

    def denorm_delta(self, x):
        return running_statistics.denormalize(x, self.delta_stats)


# -------------------------------
# State unflatten
# -------------------------------
@dataclass
class State:
    p_pos: jnp.ndarray
    p_vel: jnp.ndarray
    c: jnp.ndarray
    dones: jnp.ndarray
    step: jnp.ndarray

def manual_unflatten_state(flat_state: jnp.ndarray, num_agents: int = 3, num_land: int = 3):
    B = flat_state.shape[0]
    idx = 0
    num_object = num_agents +num_land
    # --- p_pos ---
    p_pos_dim = num_object * 2
    p_pos = flat_state[..., idx:idx + p_pos_dim].reshape(B, num_object, 2)
    idx += p_pos_dim

    # ---p_vel---
    p_vel_dim = num_object * 2
    p_vel = flat_state[..., idx: idx + p_vel_dim].reshape(B, num_object, 2)
    idx += p_vel_dim

    # --- c ---
    c_dim = num_agents * 2
    c = flat_state[..., idx : idx+c_dim].reshape(B, num_agents, 2)
    idx += c_dim

    # --- done ---
    dones = flat_state[..., idx : idx + num_agents].reshape(B, num_agents)
    idx += num_agents

    # --- step ---
    step = flat_state[..., idx:]

    restored_state = State(
        p_pos=p_pos,
        p_vel=p_vel,
        c=c,
        dones=dones,
        step=step
    )
    return restored_state

def unflatten_batch(flat_batch):
    states = []
    for i in range(flat_batch.shape[0]):
        s = manual_unflatten_state(flat_batch[i])   # 输入 shape (1,12) 或 (12,)
        states.append(s)
    # 把 256 个 State 各字段 stack 成 batched state
    return State(
        p_pos=jnp.squeeze(jnp.stack([s.p_pos for s in states]), axis=1),
        p_vel=jnp.squeeze(jnp.stack([s.p_vel for s in states]),axis=1),
        c=jnp.squeeze(jnp.stack([s.c for s in states]), axis=1),
        dones = jnp.squeeze(jnp.array([s.dones for s in states]), axis=1),
        step=jnp.squeeze(jnp.stack([s.step for s in states]),axis=1),
    )
def extract_core_state(flat_state):
    return flat_state[:, :18]

def restore_full_state(prev_flat_state, next_core_state, cfg):
    B = prev_flat_state.shape[0]

    # 1) 先把前 18 维替换掉，后 16 维先照抄
    full = jnp.concatenate(
        [next_core_state, prev_flat_state[:, 18:]],
        axis=-1
    )
    prev_step = prev_flat_state[:, -1]  # (B,)
    step_next = prev_step + 1.0
    step_next = jnp.clip(step_next, 0.0, float(cfg.train.max_steps))  # 不超过 max_steps
    full = full.at[:, -1].set(step_next)

    done_flag = (step_next >= cfg.train.max_steps).astype(full.dtype)  # (B,)
    done_vec = jnp.tile(done_flag[:, None], (1, cfg.q_function.agent_num))  # (B,3)
    full = full.at[:, 30: 30 + cfg.q_function.agent_num].set(done_vec)

    return full



def get_obs(state) -> Dict[str, jnp.ndarray]:
    """计算 batched 状态下每个智能体的观测"""
    num_agents = state.c.shape[1]                     # 第二维是 agent
    num_landmarks = state.p_pos.shape[1] - num_agents

    # === 拆分数据 ===
    agent_pos = state.p_pos[..., :num_agents, :]        # (B, num_agents, 2)
    agent_vel = state.p_vel[..., :num_agents, :]        # (B, num_agents, 2)
    landmark_pos = state.p_pos[..., num_agents:, :]     # (B, num_landmarks, 2)
    comm = state.c[..., :num_agents, :]                 # (B, num_agents, comm_dim)

    obs = {}

    # === 为每个智能体计算观测 ===
    for i in range(num_agents):
        self_pos = agent_pos[..., i, :]                 # (B, 2)
        self_vel = agent_vel[..., i, :]                 # (B, 2)

        # 相对 landmark 位置
        rel_landmark = landmark_pos - self_pos[..., None, :]  # (B, num_landmarks, 2)

        # 相对其他 agent 位置
        other_pos = jnp.concatenate(
            [agent_pos[..., :i, :], agent_pos[..., i + 1:, :]], axis=1
        )                                             # (B, num_agents - 1, 2)
        rel_others = other_pos - self_pos[..., None, :]  # (B, num_agents - 1, 2)

        # 其他 agent 的 communication
        other_comm = jnp.concatenate(
            [comm[..., :i, :], comm[..., i + 1:, :]], axis=1
        )                                             # (B, num_agents - 1, comm_dim)

        # 拼接观测
        obs_i = jnp.concatenate([
            self_vel,                                 # (B, 2)
            self_pos,                                 # (B, 2)
            rel_landmark.reshape(rel_landmark.shape[0], -1),
            rel_others.reshape(rel_others.shape[0], -1),
            other_comm.reshape(other_comm.shape[0], -1),
        ], axis=-1)                                   # (B, obs_dim)

        obs[f"agent_{i}"] = obs_i

    return obs


# -------------------------------
# Networks
# -------------------------------
class SingleDynamics(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int  # = state_dim (predict delta)
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        h = x
        for d in self.hidden_dims:
            h = nn.relu(nn.Dense(d)(h))
        mu = nn.Dense(self.out_dim)(h)
        logvar = nn.Dense(self.out_dim)(h)
        logvar = jnp.clip(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar



class EnsembleTransition(nn.Module):
    num_members: int
    hidden_dims: Sequence[int]
    out_dim: int  # state_dim
    min_logvar: float = -10.0
    max_logvar: float = 0.5

    def setup(self):
        self.member = nn.vmap(
            SingleDynamics,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_members,
        )(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
        )

    def __call__(self, x):
        return self.member(x, axis_name="ensemble")


class RewardNet(nn.Module):
    hidden_dims: Sequence[int]
    num_agents: int

    @nn.compact
    def __call__(self, x):
        h = x
        for d in self.hidden_dims:
            h = nn.relu(nn.Dense(d)(h))
        rew = nn.Dense(self.num_agents)(h)
        return rew

# -------------------------------
# Train utilities
# -------------------------------

def init_model(rng, num_agents, act_dim, opp_dim, cfg):
    core_state_dim = (num_agents*2 + cfg.train.num_landmark) * 2
    # transition model
    transition = EnsembleTransition(
        num_members=cfg.model_dynamics.num_members,
        hidden_dims=tuple(cfg.model_dynamics.hidden_dims),
        out_dim=core_state_dim,
        min_logvar=cfg.model_dynamics.min_logvar,
        max_logvar=cfg.model_dynamics.max_logvar,
    )
    rng, key1 = jax.random.split(rng)
    dummy_in = jnp.zeros((1, core_state_dim + act_dim + opp_dim))
    trans_params = transition.init(key1, dummy_in)["params"]

    # reward model
    reward_model = RewardNet(
        hidden_dims=tuple(cfg.model_dynamics.hidden_dims),
        num_agents=num_agents,
    )
    rng, key2 = jax.random.split(rng)
    rew_params = reward_model.init(key2, dummy_in)["params"]

    tx = optax.adam(cfg.model_dynamics.lr)

    transition_state = TrainState.create(apply_fn=transition.apply, params=trans_params, tx=tx)
    reward_state = TrainState.create(apply_fn=reward_model.apply, params=rew_params, tx=tx)

    return transition, reward_model, transition_state, reward_state


# Loss & Train
def _nll(mu, logvar, target):
    inv_var = jnp.exp(-logvar)
    inv_var = jnp.clip(inv_var, 1e-6, 1e3)
    mse = (mu - target) ** 2
    nll_dim = 0.5 * (mse * inv_var + logvar + jnp.log(2.0 * jnp.pi))
    nll_b = jnp.sum(nll_dim, axis=-1)
    nll = jnp.mean(nll_b, axis=-1)
    return jnp.mean(nll)




def train_transition_step(state, batch, std):
    def loss_fn(params):
        core_state = extract_core_state(batch["state"])
        core_next_state = extract_core_state(batch["next_state"])
        state_n = std.norm_state(core_state)
        a_ego_n = std.norm_a_ego(batch["a_ego"])
        a_opp_n = std.norm_a_opp(batch["a_opp"])
        delta = core_next_state - core_state
        delta_n = std.norm_delta(delta)

        x = jnp.concatenate([state_n, a_ego_n, a_opp_n], axis=-1)
        mu, logvar = state.apply_fn({"params": params}, x)

        target = jnp.tile(delta_n[None, ...], (mu.shape[0], 1, 1))
        nll = _nll(mu, logvar, target)
        mse = jnp.mean((mu - target) ** 2)

        return nll, {"transition_nll": nll, "transition_mse": mse}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(params=new_params, opt_state=opt_state)
    return new_state, metrics


def train_reward_step(state, batch, std):

    def loss_fn(params):
        core_state = extract_core_state(batch["state"])
        state_n = std.norm_state(core_state)
        a_ego_n = std.norm_a_ego(batch["a_ego"])
        a_opp_n = std.norm_a_opp(batch["a_opp"])

        x = jnp.concatenate([state_n, a_ego_n, a_opp_n], axis=-1)   # (B, S+A+O)

        # predicted reward: (B, num_agents)
        rew_pred = state.apply_fn({"params": params}, x)

        # target reward: (B, num_agents)
        # rew_target = jnp.concatenate(
        #     [batch["rew"][f"agent_{i}"].squeeze(-1) for i in range(rew_pred.shape[-1])],
        #     axis=-1
        # )
        def agents_dict_to_array(agent_dict: dict) -> jnp.ndarray:
            # 以 agent_0, agent_1, ... 的顺序稳定堆叠
            keys = sorted(agent_dict.keys(), key=lambda s: int(s.split('_')[-1]))
            arrs = [jnp.asarray(agent_dict[k]).reshape(-1) for k in keys]  # 每个是 (B,)
            return jnp.stack(arrs, axis=-1)  # -> (B, num_agents)

        rew_target = agents_dict_to_array(batch['rew']).astype(jnp.float32)  # (B, num_agents)

        mse = jnp.mean((rew_pred - rew_target) ** 2)

        return mse, {"reward_mse": mse}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    new_state = state.replace(params=new_params, opt_state=opt_state)
    return new_state, metrics

# def train_step(state: TrainState,
#                batch: dict,
#                std: Standardizer):
#     """
#     :param state: dynamics TrainState
#     :param batch: dict with keys {obs, act, next_obs}
#     :param std:   Standardizer (静态)
#     :return: new_state, metrics
#     """
#
#
#
#     def loss_fn(params):
#         # === standardize inputs/targets ===
#         state_n = std.norm_state(batch['state'])
#         a_ego_n = std.norm_a_ego(batch['a_ego'])
#         a_opp_n = std.norm_a_opp(batch['a_opp'])
#         delta = batch['next_state'] - batch['state']
#         delta_n = std.norm_delta(delta)
#
#         # === input ===
#         x = jnp.concatenate([state_n, a_ego_n, a_opp_n], axis=-1)   # (B, obs+act)
#         # === predict of model ===
#         mu, logvar = state.apply_fn({'params': params}, x)  # (E,B,D)
#         # E, B, D_out = mu.shape
#         # === target ===
#         rew_target = agents_dict_to_array(batch['rew']).astype(jnp.float32)  # (B, num_agents)
#         target = jnp.concatenate([delta_n, rew_target], axis=-1)
#         target = jnp.broadcast_to(target, mu.shape)
#         loss = _nll(mu, logvar, target)
#         mse = jnp.mean((mu - target) ** 2)
#         logvar = jnp.mean(logvar)
#         metrics = {"nll": loss, "mse": mse, "logvar": logvar}
#         return loss, metrics
#
#     (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
#     updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
#     new_params = optax.apply_updates(state.params, updates)
#     new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
#     return new_state, metrics
#
# train_step = jax.jit(train_step, static_argnums=())

# compute the entropy of dynamics model
def dynamics_uncertainty_per_dim(mu: jnp.ndarray, logvar: jnp.ndarray):
    """
    Args:
        mu:     (E, B, k)
        logvar: (E, B, k)

    Returns:
        total_var:      (B, k)
        epistemic_var:  (B, k)
        aleatoric_var:  (B, k)
        entropy_dim:    (B, k)
    """
    # aleatoric
    var_e = jnp.exp(logvar)
    prec_e = 1.0 / (var_e + 1e-6)
    aleatoric_var = 1.0 / jnp.mean(prec_e, axis=0)

    # epistemic
    mu_bar  = aleatoric_var * jnp.mean(prec_e * mu, axis=0)
    epistemic_var = jnp.mean((mu - mu_bar[None, ...]) ** 2, axis=0)

    K = aleatoric_var / (aleatoric_var + epistemic_var + 1e-6)
    tilde_var = (1.0 - K) * aleatoric_var
    # total
    total_var = aleatoric_var + epistemic_var

    # per-dim entropy (optional)
    # entropy_dim = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * total_var + 1e-6)
    entropy_dim = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * tilde_var + 1e-6)
    entropy_dim_shifted = entropy_dim - jnp.min(entropy_dim, axis=0, keepdims=True)

    return total_var, epistemic_var, aleatoric_var, entropy_dim_shifted


# Prediction & Evaluation
def predict_next(transition_state: TrainState,
                 reward_state: TrainState,
                 std: Standardizer,
                 state_agent: jnp.ndarray,   # (B, obs_dim)
                 a_ego: jnp.ndarray,   # (B, act_dim)
                 a_opp: jnp.ndarray,
                 cfg: DictConfig,
                 rng: Optional[Any] = None,
                 deterministic: bool = True,
                 member_idx: Optional[int] = None,
                 )-> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Dict[str,jnp.ndarray], Dict[str,jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """
    Return predicted next state s' (denormalized).
    - If member_idx is None: 随机选一个 ensemble 成员（需要 rng）
    - deterministic=True: 使用均值；False: 从 N(mu, var) 采样
    """
    a_ego = jnp.asarray(a_ego)
    a_opp = jnp.asarray(a_opp)
    state_agent = jnp.asarray(state_agent)
    core_state = extract_core_state(state_agent)
    state_agent_n = std.norm_state(core_state)
    a_ego_n = std.norm_a_ego(a_ego)
    a_opp_n = std.norm_a_opp(a_opp)
    x = jnp.concatenate([state_agent_n, a_ego_n, a_opp_n], axis=-1)  # (B, in_dim)

    mu, logvar = transition_state.apply_fn({"params": transition_state.params}, x)   # (E,B,D)
    var_e = jnp.exp(logvar)
    prec_e = 1.0 / (var_e + 1e-6)
    aleatoric_var = 1.0 / jnp.mean(prec_e, axis=0)  # (B,D)
    mu_bar = aleatoric_var * jnp.mean(prec_e * mu, axis=0)  # (B,D)
    epistemic_var = jnp.mean(
        (mu - mu_bar[None, ...]) ** 2, axis=0
    )  # (B,D)


    if member_idx is None:
        assert rng is not None, "predict_next: rng is required when member_idx is None."
        rng, sub = jax.random.split(rng)
        member_idx = jax.random.randint(sub, shape=(), minval=0, maxval=mu.shape[0])
    mu_m = mu[member_idx]       # (B,D)
    logvar_m = logvar[member_idx]
    var_m = var_e[member_idx]
    rng, sub1 = jax.random.split(rng)
    hat_delta_n = mu_m + jnp.sqrt(var_m) * \
                  jax.random.normal(sub1, mu_m.shape)
    K = aleatoric_var / (aleatoric_var + epistemic_var + 1e-6)

    tilde_mu = mu_bar + K * (hat_delta_n - mu_bar)
    tilde_var = (1.0 - K) * aleatoric_var

    # mu_m = jnp.mean(mu, axis=0)
    # var = jnp.exp(logvar)
    # var_m = jnp.mean(var, axis=0)
    # logvar_m = jnp.log(var_m)
    # print("mu.shape", mu.shape)
    # print("logvar.shape", logvar.shape)
    # print("mu_m.shape", mu_m.shape)
    # print("logvar_m.shape", logvar_m.shape)

    if deterministic:
        delta_n = tilde_mu
    else:
        assert rng is not None, "predict_next: rng required for stochastic sampling."
        rng, sub = jax.random.split(rng)
        # stddev = jnp.exp(0.5 * logvar_m)
        # delta_n = mu_m + stddev * jax.random.normal(sub, mu_m.shape)
        delta_n = tilde_mu + jnp.sqrt(tilde_var) * jax.random.normal(sub, tilde_mu.shape)

    reward = reward_state.apply_fn({"params": reward_state.params}, x)
    reward_dict = {f"agent_{i}": reward[..., i:i + 1] for i in range(reward.shape[-1])}

    delta = std.denorm_delta(delta_n)      # (B,D)
    core_next_state_agent = core_state + delta
    next_state_agent = restore_full_state(state_agent, core_next_state_agent, cfg) # 18 to 34

    # from state get obs and dones
    restored_state = manual_unflatten_state(next_state_agent) # 34 to State
    next_obs = get_obs(restored_state)
    dones_pred = next_state_agent[..., -4:-1]
    dones_bool = dones_pred > 0.0
    dones_dict = {f"agent_{i}": dones_bool[..., i:i+1] for i in range(dones_pred.shape[-1])}
    return next_state_agent, next_obs, reward_dict, dones_dict, mu, logvar

def eval_error(real_state:TrainState,###############
               opp_state: TrainState,###################
               std: Standardizer,
               batch: dict,
               rng: Optional[Any] = None,
               deterministic: bool = True,
               member_idx: Optional[int] = None)-> jnp.ndarray:
    """
    Evaluate model prediction error (MSE) on a given batch of real transitions.
    Used to measure model accuracy or adaptive rollout length in AORPO.
    """

    def kl_normal(mu_p, sigma_p, mu_q, sigma_q, eps=1e-6):
        sigma_p = jnp.clip(sigma_p, eps, 1e6)
        sigma_q = jnp.clip(sigma_q, eps, 1e6)
        return jnp.log(sigma_q / sigma_p) + ((sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2.0 * sigma_q ** 2)) - 0.5

    mu_real, std_real = real_state.apply_fn({"params": real_state.params}, batch["obs"]["agent_0"])
    mu_opp, std_opp = opp_state.apply_fn({"params":opp_state.params}, batch["obs"][f"agent_{member_idx+1}"])
    std_real = jnp.exp(jnp.clip(std_real, -10.0, 2.0))  # [exp(-10), exp(2)] ≈ [4.5e-5, 7.4]
    std_opp = jnp.exp(jnp.clip(std_opp, -10.0, 2.0))
    kl = kl_normal(mu_real, std_real, mu_opp, std_opp)
    kl = jnp.maximum(jnp.sum(kl, axis=-1), 0.0)
    tv = jnp.sqrt(0.5 * kl)
    tv = jnp.mean(tv)
    return tv # shape:[batch_size]
