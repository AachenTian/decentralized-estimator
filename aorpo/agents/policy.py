from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Any
from flax.training.train_state import TrainState
from flax import struct
import optax
# from jaxmarl.tutorials.smax_introduction import init_policy
from omegaconf import DictConfig


class TrainStateid(TrainState):
    agent_id: str = struct.field(pytree_node=False)  #

class PolicyNet(nn.Module):
    action_dim:int                              #action dimension
    hidden_dims:Sequence[int] = (256,256)       #hidden dimension
    min_logvar: float = -5.0
    max_logvar: float = 2.0


    @nn.compact
    def __call__(self,obs):
        x = obs
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        mu = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.min_logvar, self.max_logvar)
        return mu, log_std


    @staticmethod
    def sample_action(params, apply_fn, rng, obs):
        """Sample an action from the policy (Gaussian & tanh)."""
        mu, log_std = apply_fn({"params": params}, obs)
        std = jnp.exp(log_std)
        rng, subkey = jax.random.split(rng)
        normal_sample = mu + std * jax.random.normal(subkey, mu.shape)
        action = jnp.tanh(normal_sample)
        log_prob = -0.5 * (((normal_sample -mu) ** 2 / (std + 1e-6) ** 2) + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = jnp.sum(log_prob, axis=-1)
        log_prob -=jnp.sum(jnp.log(1- action ** 2 + 1e-6), axis=-1)

        return action, log_prob, rng

    @staticmethod
    def deterministic_action(params, apply_fn, obs):
        """
        Deterministic policy action: tanh(mu)
        This is used for evaluation, not for exploration or training.

        params: policy parameters
        apply_fn: policy_state.apply_fn
        obs: observation tensor (B, obs_dim)
        """
        mu, log_std = apply_fn({"params": params}, obs)
        action = jnp.tanh(mu)
        return action

class EnsemblePolicyNet(nn.Module):
    num_members: int
    action_dim: int
    hidden_dims: Sequence[int]
    min_logvar: float
    max_logvar: float

    def setup(self):
        self.member = nn.vmap(
            PolicyNet,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,        # obs 不带 ensemble 维
            out_axes=0,          # 输出带 ensemble 维
            axis_size=self.num_members,
        )(
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            min_logvar=self.min_logvar,
            max_logvar=self.max_logvar,
        )

    def __call__(self, obs):
        # obs: (B, obs_dim) 或 (obs_dim,)
        # return:
        #   mu:      (K, B, act_dim)
        #   log_std: (K, B, act_dim)
        return self.member(obs)


def init_policy_model(rng: Any,
                      obs_dim: int,
                      act_dim: int,
                      cfg: DictConfig,
                      agent_id: str
                      ):
    model = PolicyNet(
        action_dim=act_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        min_logvar=cfg.min_logvar,
        max_logvar=cfg.max_logvar
    )

    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(init_rng, dummy_obs)["params"]
    tx = optax.adam(cfg.lr)
#     tx = optax.chain(
#     optax.clip_by_global_norm(1.0),     # <—— 加这个！
#     optax.adam(cfg.lr)
# )

    state = TrainStateid.create(
        apply_fn = model.apply,
        params = params,
        tx = tx,
        agent_id = agent_id
    )
    return model, state

def init_policy_ensemble(
    rng: Any,
    obs_dim: int,
    act_dim: int,
    cfg: DictConfig,
    agent_id: str,
):
    # 1 用 ensemble policy（不是 PolicyNet）
    model = EnsemblePolicyNet(
        num_members=cfg.ensemble_size,
        action_dim=act_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        min_logvar=cfg.min_logvar,
        max_logvar=cfg.max_logvar,
    )

    # 2️ 只 init 一次（不要 vmap）
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((obs_dim,))
    params = model.init(init_rng, dummy_obs)["params"]

    # 3️ optimizer（不用 vmap）
    tx = optax.adam(cfg.lr)

    # 4️ TrainState
    state = TrainStateid.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        agent_id=agent_id,
    )

    return model, state



class EnsemblePolicyUtils:
    @staticmethod
    def sample_action_ensemble(params, apply_fn, rng, obs):
        """
        obs: (obs_dim,) or (B, obs_dim)

        returns:
          action: (act_dim,) or (B, act_dim)        # 用于 rollout
          mu:      (K, ..., act_dim)                # 用于 uncertainty
          log_std: (K, ..., act_dim)
          rng
        """
        # 1️ 直接 forward（已经是 ensemble）
        mu, log_std = apply_fn({"params": params}, obs)
        # mu: (K, ..., act_dim)

        # 2️ 合成 rollout 用的分布（Infoprop / MBPO 常用）
        mu_bar = jnp.mean(mu, axis=0)                         # (…, act_dim)
        var_ale = jnp.mean(jnp.exp(2 * log_std), axis=0)
        std_bar = jnp.sqrt(var_ale + 1e-6)

        # 3️ 采样一个 action
        rng, subkey = jax.random.split(rng)
        eps = jax.random.normal(subkey, mu_bar.shape)
        action = jnp.tanh(mu_bar + std_bar * eps)

        return action, mu, log_std, rng


    @staticmethod
    def sample_deterministic_action_ensemble(params, apply_fn, obs):
        mu, log_std = apply_fn({"params": params}, obs)
        mu_mean = jnp.mean(mu, axis=0)
        action = jnp.tanh(mu_mean)
        return action, mu

    @staticmethod
    def get_infoprop_entropy(mu, log_std):
        """
        mu, log_std: (K, ..., act_dim)
        """
        # Aleatoric variance
        var_ale = jnp.exp(2 * log_std)

        avg_mu = jnp.mean(mu, axis=0)
        epistemic_var = jnp.mean((mu - avg_mu) ** 2, axis=0)
        avg_ale_var = jnp.mean(var_ale, axis=0)

        total_var = epistemic_var + avg_ale_var

        entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * total_var + 1e-6)
        return entropy
