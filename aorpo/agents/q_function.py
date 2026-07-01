import jax
from flax import linen as nn
import jax.numpy as jnp
import optax
from typing import Sequence
from flax.training.train_state import TrainState
from omegaconf import DictConfig


class QNet(nn.Module):
    hidden_dims: Sequence[int] = (256,256)

    @nn.compact
    def __call__(self, state, act):
        """Forward pass for Q(s, a)."""

        x = jnp.concatenate([state, act], axis=-1)

        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        q_value = nn.Dense(1)(x)
        return jnp.squeeze(q_value, -1)


def init_q_function(rng, state_dim:int, act_dim: int, cfg: DictConfig):

    model = QNet(hidden_dims=tuple(cfg.hidden_dims))
    dummy_obs = jnp.zeros((1, state_dim))
    dummy_act = jnp.zeros((1, act_dim * cfg.agent_num))
    params = model.init(rng, dummy_obs, dummy_act)['params']
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr))
    state = TrainState.create(
        apply_fn = model.apply,
        params=params,
        tx=tx
    )
    return model, state