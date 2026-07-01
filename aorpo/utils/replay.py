# aorpo/utils/replay.py
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from omegaconf import DictConfig
from typing import Callable, Any
from jax.flatten_util import ravel_pytree


def manual_flatten_state(state):
    # 展开所有字段
    p_pos_flat = state.p_pos.reshape(-1)       # (num_obj*2,)
    p_vel_flat = state.p_vel.reshape(-1)       # (num_obj*2,)
    c_flat     = state.c.reshape(-1)           # (num_agents*2,)
    done_flat  = state.done.astype(jnp.float32).reshape(-1)  # (num_agents,)
    step_flat  = jnp.array([state.step], dtype=jnp.float32)  # scalar
    # 拼到一起 → 固定顺序
    flat = jnp.concatenate([
        p_pos_flat,
        p_vel_flat,
        c_flat,
        done_flat,
        step_flat
    ], axis=0)
    return flat

def manual_flatten_dict(state):
    # 展开所有字段
    p_pos_flat = state["p_pos"].reshape(-1)  # (num_obj*2,)
    p_vel_flat = state["p_vel"].reshape(-1)  # (num_obj*2,)
    c_flat = state["c"].reshape(-1)  # (num_agents*2,)
    done_flat = state["done"].astype(jnp.float32).reshape(-1)  # (num_agents,)
    step_flat = jnp.array([state["step"]], dtype=jnp.float32)  # scalar
    # 拼到一起 → 固定顺序
    flat = jnp.concatenate([
        p_pos_flat,
        p_vel_flat,
        c_flat,
        done_flat,
        step_flat
    ], axis=0)
    return flat



@dataclass
class ReplayBuffer:
    state: jnp.ndarray
    obs: jnp.ndarray
    a_ego: jnp.ndarray
    a_opp: jnp.ndarray
    next_state: jnp.ndarray
    next_obs: jnp.ndarray
    rew: jnp.ndarray
    dones: jnp.ndarray
    max_size: int
    size: int
    ptr: int
    unravel_fn: Callable[[jnp.ndarray], Any] = None

    @staticmethod
    def create(max_size, obs_dim, act_dim, opp_num, state_dim):
        """初始化空buffer"""
        state = jnp.zeros((max_size, state_dim), dtype=jnp.float32)
        next_state = jnp.zeros((max_size, state_dim), dtype=jnp.float32)
        obs = jnp.zeros((max_size, obs_dim*(opp_num+1)), dtype=jnp.float32)
        a_ego = jnp.zeros((max_size, act_dim), dtype=jnp.float32)
        a_opp = jnp.zeros((max_size, act_dim*opp_num), dtype=jnp.float32)
        next_obs = jnp.zeros((max_size, obs_dim*(opp_num+1)), dtype=jnp.float32)
        rew = jnp.zeros((max_size, 3), dtype=jnp.float32)
        dones = jnp.zeros((max_size, 3), dtype=jnp.bool)
        return ReplayBuffer(state, obs, a_ego, a_opp, next_state, next_obs, rew, dones, max_size, 0, 0)

    @staticmethod
    def flatten_agents_dict(agent_dict: dict[str, jnp.ndarray]) -> jnp.ndarray:
        arrays = [jnp.atleast_1d(v) for k, v in sorted(agent_dict.items()) if k != "__all__"]
        arrays = [v.reshape(-1, 1) if v.ndim == 1 else v for v in arrays]  # 确保有 batch 维
        return jnp.stack(arrays) if arrays[0].ndim == 0 else jnp.concatenate(arrays, axis=-1)

    def unflatten_state(self, flat_state):
        """还原为原始 State"""
        if self.unravel_fn is None:
            raise ValueError("unravel_fn not set — please flatten at least once first.")
        return jax.vmap(self.unravel_fn)(flat_state)

    def flatten_one(self, state_t):
        state_t = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bool_ else x,
            state_t
        )
        flat, unravel_fn = ravel_pytree(state_t)
        return flat

    def add_batch(self, batch, cfg:DictConfig):
        """Add a batch (dict of jnp arrays) to replay buffer."""
        B = batch["obs"]["agent_0"].shape[0]
        idx = (jnp.arange(B) + self.ptr) % self.max_size
        flatten_fn = jax.vmap(manual_flatten_state)
        flat_state = flatten_fn(batch["state"]) # type: ignore
        flat_next_state = flatten_fn(batch["next_state"]) # type: ignore
        flat_obs = ReplayBuffer.flatten_agents_dict(batch["obs"])
        flat_a_opp = batch["a_opp"]
        flat_next_obs = ReplayBuffer.flatten_agents_dict(batch["next_obs"])

        new_state = self.state.at[idx].set(flat_state)
        new_next_state = self.next_state.at[idx].set(flat_next_state)
        new_obs = self.obs.at[idx].set(flat_obs)
        new_a_ego = self.a_ego.at[idx].set(batch["a_ego"])
        new_a_opp = self.a_opp.at[idx].set(flat_a_opp)
        new_next_obs = self.next_obs.at[idx].set(flat_next_obs)

        # --- 确保 reward 和 done 的形状一致 ---
        flat_rew = ReplayBuffer.flatten_agents_dict(batch["rew"])
        if flat_rew.ndim == 1:
            flat_rew = flat_rew.reshape(-1, 1)
        new_rew = self.rew.at[idx].set(flat_rew)

        flat_dones = ReplayBuffer.flatten_agents_dict(batch["dones"])
        # --- 如果没有 done，就创建一个 zeros ---
        if "dones" in batch:
            dones = flat_dones
            if dones.ndim == 1:
                dones = dones.reshape(-1, 1)
        else:
            dones = jnp.zeros_like(flat_rew)
        new_dones = self.dones.at[idx].set(dones)

        # --- 更新指针 ---
        new_ptr = int((self.ptr + B) % self.max_size)
        new_size = int(jnp.minimum(self.size + B, self.max_size))

        return ReplayBuffer(
            new_state, new_obs, new_a_ego, new_a_opp, new_next_state, new_next_obs, new_rew, new_dones,
            self.max_size, new_size, new_ptr
        )
    def add_batch_model(self, batch, cfg:DictConfig):
        """Add a batch (dict of jnp arrays) to replay buffer."""
        B = batch["obs"]["agent_0"].shape[0]
        idx = (jnp.arange(B) + self.ptr) % self.max_size
        flat_state = batch["state"] # type: ignore
        flat_next_state = batch["next_state"] # type: ignore
        flat_obs = ReplayBuffer.flatten_agents_dict(batch["obs"])
        flat_a_opp = batch["a_opp"]
        flat_next_obs = ReplayBuffer.flatten_agents_dict(batch["next_obs"])

        new_state = self.state.at[idx].set(flat_state)
        new_next_state = self.next_state.at[idx].set(flat_next_state)
        new_obs = self.obs.at[idx].set(flat_obs)
        new_a_ego = self.a_ego.at[idx].set(batch["a_ego"])
        new_a_opp = self.a_opp.at[idx].set(flat_a_opp)
        new_next_obs = self.next_obs.at[idx].set(flat_next_obs)

        # --- 确保 reward 和 done 的形状一致 ---
        flat_rew = ReplayBuffer.flatten_agents_dict(batch["rew"])
        if flat_rew.ndim == 1:
            flat_rew = flat_rew.reshape(-1, 1)
        new_rew = self.rew.at[idx].set(flat_rew)

        flat_dones = ReplayBuffer.flatten_agents_dict(batch["dones"])
        # --- 如果没有 done，就创建一个 zeros ---
        if "dones" in batch:
            dones = flat_dones
            if dones.ndim == 1:
                dones = dones.reshape(-1, 1)
        else:
            dones = jnp.zeros_like(flat_rew)
        new_dones = self.dones.at[idx].set(dones)

        # --- 更新指针 ---
        new_ptr = int((self.ptr + B) % self.max_size)
        new_size = int(jnp.minimum(self.size + B, self.max_size))

        return ReplayBuffer(
            new_state, new_obs, new_a_ego, new_a_opp, new_next_state, new_next_obs, new_rew, new_dones,
            self.max_size, new_size, new_ptr
        )

    @staticmethod
    def unflatten_agents_dict(flat_array: jnp.ndarray, num_agents: int) -> dict[str, jnp.ndarray]:
        """
        把展平的数组自动还原为 {'agent_0': ..., 'agent_1': ...} 的字典。
        自动平均切分，不需要指定 agent_dim。

        参数：
            flat_array: 一维或二维的 JAX 数组
            num_agents: agent 数量

        返回：
            dict[str, jnp.ndarray]
        """
        total_dim = flat_array.shape[-1]
        assert total_dim % num_agents == 0, f"无法将长度 {total_dim} 平均分成 {num_agents} 份"
        agent_dim = total_dim // num_agents

        agent_dict = {}
        for i in range(num_agents):
            start, end = i * agent_dim, (i + 1) * agent_dim
            agent_dict[f"agent_{i}"] = flat_array[..., start:end]
        return agent_dict


    def sample(self, key, batch_size, opp_num):
        """JAX 随机采样"""
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        batch = dict(
            state=self.state[idx],
            obs=ReplayBuffer.unflatten_agents_dict(self.obs[idx], opp_num+1),
            a_ego=self.a_ego[idx],
            a_opp=self.a_opp[idx],
            next_state=self.next_state[idx],
            next_obs=ReplayBuffer.unflatten_agents_dict(self.next_obs[idx], opp_num+1),
            rew=ReplayBuffer.unflatten_agents_dict(self.rew[idx], opp_num+1),
            dones=ReplayBuffer.unflatten_agents_dict(self.dones[idx], opp_num+1),
        )
        return batch

    def __len__(self):
        return int(self.size)
