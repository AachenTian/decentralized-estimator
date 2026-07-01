# aorpo/rollout/uncertainty_threshold.py
from __future__ import annotations
import jax
import jax.numpy as jnp

from aorpo.agents.model_dynamics import extract_core_state, dynamics_uncertainty_per_dim

# -----------------------------------------------------
# uncertainty threshold of opponent model action
# -----------------------------------------------------
# 使用 partial 固定非 Array 参数，确保 jit 正常工作

def _compute_threshold_engine(opponent_states, obs_all, cfg):
    """
    内部核心引擎：对所有 opponent 计算动作熵的分位数阈值

    opponent_states: List[TrainState]，每个 state.apply_fn 是 EnsemblePolicyNet.apply
    obs_all: (num_opps, batch_size, obs_dim)
    return: (num_opps, act_dim)
    """
    num_opps = len(opponent_states)

    def get_single_opp_entropy(opp_state, single_opp_obs):
        """
        opp_state: 单个 opponent TrainState (ensemble policy)
        single_opp_obs: (B, obs_dim)
        return: entropies (B, act_dim)
        """
        # ✅ 关键：直接一次 apply，就得到 ensemble 输出
        # mu_ens/log_std_ens: (K, B, act_dim)
        mu_ens, log_std_ens = opp_state.apply_fn(
            {"params": opp_state.params},
            single_opp_obs
        )

        # ✅ 方差：sigma^2 = exp(2 * log_std)
        var_ale = jnp.exp(2.0 * log_std_ens)          # (K, B, act_dim)
        avg_ale_var = jnp.mean(var_ale, axis=0)       # (B, act_dim)

        avg_mu = jnp.mean(mu_ens, axis=0)             # (B, act_dim)
        epistemic_var = jnp.mean((mu_ens - avg_mu) ** 2, axis=0)  # (B, act_dim)

        total_var = avg_ale_var + epistemic_var       # (B, act_dim)

        # (B, act_dim) 维度熵
        ent = 0.5 * jnp.log(2 * jnp.pi * jnp.e * total_var + 1e-6)
        return ent

    # ✅ 外层不建议用 vmap 因为 opp_state 是 Python object（含 apply_fn method）
    # 用 for-loop 最稳（num_opps 很小，代价可忽略）
    all_entropies = []
    for j in range(num_opps):
        ent_j = get_single_opp_entropy(opponent_states[j], obs_all[j])  # (B, act_dim)
        all_entropies.append(ent_j)
    all_entropies = jnp.stack(all_entropies, axis=0)  # (num_opps, B, act_dim)

    # ✅ 对 batch 维度做分位数，得到每个 opponent 的阈值 lambda_j
    lambda_all = jnp.quantile(all_entropies, q=cfg.rollout.quantile_opp, axis=1)
    # shape: (num_opps, act_dim)
    return lambda_all

# opponents agents' action uncertainty threshold
def compute_opponent_threshold(opponent_states, replay_env, cfg):
    """
    opponent_states: List[TrainState] (每个都是 ensemble policy state)
    return: (num_opps, act_dim)
    """
    batch_size = cfg.rollout.batch_size
    rng = jax.random.PRNGKey(0)

    batch = replay_env.sample(
        rng,
        batch_size=batch_size,
        opp_num=cfg.train.num_opponents
    )

    obs_list = [batch["obs"][f"agent_{j + 1}"] for j in range(cfg.train.num_opponents)]
    obs_all = jnp.stack(obs_list, axis=0)  # (num_opps, B, obs_dim)

    # ✅ 不再需要 combined_params（因为 apply_fn 已经处理 ensemble params）
    lambda_all = _compute_threshold_engine(
        opponent_states=opponent_states,
        obs_all=obs_all,
        cfg=cfg
    )

    return lambda_all

def gaussian_entropy_from_logvar(mu:jnp.ndarray, logvar: jnp.ndarray):
    """
    logvar: (E, B, D) or (B, D)
    return: (B, D)
    """
    var_ale = jnp.exp(logvar)

    # 若是 ensemble，先平均 epistemic + aleatoric
    if var_ale.ndim == 3:  # (E, B, D)
        avg_var_ale = jnp.mean(var_ale, axis=0)

    mu_bar = jnp.mean(mu, axis=0)  # (B, D)
    epi_var = jnp.mean((mu - mu_bar) ** 2, axis=0)  # (B, D)
    total_var = avg_var_ale + epi_var  # (B, D)

    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * total_var + 1e-6)
    return entropy, total_var

def compute_lambda1(entropies: jnp.ndarray, zeta1=0.99):
    """
    entropies: (N, D)  # 来自 D_env 的 entropy 样本
    return: (D,)
    """
    return jnp.quantile(entropies, zeta1, axis=0)

def compute_total_var_threshold(total_var: jnp.ndarray, zeta1):
    return jnp.quantile(total_var, zeta1, axis=0)

def compute_lambda2(entropies: jnp.ndarray, zeta2=0.01, xi=100.0):
    """
    entropies: (N, D)
    return: (D,)
    """
    base = jnp.quantile(entropies, zeta2, axis=0)
    return base + jnp.log(xi)

def compute_entropy_thresholds(
    transition_state,
    std,
    replay_env,
    cfg,
    rng,
    batch_size=2048,
):
    batch = replay_env.sample(rng, batch_size=batch_size, opp_num=cfg.train.num_opponents)

    # 准备输入
    core_state = extract_core_state(batch["state"])
    a_ego = batch["a_ego"]
    a_opp = batch["a_opp"]

    # 标准化
    state_n = std.norm_state(core_state)
    a_ego_n = std.norm_a_ego(a_ego)
    a_opp_n = std.norm_a_opp(a_opp)
    x = jnp.concatenate([state_n, a_ego_n, a_opp_n], axis=-1)

    mu, logvar = transition_state.apply_fn({"params": transition_state.params}, x)

    # entropy: (B, D)
    total_var, _, _, entropy = dynamics_uncertainty_per_dim(mu, logvar)

    total_var_threshold = compute_total_var_threshold(total_var, zeta1=cfg.rollout.zeta1)
    lambda1 = jnp.quantile(entropy, cfg.rollout.zeta1, axis=0)
    lambda2 = jnp.quantile(entropy, cfg.rollout.zeta2, axis=0) * cfg.rollout.xi

    return lambda1, lambda2, total_var_threshold
