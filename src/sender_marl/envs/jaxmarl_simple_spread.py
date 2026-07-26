from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp
from jaxmarl import make

from sender_marl.core.specs import EnvSpec
from sender_marl.core.types import EnvFeatures, EnvStep
from sender_marl.envs.registry import register_env


Array = jax.Array
RemoteFeatureMode = Literal["relative_position", "relative_state"]
ActionMode = Literal["force_2d", "native"]


@dataclass(frozen=True)
class SimpleSpreadConfig:
    """Configuration specific to the JaxMARL Simple Spread adapter."""

    env_name: str = "MPE_simple_spread_v3"
    num_agents: int = 3
    num_landmarks: int = 3
    max_steps: int = 25
    local_ratio: float = 0.5
    action_mode: ActionMode = "force_2d"
    remote_feature_mode: RemoteFeatureMode = "relative_state"
    u_noise: float | tuple[float, ...] = 0.0
    collision_distance: float = 0.30

    def __post_init__(self) -> None:
        if self.num_agents < 2:
            raise ValueError("Simple Spread requires at least two agents.")
        if self.num_landmarks < 1:
            raise ValueError("num_landmarks must be positive.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if self.collision_distance <= 0.0:
            raise ValueError("collision_distance must be positive.")
        if self.action_mode not in ("force_2d", "native"):
            raise ValueError(f"Unsupported action_mode={self.action_mode!r}.")
        if self.remote_feature_mode not in (
            "relative_position",
            "relative_state",
        ):
            raise ValueError(
                f"Unsupported remote_feature_mode={self.remote_feature_mode!r}."
            )


class JaxMARLSimpleSpreadAdapter:
    """JAX-native Simple Spread adapter with an environment-agnostic contract.

    Environment-specific details remain in this file. The policy algorithms see
    only EnvSpec, EnvFeatures, standardized actions, and constructed observations.
    """

    def __init__(self, config: SimpleSpreadConfig | None = None, **kwargs: Any):
        if config is not None and kwargs:
            raise ValueError("Pass either config or keyword arguments, not both.")
        self.config = config if config is not None else SimpleSpreadConfig(**kwargs)

        # JaxMARL releases have used slightly different constructor signatures.
        # Try the richer signature first, then fall back to the stable core args.
        u_noise = jnp.asarray(self.config.u_noise, dtype=jnp.float32)
        if u_noise.ndim == 0:
            u_noise = jnp.full(
                (self.config.num_agents,),
                u_noise,
                dtype=jnp.float32,
            )
        elif u_noise.shape != (self.config.num_agents,):
            raise ValueError(
                "u_noise must be a scalar or have one value per agent; "
                f"got shape={u_noise.shape} for "
                f"num_agents={self.config.num_agents}."
            )

        rich_kwargs = dict(
            num_agents=self.config.num_agents,
            num_landmarks=self.config.num_landmarks,
            local_ratio=self.config.local_ratio,
            action_type="Continuous",
            max_steps=self.config.max_steps,
            u_noise=u_noise,
        )
        try:
            self.env = make(self.config.env_name, **rich_kwargs)
        except TypeError:
            fallback_kwargs = dict(
                action_type="Continuous",
                u_noise=u_noise,
            )
            self.env = make(self.config.env_name, **fallback_kwargs)

        self.agents = tuple(self.env.agents)
        if len(self.agents) != self.config.num_agents:
            raise ValueError(
                "Configured num_agents does not match the created JaxMARL "
                f"environment: {self.config.num_agents} vs {len(self.agents)}."
            )

        action_space = self.env.action_space(self.agents[0])
        native_shape = tuple(int(v) for v in action_space.shape)
        if len(native_shape) != 1:
            raise ValueError(
                "Expected a one-dimensional per-agent action space, got "
                f"shape={native_shape}."
            )
        native_action_dim = native_shape[0]

        if self.config.action_mode == "force_2d":
            if native_action_dim != 5:
                raise ValueError(
                    "force_2d expects JaxMARL MPE's five-channel continuous "
                    f"action, but the native dimension is {native_action_dim}. "
                    "Use action_mode='native' for another action layout."
                )
            policy_action_dim = 2
        else:
            policy_action_dim = native_action_dim

        private_obs_dim = 4  # ego [px, py, vx, vy]
        public_context_dim = 2 * self.config.num_landmarks
        local_state_dim = 4

        if self.config.remote_feature_mode == "relative_position":
            remote_per_agent_dim = 2
        else:
            remote_per_agent_dim = local_state_dim

        actor_obs_dim = (
            private_obs_dim
            + public_context_dim
            + (self.config.num_agents - 1) * remote_per_agent_dim
        )
        critic_obs_dim = (
            self.config.num_agents * local_state_dim
            + self.config.num_landmarks * 2
        )

        self.spec = EnvSpec(
            env_name="jaxmarl_simple_spread",
            num_agents=self.config.num_agents,
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            private_obs_dim=private_obs_dim,
            public_context_dim=public_context_dim,
            local_state_dim=local_state_dim,
            model_state_dim=critic_obs_dim,
            policy_action_dim=policy_action_dim,
            native_action_dim=native_action_dim,
            homogeneous_agents=True,
            shared_reward=True,
        )

    def reset(self, key: Array) -> tuple[Any, EnvFeatures]:
        _native_obs, env_state = self.env.reset(key)
        return env_state, self._extract_features(env_state)

    def step(
        self,
        key: Array,
        env_state: Any,
        policy_actions: Array,
    ) -> EnvStep:
        native_actions = self.encode_policy_actions(policy_actions)
        _native_obs, next_state, rewards, dones, _infos = self.env.step(
            key,
            env_state,
            native_actions,
        )

        reward_array = self._stack_agent_mapping(rewards)
        done_array = self._stack_agent_mapping(dones).astype(jnp.bool_)
        episode_done = self._episode_done(dones, done_array)
        features = self._extract_features(next_state).replace(
            alive=jnp.logical_not(done_array).astype(jnp.float32)
        )

        collision_metrics = self.compute_collision_metrics(
            features.local_states
        )
        metrics: Mapping[str, Array] = {
            "team_reward": jnp.sum(reward_array, axis=-1),
            "mean_reward": jnp.mean(reward_array, axis=-1),
            **collision_metrics,
        }
        return EnvStep(
            env_state=next_state,
            features=features,
            rewards=reward_array,
            dones=done_array,
            episode_done=episode_done,
            metrics=metrics,
        )

    def encode_policy_actions(self, policy_actions: Array) -> dict[str, Array]:
        actions = jnp.asarray(policy_actions, dtype=jnp.float32)
        expected_tail = (self.spec.num_agents, self.spec.policy_action_dim)
        if actions.shape[-2:] != expected_tail:
            raise ValueError(
                "policy_actions must end with "
                f"{expected_tail}, got shape={actions.shape}."
            )

        actions = jnp.clip(actions, -1.0, 1.0)
        if self.config.action_mode == "force_2d":
            native_array = self._force_2d_to_mpe_5d(actions)
        else:
            # Native MPE continuous actions are non-negative channels.
            native_array = 0.5 * (actions + 1.0)

        return {
            agent: native_array[..., index, :]
            for index, agent in enumerate(self.agents)
        }

    def build_actor_observations(
        self,
        features: EnvFeatures,
        common_local_states: Array,
    ) -> Array:
        common = jnp.asarray(common_local_states, dtype=jnp.float32)
        expected_tail = (self.spec.num_agents, self.spec.local_state_dim)
        if common.shape[-2:] != expected_tail:
            raise ValueError(
                "common_local_states must end with "
                f"{expected_tail}, got shape={common.shape}."
            )

        ego_pos = features.local_states[..., :, :2]
        common_pos = common[..., :, :2]

        pairwise_relative_pos = (
            common_pos[..., None, :, :] - ego_pos[..., :, None, :]
        )

        if self.config.remote_feature_mode == "relative_position":
            pairwise_features = pairwise_relative_pos
        else:
            common_vel = common[..., :, 2:4]
            vel_for_each_ego = jnp.broadcast_to(
                common_vel[..., None, :, :],
                pairwise_relative_pos.shape,
            )
            pairwise_features = jnp.concatenate(
                [pairwise_relative_pos, vel_for_each_ego],
                axis=-1,
            )

        remote_features = self._remove_self_pairwise(pairwise_features)
        actor_obs = jnp.concatenate(
            [features.private_obs, features.public_context, remote_features],
            axis=-1,
        )

        if actor_obs.shape[-1] != self.spec.actor_obs_dim:
            raise ValueError(
                "Internal actor observation dimension mismatch: "
                f"expected {self.spec.actor_obs_dim}, got {actor_obs.shape[-1]}."
            )
        return actor_obs

    def build_critic_observations(self, features: EnvFeatures) -> Array:
        return features.critic_state

    def _extract_features(self, env_state: Any) -> EnvFeatures:
        positions = jnp.asarray(env_state.p_pos, dtype=jnp.float32)
        velocities = jnp.asarray(env_state.p_vel, dtype=jnp.float32)

        expected_entities = self.config.num_agents + self.config.num_landmarks
        if positions.shape[-2] < expected_entities:
            raise ValueError(
                "JaxMARL state contains fewer entities than configured: "
                f"shape={positions.shape}, expected at least {expected_entities}."
            )

        agent_pos = positions[..., : self.config.num_agents, :]
        agent_vel = velocities[..., : self.config.num_agents, :]
        landmark_pos = positions[
            ...,
            self.config.num_agents : expected_entities,
            :,
        ]

        local_states = jnp.concatenate([agent_pos, agent_vel], axis=-1)
        private_obs = local_states

        landmark_relative = landmark_pos[..., None, :, :] - agent_pos[..., :, None, :]
        public_context = landmark_relative.reshape(
            *landmark_relative.shape[:-2],
            self.spec.public_context_dim,
        )

        critic_state = jnp.concatenate(
            [
                local_states.reshape(*local_states.shape[:-2], -1),
                landmark_pos.reshape(*landmark_pos.shape[:-2], -1),
            ],
            axis=-1,
        )

        alive = jnp.ones(
            local_states.shape[:-1],
            dtype=jnp.float32,
        )
        return EnvFeatures(
            private_obs=private_obs,
            public_context=public_context,
            local_states=local_states,
            critic_state=critic_state,
            model_state=critic_state,
            alive=alive,
        )


    def compute_collision_metrics(
        self,
        local_states: Array,
    ) -> dict[str, Array]:
        """Compute Simple-Spread collision diagnostics.

        Collision geometry is intentionally kept inside this environment
        adapter. Generic policy and training code only consumes the returned
        metrics and does not assume that positions occupy particular state
        coordinates.

        Args:
            local_states: (..., num_agents, local_state_dim). For this adapter,
                the first two local-state entries are planar position.

        Returns:
            collision_count: Number of colliding unordered agent pairs.
            pair_collision_rate: Fraction of all unordered pairs colliding.
            any_collision: Whether at least one pair collides.
            min_pair_distance: Minimum distance between two distinct agents.
        """

        states = jnp.asarray(local_states, dtype=jnp.float32)
        expected_tail = (self.spec.num_agents, self.spec.local_state_dim)
        if states.shape[-2:] != expected_tail:
            raise ValueError(
                "local_states must end with "
                f"{expected_tail}, got shape={states.shape}."
            )

        positions = states[..., :, :2]
        relative = positions[..., :, None, :] - positions[..., None, :, :]
        distances = jnp.linalg.norm(relative, axis=-1)

        n = self.spec.num_agents
        upper_triangle = jnp.triu(
            jnp.ones((n, n), dtype=jnp.bool_),
            k=1,
        )
        colliding_pairs = (
            distances < self.config.collision_distance
        ) & upper_triangle

        collision_count = jnp.sum(colliding_pairs, axis=(-2, -1))
        num_pairs = n * (n - 1) // 2
        pair_collision_rate = collision_count.astype(jnp.float32) / float(
            num_pairs
        )
        any_collision = collision_count > 0

        masked_distances = jnp.where(
            upper_triangle,
            distances,
            jnp.inf,
        )
        min_pair_distance = jnp.min(masked_distances, axis=(-2, -1))

        return {
            "collision_count": collision_count,
            "pair_collision_rate": pair_collision_rate,
            "any_collision": any_collision,
            "min_pair_distance": min_pair_distance,
        }

    def _stack_agent_mapping(self, values: Mapping[str, Array]) -> Array:
        return jnp.stack([jnp.asarray(values[agent]) for agent in self.agents], axis=-1)

    @staticmethod
    def _episode_done(dones: Mapping[str, Array], done_array: Array) -> Array:
        if "__all__" in dones:
            return jnp.asarray(dones["__all__"], dtype=jnp.bool_)
        return jnp.all(done_array, axis=-1)

    @staticmethod
    def _force_2d_to_mpe_5d(actions: Array) -> Array:
        ux = actions[..., 0]
        uy = actions[..., 1]
        noop = jnp.zeros_like(ux)
        return jnp.stack(
            [
                noop,
                jnp.maximum(-ux, 0.0),
                jnp.maximum(ux, 0.0),
                jnp.maximum(-uy, 0.0),
                jnp.maximum(uy, 0.0),
            ],
            axis=-1,
        )

    def _remove_self_pairwise(self, pairwise: Array) -> Array:
        """Remove diagonal j=i and flatten remote-agent features per ego agent."""

        n = self.spec.num_agents
        feature_dim = pairwise.shape[-1]
        mask = ~jnp.eye(n, dtype=jnp.bool_)
        selected = pairwise[..., mask, :]
        return selected.reshape(*pairwise.shape[:-3], n, (n - 1) * feature_dim)


@register_env("jaxmarl_simple_spread")
def make_jaxmarl_simple_spread(**kwargs: Any) -> JaxMARLSimpleSpreadAdapter:
    return JaxMARLSimpleSpreadAdapter(**kwargs)
