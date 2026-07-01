# aorpo/envs/mpe_env_wrapper.py
"""
MPEEnv wrapper for PettingZoo's simple_spread_v3 environment.
This corresponds exactly to the Cooperative Navigation task used in the AORPO paper.
"""

from pettingzoo.mpe import simple_spread_v3
import numpy as np


class MPEEnv:
    def __init__(self, num_agents: int = 3, max_cycles: int = 25, continuous_actions: bool = True, seed: int = 0):
        """
        Initialize the Multi-Agent Particle Environment (Cooperative Navigation).
        Equivalent to the 'simple_spread' task in the AORPO paper.
        """
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.seed = seed

        # Create the PettingZoo environment
        self.env = simple_spread_v3.env(
            N=self.num_agents,
            max_cycles=self.max_cycles,
            continuous_actions=self.continuous_actions,
            local_ratio=0.5
        )
        self.env.reset(seed=self.seed)

        # Agent list (agent_0, agent_1, ...)
        self.agents = self.env.possible_agents
        self.ego_index = 0  # we treat agent_0 as ego by default

        # Observation & action space info
        obs_example = self.env.observation_space(self.agents[0]).sample()
        act_example = self.env.action_space(self.agents[0]).sample()
        self.obs_shape = obs_example.shape
        self.act_dim = act_example.shape[0] if len(act_example.shape) > 0 else 1

    def reset(self, seed: int | None = None):
        """
        Reset the environment and return a list of observations (for all agents).
        """
        self.env.reset(seed=seed)
        obs_list = []
        for agent in self.agents:
            obs, _, _, _, _ = self.env.last()
            obs_list.append(obs)
            self.env.step(self.env.action_space(agent).sample())  # step once to align state
        self.env.reset(seed=seed)
        obs_list = [self.env.observe(agent) for agent in self.agents]
        info = {"agents": self.agents}
        return obs_list, info

    def step(self, actions: list[np.ndarray]):
        """
        Take a step in the multi-agents environment.
        `actions`: list of np.ndarray actions for each agents (aligned with self.agents order)
        Returns:
            obs_list: list of observations per agents
            rew_list: list of rewards per agents
            terminated: bool
            truncated: bool
            info: dict
        """
        assert len(actions) == len(self.agents), "Number of actions must equal number of agents."

        rew_dict = {agent: 0.0 for agent in self.agents}

        # run through all agents for one full environment step
        for agent in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()
            if termination or truncation:
                self.env.step(None)
                continue
            idx = self.agents.index(agent)
            act = actions[idx]
            self.env.step(act)
            rew_dict[agent] = reward

        obs_list = [self.env.observe(agent) for agent in self.agents]
        rew_list = [rew_dict[a] for a in self.agents]
        terminated = any(self.env.terminations.values())
        truncated = any(self.env.truncations.values())
        info = {}
        return obs_list, rew_list, terminated, truncated, info

    def close(self):
        self.env.close()


# --- Debug / Standalone test ---
if __name__ == "__main__":
    env = MPEEnv(num_agents=3, max_cycles=25)
    obs, info = env.reset()
    print(f"Env initialized: {len(obs)} agents, obs shape: {obs[0].shape}, act dim: {env.act_dim}")

    for t in range(5):
        actions = [np.random.uniform(0, 1, env.act_dim) for _ in range(env.num_agents)]
        obs, rew, done, trunc, info = env.step(actions)
        print(f"Step {t}: rewards = {rew}")
        if done or trunc:
            break
    env.close()
