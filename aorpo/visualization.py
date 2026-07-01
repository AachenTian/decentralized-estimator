# import jax
# from jaxmarl import make
#
# # 初始化环境
# env = make("MPE_simple_spread_v3")
# key = jax.random.PRNGKey(0)
# obs, state = env.reset(key)
#
# print("Agents:", env.agents)
# print("Obs:", obs)
# print("Obs.shape:", obs["agent_0"].shape)
# print("State:",state)
# print("State:",state.p_pos.shape)
#
# for agent in env.agents:
#     obs_space = env.observation_space(agent)
#     act_space = env.action_space(agent)
#     print(f"\nAgent: {agent}")
#     print(f"  Observation shape: {obs_space.shape}")
#     print(f"  Action shape: {act_space.shape}")
#     print(f"  Observation example: {obs[agent].shape}")
