from gym.envs.registration import register

register(
    id="CustomEnvironments/CartPoleContinuous-v1",
    entry_point="Environments.continuous_cartpole_batched:Continuous_CartPoleEnv_Batched",
    max_episode_steps=300,
)
