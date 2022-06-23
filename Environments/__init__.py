from gym.envs.registration import register

ENV_REGISTRY = {
    "CustomEnvironments/CartPoleContinuous": "Environments.continuous_cartpole_batched:Continuous_CartPoleEnv_Batched",
    "CustomEnvironments/MountainCarContinuous": "Environments.continuous_mountaincar_batched:Continuous_MountainCarEnv_Batched",
    "CustomEnvironments/Pendulum": "Environments.pendulum_batched:PendulumEnv_Batched",
}

for identifier, entry_point in ENV_REGISTRY.items():
    register(
        id=identifier,
        entry_point=entry_point,
        max_episode_steps=None,
    )
