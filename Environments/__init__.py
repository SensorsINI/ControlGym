from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
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


class EnvironmentBatched:
    def step(
        self, action: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        return NotImplementedError()

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        return NotImplementedError()
