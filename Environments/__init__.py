from typing import Union

import numpy as np
import tensorflow as tf
import torch
from gym.envs.registration import register
from numpy.random import Generator
from Utilities.utils import get_logger

log = get_logger(__name__)

ENV_REGISTRY = {
    "CustomEnvironments/CartPoleContinuous-v0": "Environments.continuous_cartpole_batched:continuous_cartpole_batched",
    "CustomEnvironments/MountainCarContinuous-v0": "Environments.continuous_mountaincar_batched:continuous_mountaincar_batched",
    "CustomEnvironments/Pendulum-v0": "Environments.pendulum_batched:pendulum_batched",
    "CustomEnvironments/AcrobotBatched-v0": "Environments.acrobot_batched:acrobot_batched",
    "CustomEnvironments/DubinsCar-v0": "Environments.dubins_car_batched:dubins_car_batched",
    "CustomEnvironments/CartPoleSimulator-v0": "Environments.cartpole_simulator_batched:cartpole_simulator_batched",
    "CustomEnvironments/HalfCheetahBatched-v0": "Environments.half_cheetah_batched:half_cheetah_batched",
}


def register_envs():
    for identifier, entry_point in ENV_REGISTRY.items():
        register(
            id=identifier,
            entry_point=entry_point,
            max_episode_steps=None,
        )


TensorType = Union[np.ndarray, tf.Tensor, torch.Tensor]
RandomGeneratorType = Union[Generator, tf.random.Generator, torch.Generator]
