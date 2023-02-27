from typing import Union

import numpy as np
import tensorflow as tf
import torch
from gymnasium.envs.registration import register
from numpy.random import Generator
from Utilities.utils import get_logger

log = get_logger(__name__)

ENV_REGISTRY = {
    "MountainCarContinuous-v0": "Environments.continuous_mountaincar_batched:continuous_mountaincar_batched",
    "Acrobot-v0": "Environments.acrobot_batched:acrobot_batched",
    "DubinsCar-v0": "Environments.dubins_car_batched:dubins_car_batched",
    "ObstacleAvoidance-v0": "Environments.obstacle_avoidance_batched:obstacle_avoidance_batched",
    "LunarLander-v2": "Environments.lunar_lander_batched:lunar_lander_batched",
    "CartPoleSimulator-v0": "Environments.cartpole_simulator_batched:cartpole_simulator_batched",
    "CartPoleContinuous-v0": "Environments.continuous_cartpole_batched:continuous_cartpole_batched",
    "Pendulum-v0": "Environments.pendulum_batched:pendulum_batched",
    "HalfCheetahBatched-v0": "Environments.half_cheetah_batched:half_cheetah_batched",
    "BipedalWalkerBatched-v0": "Environments.bipedal_walker_batched:bipedal_walker_batched",
    "Armbot-v0": "Environments.armbot_batched:armbot_batched",
}


def register_envs():
    for identifier, entry_point in ENV_REGISTRY.items():
        register(
            id=identifier,
            entry_point=entry_point,
            max_episode_steps=None,
        )
