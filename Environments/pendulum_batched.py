from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize

from Environments import EnvironmentBatched


class PendulumEnv_Batched(EnvironmentBatched, PendulumEnv):
    def __init__(self, g=10, batch_size=1, computation_lib="numpy", **kwargs):
        super().__init__(g)
        self.config = kwargs
        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)
        self._set_up_rng(kwargs["seed"])

        self.set_computation_library(computation_lib)

    def _angle_normalize(self, x):
        _pi = self._lib["to_tensor"](np.array(np.pi), self._lib["float32"])
        return ((x + _pi) % (2 * _pi)) - _pi

    def step(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)

        # Perturb action if not in planning mode
        if self._batch_size == 1:
            action += self._generate_actuator_noise()

        th, thdot = self._lib["unstack"](self.state, 2, 1)  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        action = self._lib["clip"](
            action,
            self._lib["to_tensor"](np.array(-self.max_torque), self._lib["float32"]),
            self._lib["to_tensor"](np.array(self.max_torque), self._lib["float32"]),
        )[:, 0]
        self.last_action = action  # for rendering

        newthdot = (
            thdot
            + (3 * g / (2 * l) * self._lib["sin"](th) + 3.0 / (m * l**2) * action)
            * dt
        )
        newthdot = self._lib["clip"](
            newthdot,
            self._lib["to_tensor"](np.array(-self.max_speed), self._lib["float32"]),
            self._lib["to_tensor"](np.array(self.max_speed), self._lib["float32"]),
        )
        newth = th + newthdot * dt

        self.state = self._lib["stack"]([newth, newthdot], 1)

        done = self.is_done(self.state)
        reward = self.get_reward(self.state, action)

        self.state = self._lib["squeeze"](self.state)

        if self._batch_size == 1:
            return (
                self._lib["to_numpy"](self._lib["squeeze"](self.state)),
                float(reward),
                done,
                {},
            )

        return self.state, reward, done, {}

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        if seed is not None:
            self._set_up_rng(seed)

        if state is None:
            if self._batch_size == 1:
                high = np.array([np.pi, 1])
            else:
                high = np.tile(np.array([np.pi, 1]), (self._batch_size, 1))
            self.state = self._lib["to_tensor"](
                self.np_random.uniform(low=-high, high=high), self._lib["float32"]
            )
        else:
            if self._lib["ndim"](state) < 2:
                state = self._lib["unsqueeze"](
                    self._lib["to_tensor"](state, self._lib["float32"]), 0
                )
            self.state = self._lib["tile"](state, (self._batch_size, 1))

        self.last_u = None

        return self._get_reset_return_val()

    def is_done(self, state):
        return False

    def get_reward(self, state, action):
        th, thdot = self._lib["unstack"](self.state, 2, 1)
        costs = (
            self._angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (action**2)
        )
        return -costs

    # def render(self, mode="human"):
    #     if self._batch_size == 1:
    #         return super().render(mode)
    #     else:
    #         raise NotImplementedError("Rendering not implemented for batched mode")
