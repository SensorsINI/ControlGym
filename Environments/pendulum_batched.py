from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv

from Environments import EnvironmentBatched, NumpyLibrary, cost_functions


class pendulum_batched(EnvironmentBatched, PendulumEnv):
    num_actions = 1
    num_states = 2

    def __init__(self, g=10, batch_size=1, computation_lib=NumpyLibrary, render_mode="human", **kwargs):
        super().__init__(render_mode=render_mode, g=g)
        self.config = kwargs
        high = np.array([np.pi, self.max_speed, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.cost_functions = cost_functions(self)

    def _angle_normalize(self, x):
        _pi = self.lib.to_tensor(np.pi, self.lib.float32)
        return ((x + _pi) % (2 * _pi)) - _pi

    def step_tf(self, state: tf.Tensor, action: tf.Tensor):
        state, action = self._expand_arrays(state, action)

        # Perturb action if not in planning mode
        if self._batch_size == 1:
            action += self._generate_actuator_noise()

        th, thdot, sinth, costh = self.lib.unstack(state, 4, 1)  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        action = self.lib.clip(
            action,
            self.lib.to_tensor(-self.max_torque, self.lib.float32),
            self.lib.to_tensor(self.max_torque, self.lib.float32),
        )[:, 0]

        newthdot = (
            thdot
            + (3 * g / (2 * l) * self.lib.sin(th) + 3.0 / (m * l**2) * action) * dt
        )
        newthdot = self.lib.clip(
            newthdot,
            self.lib.to_tensor(-self.max_speed, self.lib.float32),
            self.lib.to_tensor(self.max_speed, self.lib.float32),
        )
        newth = th + newthdot * dt

        state = self.lib.stack([newth, newthdot, self.lib.sin(newth), self.lib.cos(newth)], 1)
        state = self.lib.squeeze(state)

        return state

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

        th, thdot, sinth, costh = self.lib.unstack(self.state, 4, 1)  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        action = self.lib.clip(
            action,
            self.lib.to_tensor(np.array(-self.max_torque), self.lib.float32),
            self.lib.to_tensor(np.array(self.max_torque), self.lib.float32),
        )[:, 0]
        self.last_action = action  # for rendering

        newthdot = (
            thdot
            + (3 * g / (2 * l) * self.lib.sin(th) + 3.0 / (m * l**2) * action) * dt
        )
        newthdot = self.lib.clip(
            newthdot,
            self.lib.to_tensor(np.array(-self.max_speed), self.lib.float32),
            self.lib.to_tensor(np.array(self.max_speed), self.lib.float32),
        )
        newth = th + newthdot * dt

        self.state = self.lib.stack([newth, newthdot, self.lib.sin(newth), self.lib.cos(newth)], 1)

        done = self.is_done(self.state)
        reward = self.get_reward(self.state, action)

        self.state = self.lib.squeeze(self.state)

        if self._batch_size == 1:
            self.renderer.render_step()
            return (
                self.lib.to_numpy(self.lib.squeeze(self.state)),
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
                high = np.array([[np.pi, 1]])
            else:
                high = np.tile(np.array([np.pi, 1]), (self._batch_size, 1))
            self.state = self.lib.uniform(self.rng, high.shape, -high, high, self.lib.float32)
        else:
            if self.lib.ndim(state) < 2:
                state = self.lib.unsqueeze(
                    self.lib.to_tensor(state, self.lib.float32), 0
                )
            if self.lib.shape(state)[0] == 1:
                self.state = self.lib.tile(state, (self._batch_size, 1))
            else:
                self.state = state

        # Augment state with sin/cos
        self.state = self.lib.concat([self.state, self.lib.sin(self.lib.unsqueeze(self.state[..., 0], 1)), self.lib.cos(self.lib.unsqueeze(self.state[..., 0], 1))], 1)
        
        self.last_u = None
        return self._get_reset_return_val()

    def is_done(self, state):
        return False

    def get_reward(self, state, action):
        state, action = self._expand_arrays(state, action)
        th, thdot, sinth, costh = self.lib.unstack(state, 4, 1)
        costs = (
            self._angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (action[:, 0]**2)
        )
        return -costs

    # def render(self, mode="human"):
    #     if self._batch_size == 1:
    #         return super().render(mode)
    #     else:
    #         raise NotImplementedError("Rendering not implemented for batched mode")
