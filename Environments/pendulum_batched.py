from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gym.utils import seeding

from Environments import EnvironmentBatched

_PI = tf.constant(np.pi, dtype=tf.float32)


class PendulumEnv_Batched(EnvironmentBatched, PendulumEnv):
    def __init__(self, g=10, batch_size=1):
        super().__init__(g)
        self._batch_size = batch_size

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _angle_normalize(self, x):
        return ((x + _PI) % (2 * _PI)) - _PI

    def step(
        self, action: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)

        th, thdot = tf.unstack(self.state, axis=1)  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        action = tf.clip_by_value(action, -self.max_torque, self.max_torque)[:, 0]
        self.last_action = action  # for rendering
        costs = (
            self._angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (action**2)
        )

        newthdot = (
            thdot + (3 * g / (2 * l) * tf.sin(th) + 3.0 / (m * l**2) * action) * dt
        )
        newthdot = tf.clip_by_value(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = tf.squeeze(tf.stack([newth, newthdot], axis=1))

        if self._batch_size == 1:
            return tf.squeeze(self.state).numpy(), -float(costs), False, {}

        return self.state, -costs, False, {}

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        if state is None:
            if self._batch_size == 1:
                high = np.array([np.pi, 1])
            else:
                high = np.tile(np.array([np.pi, 1]), (self._batch_size, 1))
            self.state = tf.convert_to_tensor(
                self.np_random.uniform(low=-high, high=high), dtype=tf.float32
            )
        else:
            if state.ndim < 2:
                state = tf.expand_dims(state, axis=0)
            self.state = tf.tile(state, (self._batch_size, 1))

        self.last_u = None

        return self._get_reset_return_val()

    # def render(self, mode="human"):
    #     if self._batch_size == 1:
    #         return super().render(mode)
    #     else:
    #         raise NotImplementedError("Rendering not implemented for batched mode")
