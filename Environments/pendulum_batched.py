from typing import Optional
import numpy as np

from gym import spaces
from gym.utils import seeding

from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize


class PendulumEnv_Batched(PendulumEnv):
    def __init__(self, g=10, batch_size=1):
        super().__init__(g)
        self._batch_size = batch_size

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        if u.ndim < 2:
            u = np.expand_dims(u, axis=0)

        th, thdot = list(self.state.T)  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[:, 0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.squeeze(np.c_[newth, newthdot])

        if self._batch_size == 1:
            costs = float(costs)

        return self.state, -costs, False, {}

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        if state is None:
            if self._batch_size == 1:
                high = np.array([np.pi, 1])
                self.state = self.np_random.uniform(low=-high, high=high)
            else:
                high = np.tile(np.array([np.pi, 1]), (self._batch_size, 1))
                self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = np.tile(state, (self._batch_size, 1))

        self.last_u = None

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    # def render(self, mode="human"):
    #     if self._batch_size == 1:
    #         return super().render(mode)
    #     else:
    #         raise NotImplementedError("Rendering not implemented for batched mode")
