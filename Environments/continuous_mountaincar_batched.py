from typing import Optional
import numpy as np

from gym.utils import seeding

from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv


class Continuous_MountainCarEnv_Batched(Continuous_MountainCarEnv):
    """Accepts batches of data to environment

    :param Continuous_MountainCarEnv: _description_
    :type Continuous_MountainCarEnv: _type_
    """

    def __init__(self, goal_velocity=0, batch_size=1):
        super().__init__(goal_velocity)
        self._batch_size = batch_size

    def step(self, action: np.ndarray):
        if action.ndim < 2:
            action = np.expand_dims(action, axis=0)

        position, velocity = list(self.state.T)
        force = np.clip(action[:, 0], self.min_action, self.max_action)

        velocity += force * self.power - 0.0025 * np.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)

        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        velocity[(position == self.min_position) & (velocity < 0)] = 0

        done = (position >= self.goal_position) & (velocity >= self.goal_velocity)

        reward = np.sin(3 * position)
        # reward = np.zeros_like(position)
        reward[done] = 100.0
        reward -= np.power(action[:, 0], 2) * 0.1

        self.state = np.squeeze(np.c_[position, velocity])

        if self._batch_size == 1:
            done = bool(done)
            reward = float(reward)

        return self.state, reward, done, {}

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
                self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
            else:
                self.state = np.c_[
                    self.np_random.uniform(
                        low=-0.6, high=-0.4, size=(self._batch_size,)
                    ),
                    np.zeros(
                        self._batch_size,
                    ),
                ]
                self.state = self.np_random.uniform(
                    low=-0.05, high=0.05, size=(self._batch_size, 4)
                )
        else:
            self.state = np.tile(state.ravel(), (self._batch_size, 1))

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"):
        if self._batch_size == 1:
            return super().render(mode)
        else:
            raise NotImplementedError("Rendering not implemented for batched mode")
