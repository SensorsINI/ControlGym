from typing import Optional
import numpy as np

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
        position = self.state[:, 0]
        velocity = self.state[:, 1]
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

        self.state = np.c_[position, velocity]
        return self.state, reward, done, {}

    def reset(self, state: np.ndarray, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.state = np.tile(state, (self._batch_size, 1))
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
    
    def render(self, mode="human"):
        if self._batch_size == 1:
            return super().render(mode=mode)
        else:
            raise NotImplementedError("Rendering not implemented for batched mode")