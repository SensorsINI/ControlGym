from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

from Control_Toolkit.others.environment import EnvironmentBatched, NumpyLibrary
from Utilities.utils import CurrentRunMemory


class continuous_mountaincar_batched(EnvironmentBatched, Continuous_MountainCarEnv):
    """Accepts batches of data to environment

    :param Continuous_MountainCarEnv: _description_
    :type Continuous_MountainCarEnv: _type_
    """
    num_actions = 1
    num_states = 2

    def __init__(
        self, goal_velocity=0, batch_size=1, computation_lib=NumpyLibrary, render_mode="human", **kwargs
    ):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)

        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode, "goal_velocity": self.goal_velocity},
        }
        CurrentRunMemory.controller_specific_params = self.config

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.cost_functions = self.cost_functions_wrapper(self)
        
    def step_tf(self, state: tf.Tensor, action: tf.Tensor):
        state, action = self._expand_arrays(state, action)
        
        if self._batch_size == 1:
            action += self._generate_actuator_noise()
        
        position, velocity = self.lib.unstack(state, 2, 1)
        force = self.lib.clip(
            action[:, 0],
            self.lib.to_tensor(self.min_action, self.lib.float32),
            self.lib.to_tensor(self.max_action, self.lib.float32),
        )
        velocity_new = (
            velocity + force * self.power - 0.0025 * self.lib.cos(3 * position)
        )
        velocity = self.lib.clip(
            velocity_new,
            self.lib.to_tensor(-self.max_speed, self.lib.float32),
            self.lib.to_tensor(self.max_speed, self.lib.float32),
        )

        position_new = position + velocity
        position = self.lib.clip(
            position_new,
            self.lib.to_tensor(self.min_position, self.lib.float32),
            self.lib.to_tensor(self.max_position, self.lib.float32),
        )
        velocity_updated = velocity * self.lib.cast(
            ~((position == self.min_position) & (velocity < 0)), self.lib.float32
        )
        velocity = velocity_updated
        
        state = self.lib.stack([position, velocity], 1)
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
        # TODO: Set explicitly whether in planning or simulation mode, not infer from batch size
        if self._batch_size == 1:
            action += self._generate_actuator_noise()

        position, velocity = self.lib.unstack(self.state, 2, 1)
        force = self.lib.clip(
            action[:, 0],
            self.lib.to_tensor(np.array(self.min_action), self.lib.float32),
            self.lib.to_tensor(np.array(self.max_action), self.lib.float32),
        )

        velocity_new = (
            velocity + force * self.power - 0.0025 * self.lib.cos(3 * position)
        )
        velocity = self.lib.clip(
            velocity_new,
            self.lib.to_tensor(np.array(-self.max_speed), self.lib.float32),
            self.lib.to_tensor(np.array(self.max_speed), self.lib.float32),
        )

        position_new = position + velocity
        position = self.lib.clip(
            position_new,
            self.lib.to_tensor(np.array(self.min_position), self.lib.float32),
            self.lib.to_tensor(np.array(self.max_position), self.lib.float32),
        )
        velocity_updated = velocity * self.lib.cast(
            ~((position == self.min_position) & (velocity < 0)), self.lib.float32
        )
        velocity = velocity_updated

        self.state = self.lib.stack([position, velocity], 1)

        done = self.is_done(self.state)
        reward = self.get_reward(self.state, action)

        self.state = self.lib.squeeze(self.state)

        if self._batch_size == 1:
            self.renderer.render_step()
            return (
                self.lib.to_numpy(self.lib.squeeze(self.state)),
                float(reward),
                bool(done),
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
                self.state = self.lib.to_tensor(
                    [self.lib.uniform(self.rng, (), -0.6, -0.4, self.lib.float32), 0],
                    self.lib.float32,
                )
            else:
                self.state = self.lib.stack(
                    [
                        self.lib.uniform(
                            self.rng, (self._batch_size,), -0.6, -0.4, self.lib.float32
                        ),
                        self.lib.zeros((self._batch_size,)),
                    ],
                    1,
                )
        else:
            if self.lib.ndim(state) < 2:
                state = self.lib.unsqueeze(
                    self.lib.to_tensor(state, self.lib.float32), 0
                )
            if self.lib.shape(state)[0] == 1:
                self.state = self.lib.tile(state, (self._batch_size, 1))
            else:
                self.state = state

        return self._get_reset_return_val()

    def render(self, mode="human"):
        if self._batch_size == 1:
            return super().render()
        else:
            raise NotImplementedError("Rendering not implemented for batched mode")

    def is_done(self, state):
        position, velocity = self.lib.unstack(state, 2, -1)
        return (position >= self.goal_position) & (velocity >= self.goal_velocity)

    def get_reward(self, state, action):
        position, velocity = self.lib.unstack(state, 2, -1)
        reward = self.lib.sin(3 * position)
        # This part is not differentiable:
        reward += 100.0 * self.lib.cast(self.is_done(state), self.lib.float32)
        reward -= (action[:, 0] ** 2) * 0.1
        return reward
