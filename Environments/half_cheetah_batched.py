from typing import Optional, Tuple, Union

import mujoco
import numpy as np
import tensorflow as tf
from Control_Toolkit.others.environment import EnvironmentBatched, NumpyLibrary
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from tf_agents.environments import BatchedPyEnvironment, suite_gym


class half_cheetah_batched(EnvironmentBatched, HalfCheetahEnv):
    num_actions = 6
    num_states = 17

    def __init__(
        self,
        actuator_noise: "list[float]",
        seed: int,
        forward_reward_weight: float,
        ctrl_cost_weight: float,
        reset_noise_scale: float,
        exclude_current_positions_from_observation: bool,
        batch_size=1,
        computation_lib=NumpyLibrary,
        render_mode="human",
    ) -> None:
        self._envs = BatchedPyEnvironment(
            [
                suite_gym.load(
                    "HalfCheetah-v3",
                    gym_kwargs=dict(
                        forward_reward_weight=forward_reward_weight,
                        ctrl_cost_weight=ctrl_cost_weight,
                        reset_noise_scale=reset_noise_scale,
                        exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                    ),
                    render_kwargs={"mode": render_mode},
                )
                for _ in range(batch_size)
            ],
            multithreading=True,
        )

        self._batch_size = batch_size
        self._actuator_noise = np.array(actuator_noise, dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(seed)
        self.cost_functions = self.cost_functions_wrapper(self)

    def step(
        self, action: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        return self._envs.step(action)

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        if seed is not None:
            self._set_up_rng(seed)
        
        step_type, reward, discount, obs  = self._envs.reset()
        
        if state is not None:
            self._envs.set_state(state)


        if return_info:
            return obs, {}
        return obs

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
