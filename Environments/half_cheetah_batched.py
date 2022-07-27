from typing import Optional, Tuple, Union
from Environments import EnvironmentBatched, NumpyLibrary, cost_functions
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gym.utils.ezpickle import EzPickle

import numpy as np
import tensorflow as tf
import mujoco


class half_cheetah_batched(EnvironmentBatched, HalfCheetahEnv, EzPickle):
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
        super().__init__(
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
        )
        self._batch_size = batch_size
        self._actuator_noise = np.array(actuator_noise, dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(seed)
        self.cost_functions = cost_functions(self)

    def step(
        self, action: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        self.renderer.render_step()
        return observation, reward, terminated, False, info

    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if self.lib.to_numpy(self.lib.shape(ctrl)) != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def control_cost(self, action):
        return self._ctrl_cost_weight * self.lib.sum(action**2)

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = self.lib.squeeze(self.lib.concat([position, velocity], 0))
        return observation

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        if seed is not None:
            self._set_up_rng(seed)
        
        self._reset_simulation()

        state = self.reset_model()
        self.renderer.reset()
        self.renderer.render_step()
        # if not return_info:
        #     return ob
        # else:
        #     return ob, {}

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
        position, velocity = self.lib.unstack(state, 2, 1)
        return (position >= self.goal_position) & (velocity >= self.goal_velocity)

    def get_reward(self, state, action):
        state, action = self._expand_arrays(state, action)
        position, velocity = self.lib.unstack(state, 2, 1)
        reward = self.lib.sin(3 * position)
        # This part is not differentiable:
        reward += 100.0 * self.lib.cast(self.is_done(state), self.lib.float32)
        reward -= (action[:, 0] ** 2) * 0.1
        return reward
