from typing import Optional, Tuple, Union

import numpy as np
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorType


class continuous_mountaincar_batched(EnvironmentBatched, Continuous_MountainCarEnv):
    """Accepts batches of data to environment

    :param Continuous_MountainCarEnv: _description_
    :type Continuous_MountainCarEnv: _type_
    """

    num_actions = 1
    num_states = 2

    def __init__(
        self,
        goal_velocity=0,
        batch_size=1,
        computation_lib=NumpyLibrary,
        render_mode="human",
        **kwargs,
    ):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)

        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode, "goal_velocity": self.goal_velocity},
        }
        self.dt = kwargs["dt"]

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        
        self.environment_attributes = {
            "goal_position": self.goal_position,
            "goal_velocity": self.goal_velocity,
        }

    def step_dynamics(
        self,
        state: TensorType,
        action: TensorType,
        dt: float,
    ) -> TensorType:
        position, velocity = self.lib.unstack(state, 2, 1)
        force = self.lib.clip(
            action[:, 0],
            self.lib.to_tensor(self.min_action, self.lib.float32),
            self.lib.to_tensor(self.max_action, self.lib.float32),
        )
        velocity_new = velocity + dt * (
            force * self.power - 0.0025 * self.lib.cos(3 * position)
        )
        velocity = self.lib.clip(
            velocity_new,
            self.lib.to_tensor(-self.max_speed, self.lib.float32),
            self.lib.to_tensor(self.max_speed, self.lib.float32),
        )

        position_new = position + dt * velocity
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

        return state

    def step(
        self, action: TensorType
    ) -> Tuple[
        TensorType,
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)
        assert self._batch_size == 1
        action = self._apply_actuator_noise(action)

        state_updated: TensorType = self.step_dynamics(self.state, action, self.dt)
        self.state = self.lib.to_numpy(state_updated)

        terminated = bool(self.is_done(self.lib, self.state, self.goal_position, self.goal_velocity))
        truncated = False
        reward = 0.0

        self.state = self.lib.squeeze(self.state)

        return (
            self.lib.to_numpy(self.lib.squeeze(self.state)),
            float(reward),
            terminated,
            truncated,
            {},
        )

    def reset(
        self,
        seed: "Optional[int]" = None,
        options: "Optional[dict]" = None,
    ) -> "Tuple[np.ndarray, dict]":
        if seed is not None:
            self._set_up_rng(seed)
        state = options.get("state", None) if isinstance(options, dict) else None

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

    def render(self):
        if self._batch_size == 1:
            return super().render()
        else:
            raise NotImplementedError("Rendering not implemented for batched mode")
        
    @staticmethod
    def is_done(lib: "type[ComputationLibrary]", state: TensorType, goal_position: float, goal_velocity: float):
        position, velocity = lib.unstack(state, 2, -1)
        return (position >= goal_position) & (velocity >= goal_velocity)

