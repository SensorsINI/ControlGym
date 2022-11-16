from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorType


class continuous_cartpole_batched(EnvironmentBatched, CartPoleEnv):
    num_actions = 1
    num_states = 4
    
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    def __init__(
        self,
        batch_size=1,
        computation_lib=NumpyLibrary,
        render_mode="human",
        **kwargs,
    ):
        super().__init__(render_mode=render_mode)

        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode},
        }
        self.dt = kwargs["dt"]

        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )
        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])

    def step_dynamics(
        self,
        state: TensorType,
        action: TensorType,
        dt: float,
    ) -> TensorType:
        x, x_dot, theta, theta_dot = self.lib.unstack(state, 4, 1)
        force = self.lib.clip(
            action[:, 0],
            self.lib.to_tensor(self.action_space.low, self.lib.float32),
            self.lib.to_tensor(self.action_space.high, self.lib.float32),
        )
        costheta = self.lib.cos(theta)
        sintheta = self.lib.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + dt * x_dot
            x_dot = x_dot + dt * xacc
            theta = theta + dt * theta_dot
            theta_dot = theta_dot + dt * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + dt * xacc
            x = x + dt * x_dot
            theta_dot = theta_dot + dt * thetaacc
            theta = theta + dt * theta_dot

        state = self.lib.stack([x, x_dot, theta, theta_dot], 1)

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

        # Perturb action if not in planning mode
        assert self._batch_size == 1
        action = self._apply_actuator_noise(action)

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        self.state = self.step_dynamics(self.state, action, self.dt)

        terminated = bool(self.is_done(self.lib, self.state))
        truncated = False
        reward = 0.0

        self.state = self.lib.squeeze(self.state)
        
        return self.lib.to_numpy(self.state), float(reward), terminated, truncated, {}

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
                self.state = (
                    self.lib.uniform(self.rng, (4,), -0.05, 0.05, self.lib.float32),
                )
            else:
                self.state = self.lib.uniform(
                    self.rng, (self._batch_size, 4), -0.05, 0.05, self.lib.float32
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

        self.steps_beyond_done = None

        return self._get_reset_return_val()

    @staticmethod
    def is_done(lib: "type[ComputationLibrary]", state: TensorType):
        x, x_dot, theta, theta_dot = lib.unstack(state, 4, -1)
        return (
            (x < -continuous_cartpole_batched.x_threshold)
            | (x > continuous_cartpole_batched.x_threshold)
            | (theta < -continuous_cartpole_batched.theta_threshold_radians)
            | (theta > continuous_cartpole_batched.theta_threshold_radians)
        )
