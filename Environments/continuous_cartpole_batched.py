from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv

from Control_Toolkit.others.environment import EnvironmentBatched, NumpyLibrary
from Utilities.utils import CurrentRunMemory


class continuous_cartpole_batched(EnvironmentBatched, CartPoleEnv):
    num_actions = 1
    num_states = 4

    def __init__(self, batch_size=1, computation_lib=NumpyLibrary, render_mode="human", **kwargs):
        super().__init__(render_mode=render_mode)

        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode},
        }
        CurrentRunMemory.controller_specific_params = self.config

        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )
        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.cost_functions = self.cost_functions_wrapper(self)
    
    def step_tf(self, state: tf.Tensor, action: tf.Tensor):
        state, action = self._expand_arrays(state, action)

        if self._batch_size == 1:
            action += self._generate_actuator_noise()

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
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        state = self.lib.stack([x, x_dot, theta, theta_dot], 1)
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

        err_msg = f"{action!r} ({type(action)}) invalid"
        # assert np.all(
        #     [self.action_space.contains(a) for a in self.lib.to_numpy(action)]
        # ), err_msg
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.lib.unstack(self.state, 4, 1)
        force = self.lib.clip(
            action[:, 0],
            self.lib.to_tensor(self.action_space.low, self.lib.float32),
            self.lib.to_tensor(self.action_space.high, self.lib.float32),
        )
        costheta = self.lib.cos(theta)
        sintheta = self.lib.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = self.lib.stack([x, x_dot, theta, theta_dot], 1)

        done = self.is_done(self.state)
        reward = self.get_reward(self.state, action)

        self.state = self.lib.squeeze(self.state)

        if self._batch_size == 1:
            self.renderer.render_step()
            return self.lib.to_numpy(self.state), float(reward), bool(done), {}

        return self.state, reward, done, {}

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[Union[np.ndarray, tf.Tensor, torch.Tensor], Optional[dict]]:
        if seed is not None:
            self._set_up_rng(seed)

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

    def is_done(self, state):
        x, x_dot, theta, theta_dot = self.lib.unstack(state, 4, 1)
        return (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

    def get_reward(self, state, action):
        state, action = self._expand_arrays(state, action)
        x, x_dot, theta, theta_dot = self.lib.unstack(state, 4, 1)
        reward = -(100*(theta**2) + theta_dot**2 + (x**2) + x_dot**2)
        return reward
