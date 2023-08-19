from typing import Optional, Tuple, Union
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import torch
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.envs.classic_control.acrobot import AcrobotEnv

from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorType
from Acrobot_helper_functions import _dsdt, rk4

class acrobot_batched(EnvironmentBatched, AcrobotEnv):
    num_actions = 1
    num_states = 4
    book_or_nips = "nips"

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

        high = np.array(
            [np.pi, np.pi, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, dtype=np.float32)

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])

        parameters = SimpleNamespace()
        parameters.m1 = self.LINK_MASS_1
        parameters.m2 = self.LINK_MASS_2
        parameters.l1 = self.LINK_LENGTH_1
        parameters.lc1 = self.LINK_COM_POS_1
        parameters.lc2 = self.LINK_COM_POS_2
        parameters.I1 = self.LINK_MOI
        parameters.I2 = self.LINK_MOI
        parameters.g = 9.8

        self._dsdt = _dsdt(parameters, self.lib, source='gym', book_or_nips=self.book_or_nips)


    def step_dynamics(
        self,
        state: TensorType,
        action: TensorType,
        dt: float,
    ) -> TensorType:
        torque = action
        s_augmented = self.lib.concat([state, torque], 1)

        th1_new, th2_new, th1_vel_new, th2_vel_new = self.lib.unstack(
            rk4(self._dsdt, s_augmented, [0, dt], self.lib), 4, 1
        )

        # Wrap angles
        th1_new = (
            self.lib.floormod(th1_new + self.lib.pi, 2 * self.lib.pi) - self.lib.pi
        )
        th2_new = (
            self.lib.floormod(th2_new + self.lib.pi, 2 * self.lib.pi) - self.lib.pi
        )

        # Clip angular velocities
        th1_vel_new = self.lib.clip(th1_vel_new, -self.MAX_VEL_1, self.MAX_VEL_1)
        th2_vel_new = self.lib.clip(th2_vel_new, -self.MAX_VEL_2, self.MAX_VEL_2)
        state = self.lib.stack([th1_new, th2_new, th1_vel_new, th2_vel_new], 1)

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

        self.state = self.step_dynamics(self.state, action, self.dt)

        terminated = bool(self.is_done(self.lib, self.state))
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
            low, high = utils.maybe_parse_reset_bounds(options, -0.1, 0.1)
            ns = self.lib.uniform(
                self.rng, (self._batch_size, 4), low, high, self.lib.float32
            )
        else:
            if self.lib.ndim(state) < 2:
                state = self.lib.unsqueeze(
                    self.lib.to_tensor(state, self.lib.float32), 0
                )
            if self.lib.shape(state)[0] == 1:
                ns = self.lib.tile(state, (self._batch_size, 1))
            else:
                ns = state

        # ns = self.lib.uniform(self.rng, (4,), -0.0, 0.0, self.lib.float32) #* self.lib.to_tensor([np.pi, np.pi, 4.0 * np.pi, 9.0 * np.pi], self.lib.float32)
        # ns = ns*self.lib.to_tensor([0.0, 1.0, 1.0, 1.0], self.lib.float32) + self.lib.to_tensor([np.pi/2.0, np.pi/2.0, 0.0, 0.0], self.lib.float32)
        # ns = ns * self.lib.to_tensor([0.0, 1.0, 1.0, 1.0], self.lib.float32) + self.lib.to_tensor([np.pi / 2.0, np.pi / 2.0, 0.0, 0.0], self.lib.float32)
        ns = self.lib.to_tensor([0.0, 0.0, 0.0, 0.0], self.lib.float32)
        # Augment state with sin/cos
        self.state = ns

        return self._get_reset_return_val()

    @staticmethod
    def is_done(lib: "type[ComputationLibrary]", state: TensorType):
        return False

    def _convert_to_state(self, state):
        if self.lib.shape(state)[-1] == self.num_states:
            return state
        return self.lib.concat(
            [
                self.lib.cos(self.lib.unsqueeze(state[..., 0], 1)),
                self.lib.sin(self.lib.unsqueeze(state[..., 0], 1)),
                self.lib.cos(self.lib.unsqueeze(state[..., 1], 1)),
                self.lib.sin(self.lib.unsqueeze(state[..., 1], 1)),
                self.lib.unsqueeze(state[..., 2], 1),
                self.lib.unsqueeze(state[..., 3], 1),
            ],
            1,
        )


    def bound(self, x, m, M=None):
        """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
        have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

        Args:
            x: scalar
            m: The lower bound
            M: The upper bound

        Returns:
            x: scalar, bound between min (m) and Max (M)
        """
        if M is None:
            M = m[1]
            m = m[0]
        # bound x between min (m) and Max (M)
        return self.lib.min(self.lib.max(x, m), M)
