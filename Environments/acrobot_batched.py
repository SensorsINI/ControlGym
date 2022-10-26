from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym import spaces
from gym.envs.classic_control import utils
from gym.envs.classic_control.acrobot import AcrobotEnv

from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorType


class acrobot_batched(EnvironmentBatched, AcrobotEnv):
    num_actions = 1
    num_states = 4

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

    def step_dynamics(
        self,
        state: TensorType,
        action: TensorType,
        dt: float,
    ) -> TensorType:
        torque = action
        s_augmented = self.lib.concat([state, torque], 1)

        th1_new, th2_new, th1_vel_new, th2_vel_new = self.lib.unstack(
            self.rk4(self._dsdt, s_augmented, [0, dt]), 4, 1
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
        state = self.lib.squeeze(state)

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

        torque = action
        s_augmented = self.lib.concat([self.state, torque], 1)

        th1_new, th2_new, th1_vel_new, th2_vel_new = self.lib.unstack(
            self.rk4(self._dsdt, s_augmented, [0, self.dt]), 4, 1
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
        self.state = self.lib.stack([th1_new, th2_new, th1_vel_new, th2_vel_new], 1)

        terminated = self.is_done(self.lib, self.state)
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

    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        theta1, theta2, dtheta1, dtheta2, a = self.lib.unstack(s_augmented, 5, 1)
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * self.lib.cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * self.lib.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * self.lib.cos(theta1 + theta2 - self.lib.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * self.lib.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * self.lib.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * self.lib.cos(theta1 - self.lib.pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a
                + d2 / d1 * phi1
                - m2 * l1 * lc2 * dtheta1**2 * self.lib.sin(theta2)
                - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return self.lib.stack(
            [
                dtheta1,
                dtheta2,
                ddtheta1,
                ddtheta2,
                self.lib.zeros_like(dtheta1),
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

    def rk4(self, derivs, y0, t):
        """
        Integrate 1-D or N-D system of ODEs batch-wise using 4-th order Runge-Kutta.

        Example for 2D system:

            >>> def derivs(x):
            ...     d1 =  x[0] + 2*x[1]
            ...     d2 =  -3*x[0] + 4*x[1]
            ...     return d1, d2

            >>> dt = 0.0005
            >>> t = np.arange(0.0, 2.0, dt)
            >>> y0 = (1,2)
            >>> yout = rk4(derivs, y0, t)

        Args:
            derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
            y0: initial state vector
            t: sample times

        Returns:
            yout: Runge-Kutta approximation of the ODE
        """
        yout = self.lib.unsqueeze(y0, 1)  # batch_size x rk-steps x states

        for i in np.arange(len(t) - 1):

            this = t[i]
            dt = t[i + 1] - this
            dt2 = dt / 2.0
            y0 = yout[:, i, :]

            k1 = derivs(y0)
            k2 = derivs(y0 + dt2 * k1)
            k3 = derivs(y0 + dt2 * k2)
            k4 = derivs(y0 + dt * k3)
            y_next = self.lib.unsqueeze(y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 1)
            yout = self.lib.concat([yout, y_next], 1)
        # We only care about the final timestep and we cleave off action value which will be zero
        return yout[:, -1, :4]
