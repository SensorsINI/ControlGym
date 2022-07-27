from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym import spaces
from gym.envs.classic_control import utils
from gym.envs.classic_control.acrobot import AcrobotEnv

from Environments import EnvironmentBatched, NumpyLibrary, cost_functions


class acrobot_batched(EnvironmentBatched, AcrobotEnv):
    num_actions = 1
    num_states = 6

    def __init__(self, batch_size=1, computation_lib=NumpyLibrary, render_mode="human", **kwargs):
        super().__init__(render_mode=render_mode)
        self.config = kwargs
        self.action_space = spaces.Box(low=-1., high=1., dtype=np.float32)

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.cost_functions = cost_functions(self)

    def step_tf(self, state: tf.Tensor, action: tf.Tensor):
        state, action = self._expand_arrays(state, action)

        # Perturb action if not in planning mode
        if self._batch_size == 1:
            action += self._generate_actuator_noise()

        th, thdot, sinth, costh = self.lib.unstack(state, 4, 1)  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        action = self.lib.clip(
            action,
            self.lib.to_tensor(-self.max_torque, self.lib.float32),
            self.lib.to_tensor(self.max_torque, self.lib.float32),
        )[:, 0]

        newthdot = (
            thdot
            + (3 * g / (2 * l) * self.lib.sin(th) + 3.0 / (m * l**2) * action) * dt
        )
        newthdot = self.lib.clip(
            newthdot,
            self.lib.to_tensor(-self.max_speed, self.lib.float32),
            self.lib.to_tensor(self.max_speed, self.lib.float32),
        )
        newth = th + newthdot * dt

        state = self.lib.stack([newth, newthdot, self.lib.sin(newth), self.lib.cos(newth)], 1)
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

        cos_th1, sin_th1, cos_th2, sin_th2, th1_vel, th2_vel = self.lib.unstack(self.state, 6, 1)  # th := theta

        assert self.state is not None, "Call reset before using AcrobotEnv object."
        torque = action

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = self.lib.concat([self.state, torque], 1)

        ns = self.rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = self.wrap(ns[0], -self.lib.pi, self.lib.pi)
        ns[1] = self.wrap(ns[1], -self.lib.pi, self.lib.pi)
        ns[2] = self.bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = self.bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        done = self.is_done(self.state)
        reward = self.get_reward(self.state, action)

        self.state = self.lib.squeeze(self.state)

        if self._batch_size == 1:
            return (
                self.lib.to_numpy(self.lib.squeeze(self.state)),
                float(reward),
                done,
                {},
            )

        self.renderer.render_step()
        return self.state, reward, False, {}

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
            low, high = utils.maybe_parse_reset_bounds(
                options, -0.1, 0.1
            )
            if self._batch_size == 1:
                self.state = self.lib.uniform(self.rng, (4,), low, high, self.lib.float32)
            else:
                self.state = self.lib.uniform(self.rng, (self._batch_size, 4), low, high, self.lib.float32)
        else:
            if self.lib.ndim(state) < 2:
                state = self.lib.unsqueeze(
                    self.lib.to_tensor(state, self.lib.float32), 0
                )
            if self.lib.shape(state)[0] == 1:
                self.state = self.lib.tile(state, (self._batch_size, 1))
            else:
                self.state = state

        self.renderer.reset()
        self.renderer.render_step()

        # Augment state with sin/cos
        self.state = self.lib.concat([
            self.lib.cos(self.lib.unsqueeze(self.state[..., 0], 1)),
            self.lib.sin(self.lib.unsqueeze(self.state[..., 0], 1)),
            self.lib.cos(self.lib.unsqueeze(self.state[..., 1], 1)),
            self.lib.sin(self.lib.unsqueeze(self.state[..., 1], 1)),
            self.lib.unsqueeze(self.state[..., 2], 1),
            self.lib.unsqueeze(self.state[..., 3], 1),
        ], 1)
        
        return self._get_reset_return_val()

    def is_done(self, state):
        assert state is not None, "Call reset before using AcrobotEnv object."
        return -self.lib.cos(state[0]) - self.lib.cos(state[1] + state[0]) > 1.0

    def get_reward(self, state, action):
        return (-1.0) * self.lib.cast(self.is_done(state), self.lib.float32)
    
    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return self.lib.to_tensor(
            [self.lib.cos(s[0]), self.lib.sin(s[0]), self.lib.cos(s[1]), self.lib.sin(s[1]), s[2], s[3]], dtype=np.float32
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
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
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
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * self.lib.sin(theta2) - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, self.lib.zeros((self._batch_size, 1))
    

    def wrap(self, x, m, M):
        """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
        truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
        For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

        Args:
            x: a scalar
            m: minimum possible value in range
            M: maximum possible value in range

        Returns:
            x: a scalar, wrapped
        """
        diff = M - m
        while self.lib.any(x > M):
            x[x > M] = x[x > M] - diff
        while self.lib.any(x < m):
            x[x < m] = x[x < m] + diff
        return x


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
        Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.

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

        try:
            Ny = len(y0)
        except TypeError:
            yout = np.zeros((len(t),), np.float_)
        else:
            yout = np.zeros((len(t), Ny), np.float_)

        yout[0] = y0

        for i in np.arange(len(t) - 1):

            this = t[i]
            dt = t[i + 1] - this
            dt2 = dt / 2.0
            y0 = yout[i]

            k1 = np.asarray(derivs(y0))
            k2 = np.asarray(derivs(y0 + dt2 * k1))
            k3 = np.asarray(derivs(y0 + dt2 * k2))
            k4 = np.asarray(derivs(y0 + dt * k3))
            yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        # We only care about the final timestep and we cleave off action value which will be zero
        return yout[-1][:4]
