from typing import Optional
import numpy as np

from gym import spaces, logger
from gym.utils import seeding

from gym.envs.classic_control.cartpole import CartPoleEnv


class Continuous_CartPoleEnv_Batched(CartPoleEnv):
    def __init__(self, batch_size=1):
        super().__init__()
        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )

        self._batch_size = batch_size

    def step(self, action):
        if action.ndim < 2:
            action = np.expand_dims(action, axis=0)
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert np.all([self.action_space.contains(a) for a in action]), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = list(self.state.T)
        force = action[:, 0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

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

        self.state = np.squeeze(np.c_[x, x_dot, theta, theta_dot])

        done = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        reward = np.zeros((self._batch_size))
        reward[~done] = 1.0
        if self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward[done] = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward[done] = 0.0

        if self._batch_size == 1:
            done = bool(done)
            reward = float(reward)

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        if state is None:
            if self._batch_size == 1:
                self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
            else:
                self.state = self.np_random.uniform(
                    low=-0.05, high=0.05, size=(self._batch_size, 4)
                )
        else:
            self.state = np.tile(state.ravel(), (self._batch_size, 1))

        self.steps_beyond_done = None

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
