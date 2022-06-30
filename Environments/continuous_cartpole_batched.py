from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from gym import logger, spaces
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.utils import seeding

from Environments import EnvironmentBatched


class Continuous_CartPoleEnv_Batched(EnvironmentBatched, CartPoleEnv):
    def __init__(self, batch_size=1):
        super().__init__()
        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )
        self._batch_size = batch_size

    def step(
        self, action: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert np.all([self.action_space.contains(a) for a in action.numpy()]), err_msg
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = tf.unstack(self.state, axis=1)
        force = tf.clip_by_value(
            action[:, 0], self.action_space.low, self.action_space.high
        )
        costheta = tf.cos(theta)
        sintheta = tf.sin(theta)

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

        self.state = tf.squeeze(tf.stack([x, x_dot, theta, theta_dot], axis=1))

        done = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        reward = -(tf.abs(theta) + tf.abs(theta_dot) + tf.abs(x) + tf.abs(x_dot))
        if self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward += tf.cast(done, dtype=tf.float32)
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1

        if self._batch_size == 1:
            return np.array(self.state, dtype=np.float32), float(reward), bool(done), {}

        return self.state, reward, done, {}

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        if state is None:
            if self._batch_size == 1:
                self.state = tf.convert_to_tensor(
                    [self.np_random.uniform(low=-0.05, high=0.05, size=(4,))],
                    dtype=tf.float32,
                )
            else:
                self.state = tf.convert_to_tensor(
                    [
                        self.np_random.uniform(
                            low=-0.05, high=0.05, size=(self._batch_size, 4)
                        )
                    ],
                    dtype=tf.float32,
                )
        else:
            if state.ndim < 2:
                state = tf.expand_dims(state, axis=0)
            self.state = tf.tile(state, [self._batch_size, 1])

        self.steps_beyond_done = None

        return self._get_reset_return_val()
