from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from gym.envs.classic_control.continuous_mountain_car import \
    Continuous_MountainCarEnv
from gym.utils import seeding

from Environments import EnvironmentBatched


class Continuous_MountainCarEnv_Batched(EnvironmentBatched, Continuous_MountainCarEnv):
    """Accepts batches of data to environment

    :param Continuous_MountainCarEnv: _description_
    :type Continuous_MountainCarEnv: _type_
    """

    def __init__(self, goal_velocity=0, batch_size=1, **kwargs):
        super().__init__(goal_velocity)
        self.config = kwargs
        self._batch_size = batch_size
        self._set_up_rng(**kwargs)

    def step(
        self, action: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)

        # Perturb action if not in planning mode
        if self._batch_size == 1:
            action += self._generate_actuator_noise()

        position, velocity = tf.unstack(self.state, axis=1)
        force = tf.clip_by_value(action[:, 0], self.min_action, self.max_action)

        velocity += force * self.power - 0.0025 * tf.cos(3 * position)
        velocity = tf.clip_by_value(velocity, -self.max_speed, self.max_speed)

        position += velocity
        position = tf.clip_by_value(position, self.min_position, self.max_position)
        velocity *= tf.cast(
            ~((position == self.min_position) & (velocity < 0)), dtype=tf.float32
        )

        done = (position >= self.goal_position) & (velocity >= self.goal_velocity)

        reward = tf.sin(3 * position)
        # reward = np.zeros_like(position)
        reward += 100.0 * tf.cast(done, dtype=tf.float32)
        reward -= tf.pow(action[:, 0], 2) * 0.1

        self.state = tf.squeeze(tf.stack([position, velocity], axis=1))

        if self._batch_size == 1:
            return tf.squeeze(self.state).numpy(), float(reward), bool(done), {}

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
                    [self.np_random.uniform(low=-0.6, high=-0.4), 0]
                )
            else:
                self.state = tf.stack(
                    [
                        tf.convert_to_tensor(
                            self.np_random.uniform(
                                low=-0.6, high=-0.4, size=(self._batch_size,)
                            )
                        ),
                        tf.zeros(
                            self._batch_size,
                        ),
                    ],
                    axis=1,
                )
        else:
            if state.ndim < 2:
                state = tf.expand_dims(state, axis=0)
            self.state = tf.tile(state, [self._batch_size, 1])

        return self._get_reset_return_val()

    def render(self, mode="human"):
        if self._batch_size == 1:
            return super().render(mode)
        else:
            raise NotImplementedError("Rendering not implemented for batched mode")
