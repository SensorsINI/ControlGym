from importlib import import_module

import numpy as np
import tensorflow as tf
from gym import Env
from gym.spaces.box import Box

from Controllers import Controller


class ControllerCemGradient(Controller):
    def __init__(self, environment: Env, **controller_config) -> None:
        super().__init__(environment)

        assert isinstance(self._env.action_space, Box)

        self._rng = tf.random.Generator.from_seed(controller_config["SEED"])

        self._num_rollouts = controller_config["cem_rollouts"]
        self._horizon_steps = int(
            controller_config["mpc_horizon"] / controller_config["dt"]
        )
        self._outer_it = controller_config["cem_outer_it"]
        self._max_grad = controller_config["max_grad"]
        self._grad_alpha = controller_config["grad_alpha"]
        self._select_best_k = controller_config["cem_best_k"]
        self._initial_action_variance = tf.constant(
            controller_config["cem_initial_action_variance"], dtype=tf.float32
        )

        self.dist_mean = tf.zeros([1, self._horizon_steps], dtype=tf.float32)
        self.dist_stdev = tf.sqrt(self._initial_action_variance) * tf.ones(
            [1, self._horizon_steps], dtype=tf.float32
        )
        self.u = 0.0
        self._predictor_environment = environment.unwrapped.__class__(
            batch_size=self._num_rollouts
        )

    # @tf.function(jit_compile=True)
    def _predict_and_cost(self, s: tf.Tensor):
        # Sample input trajectories and clip
        Q = tf.tile(
            self.dist_mean, [self._num_rollouts, 1]
        ) + self.dist_stdev * self._rng.normal(
            [self._num_rollouts, self._horizon_steps], dtype=tf.float32
        )
        Q = tf.clip_by_value(Q, self._env.action_space.low, self._env.action_space.high)

        # Rollout trajectories, record gradients
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            rollout_trajectory = np.zeros(
                [self._num_rollouts, self._horizon_steps + 1, self._n], dtype=np.float32
            )
            rollout_trajectory[:, 0, :] = s.numpy()
            traj_cost = tf.zeros([self._num_rollouts])

            for horizon_step in range(self._horizon_steps):
                new_obs, reward, done, info = self._predictor_environment.step(
                    Q[:, horizon_step, tf.newaxis]
                )
                traj_cost -= reward
                s = new_obs
                rollout_trajectory[:, horizon_step + 1, :] = s.numpy()

        # Compute gradient and clip for each rollout where max value is surpassed
        dJ_dQ = tape.gradient(traj_cost, Q)
        dJ_dQ_max = tf.reduce_max(tf.abs(dJ_dQ), axis=1, keepdims=True)
        mask = dJ_dQ_max > self._max_grad

        dJ_dQ_clipped = tf.cast(~mask, dtype=tf.float32) * dJ_dQ + tf.cast(
            mask, dtype=tf.float32
        ) * self._max_grad * (dJ_dQ / dJ_dQ_max)

        # Vanilla gradient descent
        Q_updated = Q - self._grad_alpha * dJ_dQ_clipped
        Q_updated = tf.clip_by_value(
            Q_updated, self._env.action_space.low, self._env.action_space.high
        )

        # Final rollout
        self._predictor_environment.reset(state=rollout_trajectory[0, 0, :].copy())
        final_traj_cost = tf.zeros([self._num_rollouts])
        for horizon_step in range(self._horizon_steps):
            new_obs, reward, done, info = self._predictor_environment.step(
                Q_updated[:, horizon_step, tf.newaxis]
            )
            final_traj_cost -= reward
            s = new_obs

        # Sort for best costs
        costs_sorted = tf.argsort(final_traj_cost)
        best_idx = costs_sorted[0 : self._select_best_k]
        Q_keep = tf.gather(Q_updated, best_idx, axis=0)

        # Update sampling distribution
        self.dist_mean = tf.math.reduce_mean(Q_keep, axis=0, keepdims=True)
        self.dist_stdev = tf.math.reduce_std(Q_keep, axis=0, keepdims=True)

    def step(self, s: np.ndarray) -> np.ndarray:
        self._predictor_environment.reset(s)
        s = self._predictor_environment.state

        for _ in range(self._outer_it):
            self._predict_and_cost(s)

        self.dist_stdev = tf.clip_by_value(self.dist_stdev, 0.1, 10)
        self.dist_stdev = tf.concat(
            [self.dist_stdev[:, 1:], tf.sqrt([[self._initial_action_variance]])], -1
        )
        u = self.dist_mean[0, 0]
        self.dist_mean = tf.concat(
            [self.dist_mean[:, 1:], tf.constant(0.0, shape=[1, 1])], -1
        )
        return tf.expand_dims(u, 0).numpy()
