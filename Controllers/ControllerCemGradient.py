from importlib import import_module
import numpy as np
import tensorflow as tf
from gym import Env
from yaml import FullLoader, load

from Controllers import Controller

config = load(open("config.yml", "r"), Loader=FullLoader)
if config["debug"]:
    tf.config.run_functions_eagerly(True)


class ControllerCemGradient(Controller):
    def __init__(self, environment: Env, **controller_config) -> None:
        super().__init__(environment, **controller_config)

        self._num_rollouts = controller_config["cem_rollouts"]
        self._horizon_steps = int(
            controller_config["mpc_horizon"] / controller_config["dt"]
        )
        self._outer_it = controller_config["cem_outer_it"]
        self._grad_max = controller_config["grad_max"]
        self._grad_alpha = controller_config["grad_alpha"]
        self._select_best_k = controller_config["cem_best_k"]
        self._initial_action_variance = tf.constant(
            controller_config["cem_initial_action_variance"], dtype=tf.float32
        )

        self.dist_mean = tf.zeros([1, self._horizon_steps], dtype=tf.float32)
        self.dist_stdev = tf.sqrt(self._initial_action_variance) * tf.ones(
            [1, self._horizon_steps], dtype=tf.float32
        )

        _planning_env_config = environment.unwrapped.config.copy()
        _planning_env_config.update({"computation_lib": "tensorflow"})
        self._predictor_environment = getattr(
            import_module(f"Predictors.{controller_config['predictor']}"),
            controller_config["predictor"],
        )(
            environment.unwrapped.__class__(
                batch_size=self._num_rollouts, **_planning_env_config
            ),
            controller_config["seed"],
        )

    def _rollout_trajectories(self, Q: tf.Tensor, rollout_trajectory: tf.Tensor = None):
        traj_cost = tf.zeros([self._num_rollouts])
        for horizon_step in range(self._horizon_steps):
            new_obs, reward, done, info = self._predictor_environment.step(
                self.Q[:, horizon_step, tf.newaxis]
            )
            traj_cost -= reward
            s = new_obs
            if rollout_trajectory is not None:
                rollout_trajectory[:, horizon_step + 1, :] = s.numpy()
        return traj_cost, rollout_trajectory

    def _predict_and_cost(self, s: tf.Tensor):
        # Sample input trajectories and clip
        self.Q = tf.tile(
            self.dist_mean, [self._num_rollouts, 1]
        ) + self.dist_stdev * self._rng_tf.normal(
            [self._num_rollouts, self._horizon_steps], dtype=tf.float32
        )
        self.Q = tf.clip_by_value(
            self.Q, self._env.action_space.low, self._env.action_space.high
        )

        # Rollout trajectories, record gradients
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.Q)
            rollout_trajectory = np.zeros(
                [self._num_rollouts, self._horizon_steps + 1, self._n], dtype=np.float32
            )
            rollout_trajectory[:, 0, :] = s.numpy()
            self.J, rollout_trajectory = self._rollout_trajectories(
                self.Q, rollout_trajectory
            )

        # Compute gradient and clip for each rollout where max value is surpassed
        dJ_dQ = tape.gradient(self.J, self.Q)
        dJ_dQ_max = tf.reduce_max(tf.abs(dJ_dQ), axis=1, keepdims=True)
        mask = dJ_dQ_max > self._grad_max

        dJ_dQ_clipped = tf.cast(~mask, dtype=tf.float32) * dJ_dQ + tf.cast(
            mask, dtype=tf.float32
        ) * self._grad_max * (dJ_dQ / dJ_dQ_max)

        # Vanilla gradient descent
        self.Q = self.Q - self._grad_alpha * dJ_dQ_clipped
        self.Q = tf.clip_by_value(
            self.Q, self._env.action_space.low, self._env.action_space.high
        )

        # Final rollout
        self._predictor_environment.reset(state=rollout_trajectory[0, 0, :].copy())
        self.J, _ = self._rollout_trajectories(self.Q)

        # Sort for best costs
        costs_sorted = tf.argsort(self.J)
        best_idx = costs_sorted[0 : self._select_best_k]
        Q_keep = tf.gather(self.Q, best_idx, axis=0)

        # Update sampling distribution
        self.dist_mean = tf.math.reduce_mean(Q_keep, axis=0, keepdims=True)
        self.dist_stdev = tf.math.reduce_std(Q_keep, axis=0, keepdims=True)

    def step(self, s: np.ndarray) -> np.ndarray:
        self.s = s.copy()
        self._predictor_environment.reset(s)
        s = self._predictor_environment.get_state()

        for _ in range(self._outer_it):
            self._predict_and_cost(s)

        self.dist_stdev = tf.clip_by_value(self.dist_stdev, 0.1, 10)
        self.dist_stdev = tf.concat(
            [self.dist_stdev[:, 1:], tf.sqrt([[self._initial_action_variance]])], -1
        )
        self.u = tf.expand_dims(self.dist_mean[0, 0], 0).numpy()
        self._update_logs()
        self.dist_mean = tf.concat(
            [self.dist_mean[:, 1:], tf.constant(0.0, shape=[1, 1])], -1
        )
        return self.u
