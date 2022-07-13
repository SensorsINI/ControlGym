from importlib import import_module
import numpy as np
import tensorflow as tf
from gym import Env

from Controllers import Controller
from Environments import TensorFlowLibrary
from Utilities.utils import CompileTF


class ControllerMPPIGradient(Controller):
    def __init__(self, environment: Env, **controller_config) -> None:
        super().__init__(environment, **controller_config)

        self._num_rollouts = controller_config["cem_rollouts"]
        self._horizon_steps = int(
            controller_config["mpc_horizon"] / controller_config["dt"]
        )
        self._outer_it = controller_config["cem_outer_it"]
        self._grad_max = controller_config["grad_max"]
        self._select_best_k = controller_config["cem_best_k"]
        self._initial_action_variance = tf.constant(
            controller_config["cem_initial_action_variance"], dtype=tf.float32
        )
        self._minimal_action_stdev = controller_config["cem_stdev_min"]
        self._resamp_every = controller_config["resamp_every"]
        self._do_warmup = controller_config["do_warmup"]
        self._lambda = controller_config["mppi_lambda"]

        self.dist_mean = tf.zeros([1, self._horizon_steps], dtype=tf.float32)
        self.dist_stdev = tf.sqrt(self._initial_action_variance) * tf.ones(
            [1, self._horizon_steps], dtype=tf.float32
        )

        self.Q = tf.Variable(self._sample_inputs((self._num_rollouts, self._horizon_steps)))
        self.iteration = 0
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=controller_config["grad_alpha"],
            beta_1=controller_config["grad_beta_1"],
            beta_2=controller_config["grad_beta_2"],
            epsilon=controller_config["grad_epsilon"],
        )

        _planning_env_config = environment.unwrapped.config.copy()
        _planning_env_config.update({"computation_lib": TensorFlowLibrary})
        self._predictor_environment = getattr(
            import_module(f"Predictors.{controller_config['predictor']}"),
            controller_config["predictor"],
        )(
            environment.unwrapped.__class__(
                batch_size=self._num_rollouts, **_planning_env_config
            ),
            controller_config["seed"],
        )

    def _sample_inputs(self, shape: "tuple[int]"):
        Q = tf.sqrt(self._initial_action_variance) * self._rng_tf.normal(
            shape, dtype=tf.float32
        )
        Q = tf.clip_by_value(Q, self._env.action_space.low, self._env.action_space.high)
        return Q

    def _rollout_trajectories(
        self, Q: tf.Variable, rollout_trajectory: tf.Tensor = None
    ):
        traj_cost = tf.zeros([self._num_rollouts], dtype=tf.float32)
        for horizon_step in range(self._horizon_steps):
            new_obs, reward, done, info = self._predictor_environment.step(
                Q[:, horizon_step, tf.newaxis]
            )
            traj_cost -= reward
            s = new_obs
            if rollout_trajectory is not None:
                rollout_trajectory = tf.stop_gradient(
                    tf.concat([rollout_trajectory, tf.expand_dims(s, axis=1)], axis=1)
                )

        return traj_cost, rollout_trajectory

    # @CompileTF
    def _grad_step(self, s: tf.Tensor, Q: tf.Variable):
        self._predictor_environment.reset(self.s)
        rollout_trajectory = tf.expand_dims(s, axis=1)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(Q)
            traj_cost, rollout_trajectory = self._rollout_trajectories(
                Q, rollout_trajectory
            )

        dJ_dQ = tape.gradient(traj_cost, Q)
        # dJ_dQ_clipped = tf.clip_by_norm(dJ_dQ, self._grad_max, axes=1)
        dJ_dQ_clipped = tf.clip_by_norm(dJ_dQ, self._grad_max)

        self.opt.apply_gradients(zip([dJ_dQ_clipped], [Q]))
        Q_updated = tf.clip_by_value(
            Q, self._env.action_space.low, self._env.action_space.high
        )
        return Q_updated

    def _retrieve_action(self, s0: np.ndarray, Q: tf.Variable):
        self._predictor_environment.reset(state=s0)
        self.J, _ = self._rollout_trajectories(Q)

        # Do MPPI-style averaging
        rho = tf.math.reduce_min(self.J)
        exp_s = tf.expand_dims(tf.exp(-1.0 / self._lambda * (self.J - rho)), axis=1)
        a = tf.math.reduce_sum(exp_s)
        u = (tf.math.reduce_sum(exp_s * Q, axis=0) / a)[0][tf.newaxis]

        sorted_cost = tf.argsort(self.J)
        best_idx = sorted_cost[: self._select_best_k]
        Q_keep = tf.gather(Q, best_idx, axis=0)

        self.dist_mean = tf.reduce_mean(Q_keep, axis=0, keepdims=True)
        self.dist_mean = tf.concat([self.dist_mean[:, 1:], tf.zeros([1, 1])], -1)
        self.dist_stdev = tf.math.reduce_std(Q_keep, axis=0, keepdims=True)
        self.dist_stdev = tf.clip_by_value(
            self.dist_stdev, self._minimal_action_stdev, 10.0
        )
        self.dist_stdev = tf.concat(
            [
                self.dist_stdev[:, 1:],
                tf.sqrt(self._initial_action_variance)[tf.newaxis, tf.newaxis],
            ],
            axis=1,
        )

        return u, Q_keep, best_idx

    def step(self, s: np.ndarray) -> np.ndarray:
        # s: ndarray(n,)
        self.s = s.copy()
        self._predictor_environment.reset(s)
        s = self._predictor_environment.get_state()
        # s: Tensor(num_rollouts, n)

        # If warm-start: Increase initial optimization steps
        iterations = self._outer_it
        if self.iteration == 0 and self._do_warmup:
            iterations *= self._horizon_steps

        # Optimize input plans w.r.t. cost
        for _ in range(self._outer_it):
            Q_updated = self._grad_step(s, self.Q)
            self.Q.assign(Q_updated)

        # Final rollout
        self.u, Q_keep, best_idx = self._retrieve_action(self.s, self.Q)
        self.u = self.u.numpy()

        # Adam variables: m, v (1st/2nd moment of the gradient computation)
        adam_weights = self.opt.get_weights()
        if self.iteration % self._resamp_every == 0:
            Q_new = tf.concat(
                [
                    tf.concat(
                        [
                            self._sample_inputs(
                                (
                                    self._num_rollouts - self._select_best_k,
                                    self._horizon_steps - 1,
                                )
                            ),
                            Q_keep[:, 1:],
                        ],
                        axis=0,
                    ),
                    self._sample_inputs((self._num_rollouts, 1)),
                ],
                axis=1,
            )
            # Shift Adam weights of kept trajectories by one
            # Set all new Adam weights to 0
            w_m = tf.concat(
                [
                    tf.zeros(
                        [self._num_rollouts - self._select_best_k, self._horizon_steps],
                        dtype=tf.float32,
                    ),
                    tf.concat(
                        [
                            tf.gather(adam_weights[1], best_idx)[:, 1:],
                            tf.zeros([self._select_best_k, 1]),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            w_v = tf.concat(
                [
                    tf.zeros(
                        [self._num_rollouts - self._select_best_k, self._horizon_steps],
                        dtype=tf.float32,
                    ),
                    tf.concat(
                        [
                            tf.gather(adam_weights[2], best_idx)[:, 1:],
                            tf.zeros([self._select_best_k, 1]),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
        else:
            Q_new = tf.concat(
                [
                    self.Q[:, 1:],
                    self._sample_inputs((self._horizon_steps, 1)),
                ],
                axis=1,
            )
            w_m = tf.concat(
                [
                    adam_weights[1][:, 1:],
                    tf.zeros([self._num_rollouts, 1], dtype=tf.float32),
                ],
                axis=1,
            )
            w_v = tf.concat(
                [
                    adam_weights[2][:, 1:],
                    tf.zeros([self._num_rollouts, 1], dtype=tf.float32),
                ],
                axis=1,
            )

        self.opt.set_weights([adam_weights[0], w_m, w_v])
        self._update_logs()
        self.Q.assign(Q_new)
        self.iteration += 1
        return self.u
