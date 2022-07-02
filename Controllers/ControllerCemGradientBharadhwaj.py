from turtle import forward
import tensorflow as tf
from Environments import EnvironmentBatched
import types
import numpy as np
from copy import deepcopy

# Import original paper's code
from mpc.gradcem import GradCEMPlan

from Controllers import Controller
from Predictors.predictor_euler import EulerPredictor


class GradCemPlanTF(GradCEMPlan):
    def __init__(self, learning_rate: float, rng: tf.random.Generator, **kwargs):
        super().__init__(**kwargs)
        self._lr = learning_rate
        self._rng = rng

    def forward(
        self, batch_size, return_plan=False, return_plan_each_iter=False, max_grad=1000
    ):
        # Here batch is strictly if multiple Plans should be performed!
        B = batch_size

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        a_mu = tf.zeros((self.H, B, 1, self.a_size))
        a_std = tf.ones((self.H, B, 1, self.a_size))

        # Sample actions (T x (B*K) x A)
        actions = tf.clip_by_value(
            tf.reshape(
                a_mu + a_std * self._rng.normal((self.H, B, self.K, self.a_size)),
                (self.H, B * self.K, self.a_size),
            ),
            self.env.action_space.low,
            self.env.action_space.high,
        )

        # optimizer = tf.keras.optimizers.SGD(learning_rate=self._lr, momentum=0.0)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self._lr)
        plan_each_iter = []
        for _ in range(self.opt_iters):
            actions = tf.Variable(actions)
            # self.env.reset_state(B * self.K)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)

                # Returns (B*K)
                returns = self.env.rollout(actions)
                tot_cost = (-1.0) * returns

            dJ_dQ = tape.gradient(tot_cost, actions)

            # grad clip
            # Find norm across batch
            if self.grad_clip:
                epsilon = 1e-6
                actions_grad_norm = (
                    tf.norm(dJ_dQ, ord="euclidean", axis=0, keepdims=True) + epsilon
                )

                # # Normalize by that
                dJ_dQ = (
                    dJ_dQ * tf.clip_by_value(actions_grad_norm, 0, max_grad)
                ) / actions_grad_norm

            optimizer.apply_gradients(zip([dJ_dQ], [actions]))

            _, topk = tf.math.top_k(
                tf.reshape(returns, (B, self.K)), k=self.top_K, sorted=True
            )
            topk += self.K * tf.expand_dims(tf.range(0, B, dtype=tf.int32), axis=1)
            best_actions = tf.reshape(
                tf.gather(actions, tf.reshape(topk, -1), axis=1),
                (self.H, B, self.top_K, self.a_size),
            )
            a_mu = tf.reduce_mean(best_actions, axis=2, keepdims=True)
            a_std = tf.math.reduce_std(best_actions, axis=2, keepdims=True)

            if return_plan_each_iter:
                _, topk = tf.math.top_k(
                    tf.reshape(returns, (B, self.K)), k=1, sorted=True
                )
                best_plan = tf.reshape(
                    actions[:, topk[0]], (self.H, B, self.a_size)
                ).numpy()
                plan_each_iter.append(best_plan.copy())

            # There must be cleaner way to do this
            k_resamp = self.K - self.top_K
            _, botn_k = tf.math.top_k(
                tf.reshape(returns, (B, self.K)), k=k_resamp, sorted=False
            )
            botn_k += self.K * tf.expand_dims(tf.range(0, B, dtype=tf.int32), axis=1)

            resample_actions = tf.clip_by_value(
                tf.reshape(
                    a_mu
                    + a_std
                    * self._rng.normal(
                        (self.H, B, k_resamp, self.a_size), dtype=tf.float32
                    ),
                    (self.H, B * k_resamp, self.a_size),
                ),
                self.env.action_space.low,
                self.env.action_space.high,
            )
            actions = actions.numpy()
            actions[:, tf.reshape(botn_k, -1)] = resample_actions.numpy()
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        # Re-fit belief to the K best action sequences
        self.J = tot_cost.numpy()
        self.Q = actions.numpy()
        _, topk = tf.math.top_k(tf.reshape(returns, (B, self.K)), k=1, sorted=True)
        best_plan = self.Q[:, topk[0]].reshape((self.H, B, self.a_size))
        self.Q = tf.squeeze(tf.transpose(self.Q, perm=(1, 0, 2)))

        if return_plan_each_iter:
            return plan_each_iter
        if return_plan:
            return best_plan
        else:
            return best_plan[0]


class WrappedEnv:
    def build_from_env(self, environment: EnvironmentBatched, batch_size: int):
        self.__dict__.update(environment.unwrapped.__dict__.copy())
        self._env = deepcopy(environment.unwrapped)
        self._predictor_environment = EulerPredictor(
            environment.unwrapped.__class__(
                batch_size=batch_size, **environment.unwrapped.config
            )
        )
        self.a_size = self._env.action_space.shape[0]
        return self

    def reset_state(self, batch_size):
        self._env.reset()

    def rollout(self, actions: tf.Variable):
        self._predictor_environment.reset(self._env.state)
        horizon_length, batch_size, _ = actions.shape
        traj_reward = tf.zeros([batch_size], dtype=tf.float32)
        for horizon_step in range(horizon_length):
            new_obs, reward, done, info = self._predictor_environment.step(
                actions[horizon_step, :, :]
            )
            traj_reward += reward

        return traj_reward


class ControllerCemGradientBharadhwaj(Controller):
    def __init__(self, environment: EnvironmentBatched, **controller_config) -> None:
        super().__init__(environment, **controller_config)

        self._num_rollouts = controller_config["cem_rollouts"]
        self._horizon_steps = int(
            controller_config["mpc_horizon"] / controller_config["dt"]
        )
        self._opt_iters = controller_config["cem_outer_it"]
        self._select_best_k = controller_config["cem_best_k"]
        self._max_grad = controller_config["max_grad"]

        self._controller = GradCemPlanTF(
            planning_horizon=self._horizon_steps,
            opt_iters=self._opt_iters,
            samples=self._num_rollouts,
            top_samples=self._select_best_k,
            env=WrappedEnv().build_from_env(environment, batch_size=self._num_rollouts),
            device=tf.device("/physical_device:CPU:0"),
            grad_clip=True,
            learning_rate=controller_config["grad_learning_rate"],
            rng=self._rng_tf,
        )

    def step(self, s: np.ndarray) -> np.ndarray:
        self.u = np.reshape(
            self._controller.forward(
                batch_size=1,
                return_plan=False,
                return_plan_each_iter=False,
                max_grad=self._max_grad,
            ),
            (self._m,),
        )
        self.Q, self.J, self.s = (
            self._controller.Q,
            self._controller.J,
            s.copy(),
        )
        self._update_logs()
        return self.u
