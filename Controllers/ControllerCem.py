from importlib import import_module
import numpy as np
import tensorflow as tf
from gym import Env
from yaml import FullLoader, load

from Controllers import Controller

config = load(open("config.yml", "r"), Loader=FullLoader)
if config["debug"]:
    tf.config.run_functions_eagerly(True)


class ControllerCem(Controller):
    def __init__(self, environment: Env, **controller_config) -> None:
        super().__init__(environment, **controller_config)

        self._num_rollouts = controller_config["cem_rollouts"]
        self._horizon_steps = int(
            controller_config["mpc_horizon"] / controller_config["dt"]
        )
        self._outer_it = controller_config["cem_outer_it"]
        self._best_k = controller_config["cem_best_k"]
        self._stdev_min = controller_config["cem_stdev_min"]
        self._initial_action_variance = controller_config["cem_initial_action_variance"]

        self.dist_mean = np.zeros((1, self._horizon_steps))
        self.dist_stdev = np.sqrt(self._initial_action_variance) * np.ones(
            (1, self._horizon_steps)
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

    def predict_and_cost(self, s: np.ndarray, Q: np.ndarray):
        # rollout trajectories and retrieve cost
        rollout_trajectory = np.empty(
            (self._num_rollouts, self._horizon_steps + 1, self._n), dtype=np.float32
        )
        rollout_trajectory[:, 0, :] = s.numpy()
        traj_cost = np.zeros((self._num_rollouts))

        for horizon_step in range(self._horizon_steps):
            new_obs, reward, done, info = self._predictor_environment.step(
                Q[:, horizon_step, tf.newaxis].numpy()
            )
            traj_cost -= reward
            s = new_obs
            rollout_trajectory[:, horizon_step + 1, :] = s.numpy()

        return traj_cost, rollout_trajectory

    def step(self, s: np.ndarray) -> np.ndarray:
        self.s = s.copy()
        self._predictor_environment.reset(state=s)
        s = self._predictor_environment.get_state()

        for _ in range(0, self._outer_it):
            # generate random input sequence and clip to control limits
            self.Q = np.tile(
                self.dist_mean, (self._num_rollouts, 1)
            ) + self.dist_stdev * self._rng_np.standard_normal(
                size=(self._num_rollouts, self._horizon_steps), dtype=np.float32
            )
            self.Q = np.clip(self.Q, -1.0, 1.0, dtype=np.float32)
            self.Q = tf.convert_to_tensor(self.Q, dtype=tf.float32)

            # rollout the trajectories and get cost
            self.J, rollout_trajectory = self.predict_and_cost(s, self.Q)
            self.Q = self.Q.numpy()
            # sort the costs and find best k costs
            sorted_cost = np.argsort(self.J)
            best_idx = sorted_cost[0 : self._best_k]
            elite_Q = self.Q[best_idx, :]
            # update the distribution for next inner loop
            self.dist_mean = np.mean(elite_Q, axis=0)
            self.dist_stdev = np.std(elite_Q, axis=0)

        # after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        self.dist_stdev = np.clip(self.dist_stdev, self._stdev_min, None)
        self.dist_stdev = np.append(
            self.dist_stdev[1:], np.sqrt(self._initial_action_variance)
        ).astype(np.float32)
        self.u = np.array([self.dist_mean[0]], dtype=np.float32)
        self._update_logs()
        self.dist_mean = np.append(self.dist_mean[1:], 0).astype(np.float32)
        return self.u

    def controller_reset(self):
        self.dist_mean = np.zeros([1, self._horizon_steps])
        self.dist_stdev = np.sqrt(self._initial_action_variance) * np.ones(
            (1, self._horizon_steps)
        )
