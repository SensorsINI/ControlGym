from copy import copy, deepcopy
import tensorflow as tf

from Configs.default_cem import ENV

tf.config.run_functions_eagerly(True)
import numpy as np
from gym import Env, vector

from Controllers import Controller

class ControllerCem(Controller):
    def __init__(self, environment: Env, **controller_config) -> None:
        super().__init__(environment=environment, seed=controller_config["SEED"])

        self._num_rollouts = controller_config["cem_rollouts"]
        self._horizon_steps = int(controller_config["mpc_horizon"] / controller_config["dt"])
        self._outer_it = controller_config["cem_outer_it"]
        self._best_k = controller_config["cem_best_k"]
        self._stdev_min = controller_config["cem_stdev_min"]
        self._initial_action_variance = controller_config["cem_initial_action_variance"]

        self.dist_mean = np.zeros((1, self._horizon_steps))
        self.dist_stdev = np.sqrt(self._initial_action_variance) * np.ones((1, self._horizon_steps))

    @tf.function(jit_compile=True)
    def predict_and_cost(self, s: np.ndarray, Q: np.ndarray, target_position: np.ndarray):
        # rollout trajectories and retrieve cost
        rollout_trajectory = np.empty((self._num_rollouts, self._horizon_steps+1, self._n), dtype=np.float32)
        rollout_trajectory[:, 0, :] = s.numpy().copy()
        traj_cost = np.zeros((self._num_rollouts))

        for horizon_step in range(self._horizon_steps):
            # TODO: Create a list of predictor envs, one for each rollout
            # TODO: Implement this more efficiently?
            new_obs, reward, done, info = self._predictor_environment.step(Q[:, horizon_step, tf.newaxis].numpy())
            traj_cost -= reward
            s = new_obs
            rollout_trajectory[:, horizon_step+1, :] = s.copy()

        return traj_cost, rollout_trajectory
    
    def step(self, s: np.ndarray, env: Env, target_position: np.ndarray=0, time=None):
        self._predictor_environment = ENV(batch_size=self._num_rollouts)
        self._predictor_environment.reset(s)

        s = np.tile(s, tf.constant([self._num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        for _ in range(0, self._outer_it):
            #generate random input sequence and clip to control limits
            Q = (
                np.tile(self.dist_mean, (self._num_rollouts, 1))
                + self.dist_stdev * self._rng.standard_normal(size=(self._num_rollouts, self._horizon_steps), dtype=np.float32)
            )
            Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)

            Q = tf.convert_to_tensor(Q, dtype=tf.float32)
            target_position = tf.convert_to_tensor(target_position, dtype=tf.float32)

            #rollout the trajectories and get cost
            traj_cost, rollout_trajectory = self.predict_and_cost(s, Q, target_position)
            Q = Q.numpy()
            #sort the costs and find best k costs
            sorted_cost = np.argsort(traj_cost)
            best_idx = sorted_cost[0: self._best_k]
            elite_Q = Q[best_idx, :]
            #update the distribution for next inner loop
            self.dist_mean = np.mean(elite_Q, axis=0)
            self.dist_stdev = np.std(elite_Q, axis=0)

        #after all inner loops, clip std min, so enough is explored and shove all the values down by one for next control input
        self.dist_stdev = np.clip(self.dist_stdev, self._stdev_min, None)
        self.dist_stdev = np.append(self.dist_stdev[1:], np.sqrt(self._initial_action_variance)).astype(np.float32)
        self.u = self.dist_mean[0]
        self.dist_mean = np.append(self.dist_mean[1:], 0).astype(np.float32)
        return np.array([self.u])

    def controller_reset(self):
        self.dist_mean = np.zeros([1, self._horizon_steps])
        self.dist_stdev = np.sqrt(self._initial_action_variance) * np.ones((1, self._horizon_steps))