import numpy as np
import tensorflow as tf

from typing import Optional, Union, Tuple

from Environments import EnvironmentBatched, NumpyLibrary, cost_functions

from CartPoleSimulation.GymlikeCartPole.CartPoleEnv_LTC import CartPoleEnv_LTC


class cartpole_simulator_batched(EnvironmentBatched, CartPoleEnv_LTC):
    num_actions = 1
    num_states = 6

    def __init__(
        self, batch_size=1, computation_lib=NumpyLibrary, render_mode="human", **kwargs
    ):
        super().__init__()
        self.config = kwargs

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.cost_functions = cost_functions(self)

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        return CartPoleEnv_LTC.reset(
            self, state, seed=seed, return_info=return_info, options=options
        )

    def step_tf(self, state: tf.Tensor, action: tf.Tensor):

        self.action = np.atleast_1d(action).astype(np.float32)

        self.step_physics()

        self.step_termination_and_reward()

        return self.state, self.reward, self.done, {"target": self.CartPoleInstance.target_position}