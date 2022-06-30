import numpy as np
from numpy.random import Generator, SFC64
import tensorflow as tf
from gym import Env
from gym.spaces.box import Box


class Controller:
    def __init__(
        self,
        environment: Env,
        **controller_config,
    ) -> None:
        self._env = environment
        assert isinstance(self._env.action_space, Box)
        self._n = environment.observation_space.shape[0]
        self._m = environment.action_space.shape[0]
        self._rng_np = Generator(SFC64(seed=controller_config["seed"]))
        self._rng_tf = tf.random.Generator.from_seed(controller_config["seed"])
        self._controller_logging = controller_config["controller_logging"]
        self.Q_logged = []
        self.J_logged = []

    def step(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def controller_reset(self) -> None:
        raise NotImplementedError()

    def get_outputs(self) -> dict[np.ndarray]:
        """Retrieve a dictionary of controller outputs. These could be saved traces of input plans or the like.

        :return: A dictionary of numpy arrays
        :rtype: dict[np.ndarray]
        """
        return {
            "Q_logged": np.stack(self.Q_logged, axis=0),
            "J_logged": np.stack(self.J_logged, axis=0),
        }

    def _update_logs(self):
        if self._controller_logging:
            self.Q_logged.append(self.Q.numpy() if hasattr(self.Q, "numpy") else self.Q)
            self.J_logged.append(self.J.numpy() if hasattr(self.Q, "numpy") else self.J)
