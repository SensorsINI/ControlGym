import numpy as np
import tensorflow as tf
from gym import Env


class Controller:
    def __init__(
        self,
        environment: Env,
        **controller_config,
    ) -> None:
        self._n = environment.observation_space.shape[0]
        self._m = environment.action_space.shape[0]
        self._env = environment
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
            self.Q_logged.append(
                self.Q.numpy() if isinstance(self.Q, tf.Tensor) else self.Q
            )
            self.J_logged.append(
                self.J.numpy() if isinstance(self.J, tf.Tensor) else self.J
            )
