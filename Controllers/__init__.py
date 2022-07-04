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
        self._rng_np = Generator(SFC64(controller_config["seed"]))
        self._rng_tf = tf.random.Generator.from_seed(controller_config["seed"])
        self._controller_logging = controller_config["controller_logging"]
        self._save_vars = ["Q_logged", "J_logged", "s_logged", "u_logged"]
        self.logs = {s: [] for s in self._save_vars}

    def step(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def controller_reset(self) -> None:
        raise NotImplementedError()

    def get_outputs(self) -> dict[np.ndarray]:
        """Retrieve a dictionary of controller outputs. These could be saved traces of input plans or the like.

        :return: A dictionary of numpy arrays
        :rtype: dict[np.ndarray]
        """
        return {k: np.stack(v, axis=0) for k, v in self.logs.items()}

    def _update_logs(self):
        if self._controller_logging:
            for name, var in zip(self._save_vars, [self.Q, self.J, self.s, self.u]):
                self.logs[name].append(var.numpy().copy() if hasattr(var, "numpy") else var.copy())
