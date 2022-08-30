import numpy as np
from numpy.random import Generator, SFC64
import tensorflow as tf
from gym import Env
from gym.spaces.box import Box


class Controller:
    def __init__(
        self,
        environment: Env,
        seed: int,
        controller_logging: bool,
        **kwargs,
    ) -> None:
        self._env = environment
        assert isinstance(self._env.action_space, Box)
        self._n = environment.observation_space.shape[0]
        self._m = environment.action_space.shape[0]
        self._rng_np = Generator(SFC64(seed))
        self._rng_tf = tf.random.Generator.from_seed(seed)
        self._controller_logging = controller_logging
        self.save_vars = [
            "Q_logged",
            "J_logged",
            "realized_cost_logged",
            "s_logged",
            "u_logged",
            "trajectory_ages_logged",
        ]
        self.logs = {s: [] for s in self.save_vars}
        for v in self.save_vars:
            setattr(self, v, None)

    def step(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def controller_reset(self) -> None:
        raise NotImplementedError()

    def get_outputs(self) -> "dict[str, np.ndarray]":
        """Retrieve a dictionary of controller outputs. These could be saved traces of input plans or the like.

        :return: A dictionary of numpy arrays
        :rtype: dict[str, np.ndarray]
        """
        return {
            k: np.stack(v, axis=0) if len(v) > 0 else None for k, v in self.logs.items()
        }

    def update_logs(self) -> None:
        if self._controller_logging:
            for name, var in zip(
                self.save_vars, [getattr(self, var_name) for var_name in self.save_vars]
            ):
                if var is not None:
                    self.logs[name].append(
                        var.numpy().copy() if hasattr(var, "numpy") else var.copy()
                    )
