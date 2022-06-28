import numpy as np
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
    
    def step(self, s: np.ndarray):
        raise NotImplementedError()
    
    def controller_reset(self):
        raise NotImplementedError()
