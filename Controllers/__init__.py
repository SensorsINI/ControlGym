from numpy.random import default_rng
from gym import Env


class Controller:
    def __init__(self, environment: Env, seed, ) -> None:
        self._n = environment.observation_space.shape[0]
        self._m = environment.action_space.shape[0]
        self._rng = default_rng(seed=seed)