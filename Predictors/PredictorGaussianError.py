from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from Environments import EnvironmentBatched

from Predictors import Predictor

from yaml import load, FullLoader

config = load(open("config.yml", "r"), Loader=FullLoader)


class PredictorGaussianError(Predictor):
    """Predictor which artificially assumes that a model has been fit with Gaussian modeling error.

    :param Predictor: Base Predictor class
    :type Predictor: _type_
    """

    def __init__(self, environment: EnvironmentBatched, seed: int) -> None:
        super().__init__(environment, seed)
        self.stdev = config["3_predictors"][self.__class__.__name__][
            "prediction_error_stdev"
        ]

    def step(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        # Compute true model prediction
        state, _, _, info = self._env.step(action)

        state += self.stdev * self._env.lib.standard_normal(self._rng, state.shape)
        reward = self._env.get_reward(state, action)
        done = self._env.is_done(state)

        return state, reward, done, info

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        return self._env.reset(state, seed, return_info, options)

    def get_state(self) -> Union[np.ndarray, tf.Tensor, torch.Tensor]:
        return self._env.state
