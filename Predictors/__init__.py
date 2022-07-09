from typing import Any, Optional, Tuple, Union

from Environments import EnvironmentBatched

from numpy.random import Generator, SFC64

import numpy as np
import tensorflow as tf
import torch


class Predictor(object):
    def __init__(self, environment: EnvironmentBatched, seed: int) -> None:
        self._env = environment
        self._rng = self._env.lib.create_rng(seed)
        self.n_obs = sum(self._env.observation_space.shape)

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError as error:
            try:
                return object.__getattribute__(self._env, name)
            except AttributeError:
                raise error

    def step(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        raise NotImplementedError()

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        raise NotImplementedError()

    def train(self, dataset: Union[np.ndarray, tf.Tensor, torch.Tensor]) -> None:
        raise NotImplementedError()

    def get_state(self) -> Union[np.ndarray, tf.Tensor, torch.Tensor]:
        raise NotImplementedError()
