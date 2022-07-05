from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from Utilities.utils import get_logger

from Predictors import Predictor

logger = get_logger(__name__)


class PredictorEuler(Predictor):
    """Predictor which utilizes the environment's actual dynamics equation"""

    def step(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        # Return true environment prediction
        return self._env.step(action)

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        return self._env.reset(state, seed, return_info, options)

    def train(self, dataset: Union[np.ndarray, tf.Tensor, torch.Tensor]) -> None:
        logger.info("No training required for Euler predictor.")

    def get_state(self) -> Union[np.ndarray, tf.Tensor, torch.Tensor]:
        return self._env.state
