from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf


class Predictor:
    def step(
        self, action: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor],
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

    def train(self, dataset: Union[np.ndarray, tf.Tensor]) -> None:
        raise NotImplementedError()

    def get_state(self) -> Union[np.ndarray, tf.Tensor]:
        raise NotImplementedError()
