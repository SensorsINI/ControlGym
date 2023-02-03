import yaml
import os
import numpy as np

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
altitude_weight = float(
    config["continuous_mountaincar_batched"]["default"]["altitude_weight"]
)
done_reward = float(config["continuous_mountaincar_batched"]["default"]["done_reward"])
control_penalty = float(
    config["continuous_mountaincar_batched"]["default"]["control_penalty"]
)

thdot_weight = 0.1
control_cost_weight = 0.001


class default(cost_function_base):
    MAX_COST = (0.5 * np.pi) ** 2 + thdot_weight * 64.0 + control_cost_weight * 4.0
    
    def _angle_normalize(self, x):
        _pi = self.lib.to_tensor(self.lib.pi, self.lib.float32)
        return ((x + _pi) % (2 * _pi)) - _pi
    
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        th, thdot, sinth, costh = self.lib.unstack(states, 4, -1)
        costs = (
            self._angle_normalize(th) ** 2
            + thdot_weight * thdot**2
            + control_cost_weight * (inputs[:, 0] ** 2)
        )
        return costs
