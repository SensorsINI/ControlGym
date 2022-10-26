import yaml
import os
import numpy as np

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
angle_weight = float(config["continuous_cartpole_batched"]["default"]["angle_weight"])
angleD_weight = float(config["continuous_cartpole_batched"]["default"]["angleD_weight"])
position_weight = float(config["continuous_cartpole_batched"]["default"]["position_weight"])
positionD_weight = float(config["continuous_cartpole_batched"]["default"]["positionD_weight"])


class default(cost_function_base):
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        x, x_dot, theta, theta_dot = self.lib.unstack(states, 4, -1)
        cost = (
            angle_weight * (theta**2)
            + angleD_weight * (theta_dot**2)
            + position_weight * (x**2)
            + positionD_weight * (x_dot**2)
        )
        return cost

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
