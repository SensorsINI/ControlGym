import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
angle_weight = float(config["cartpole_simulator_batched"]["default"]["angle_weight"])
position_weight = float(config["cartpole_simulator_batched"]["default"]["position_weight"])


class default(cost_function_base):
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        angle, angleD, angle_cos, angle_sin, position, positionD = self.lib.unstack(states, 6, -1)
        cost = (
            - angle_weight * angle_cos
            + position_weight * (position - self.controller.target_position) ** 2
        )
        return cost

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
