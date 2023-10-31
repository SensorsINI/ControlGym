import os

from SI_Toolkit.computation_library import TensorType
from SI_Toolkit.load_and_normalize import load_yaml
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.cartpole_simulator_batched import cartpole_simulator_batched

config = load_yaml(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), 'r')
angle_weight = float(config["cartpole_simulator_batched"]["default"]["angle_weight"])
position_weight = float(config["cartpole_simulator_batched"]["default"]["position_weight"])


class default(cost_function_base):
    MAX_COST = angle_weight + position_weight * 4 * (cartpole_simulator_batched.x_threshold ** 2)
    
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        angle, angleD, angle_cos, angle_sin, position, positionD = self.lib.unstack(states, 6, -1)
        cost = (
            - angle_weight * angle_cos
            + position_weight * (position - self.variable_parameters.target_position) ** 2
        )
        return cost
