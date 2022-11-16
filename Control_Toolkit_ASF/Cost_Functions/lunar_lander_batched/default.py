import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.lunar_lander_batched import lunar_lander_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
pos_y_weight = float(config["lunar_lander_batched"]["default"]["pos_y_weight"])
vel_y_weight = float(config["lunar_lander_batched"]["default"]["vel_y_weight"])


class default(cost_function_base):
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact_1, contact_2 = self.lib.unstack(states, 8, -1)
        throttle_main, throttle_lr = self.lib.unstack(inputs, 2, -1)
        
        cost = (
            + pos_y_weight * pos_y
            + vel_y_weight * (vel_y ** 2)
        )
        return cost

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
