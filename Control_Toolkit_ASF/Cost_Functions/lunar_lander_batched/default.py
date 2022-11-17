import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.lunar_lander_batched import lunar_lander_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
pos_weight = float(config["lunar_lander_batched"]["default"]["pos_weight"])
vel_y_weight = float(config["lunar_lander_batched"]["default"]["vel_y_weight"])
vel_angle_weight = float(config["lunar_lander_batched"]["default"]["vel_angle_weight"])


class default(cost_function_base):
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact = self.lib.unstack(states, 7, -1)
        throttle_main, throttle_lr = self.lib.unstack(inputs, 2, -1)
        target_point = self.lib.to_tensor(self.controller.target_point, self.lib.float32)
        terminated_successfully = self.lib.cast(lunar_lander_batched.is_done(self.lib, states, target_point), self.lib.float32)
        
        cost = (
            pos_weight * (
                (pos_x - target_point[0, 0]) ** 2
                + (pos_y - target_point[0, 1]) ** 2
            )
            + vel_y_weight * (vel_y ** 2)
            + vel_angle_weight * (vel_angle ** 2)
            + contact * (1.0 - terminated_successfully) * 100.0
        )
        return cost

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact = self.lib.unstack(terminal_states, 7, -1)
        target_point = self.lib.to_tensor(self.controller.target_point, self.lib.float32)
        terminated_successfully = self.lib.cast(lunar_lander_batched.is_done(self.lib, terminal_states, target_point), self.lib.float32)
        return (
            - terminated_successfully * 100.0
        )
