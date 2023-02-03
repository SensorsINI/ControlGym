import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.acrobot_batched import acrobot_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
angle1_weight = float(config["acrobot_batched"]["discounted_horizon"]["angle1_weight"])
angle2_weight = float(config["acrobot_batched"]["discounted_horizon"]["angle2_weight"])
angleD1_weight = float(config["acrobot_batched"]["discounted_horizon"]["angleD1_weight"])
angleD2_weight = float(config["acrobot_batched"]["discounted_horizon"]["angleD2_weight"])
discount_factor = float(config["acrobot_batched"]["discounted_horizon"]["discount_factor"])


class discounted_horizon(cost_function_base):
    MAX_COST = angle1_weight + angle2_weight + angleD1_weight * (acrobot_batched.MAX_VEL_1 ** 2) + angleD2_weight * (acrobot_batched.MAX_VEL_2 ** 2)
    
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        th1, th2, th1_vel, th2_vel = self.lib.unstack(states, 4, -1)
        cost = (
            angle1_weight * self.lib.cos(th1)
            + angle2_weight * self.lib.cos(th2 + th1)
            + angleD1_weight * th1_vel ** 2
            + angleD2_weight * th2_vel ** 2
        )
        return cost
    
    def get_trajectory_cost(self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None) -> TensorType:
        stage_costs = self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input)  # Select all but last state of the horizon
        gamma = discount_factor * self.lib.ones_like(stage_costs)
        gamma = self.lib.cumprod(gamma, 1)

        terminal_costs = self.get_terminal_cost(state_horizon[:, -1, :])
        total_cost = self.lib.mean(self.lib.concat([gamma * stage_costs, terminal_costs], 1), 1)  # Mean across the MPC horizon dimension
        return total_cost
