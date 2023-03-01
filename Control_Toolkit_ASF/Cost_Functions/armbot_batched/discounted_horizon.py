import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.armbot_batched import armbot_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
discount_factor = float(config["armbot_batched"]["discounted_horizon"]["discount_factor"])
xtarget = armbot_batched.xtarget
ytarget = armbot_batched.ytarget

class discounted_horizon(cost_function_base):
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        tuple2 = self.lib.unstack(states, armbot_batched.num_states, -1)
        theta = tuple2[0]
        xee = tf.cos(theta)
        yee = tf.cos(theta)
        for i in range(armbot_batched.num_states):
            if i > 0:
                theta += tuple2[i]
                xee += tf.cos(theta)
                yee += tf.sin(theta)
        cost = (
                (xee - xtarget) ** 2 + (yee - ytarget) ** 2
        )
        return cost
    #discounted cost adapted from existing discount horizon implementation
    def get_trajectory_cost(self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None) -> TensorType:
        stage_costs = self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input)  # Select all but last state of the horizon
        gamma = discount_factor * self.lib.ones_like(stage_costs)
        gamma = self.lib.cumprod(gamma, 1)

        terminal_costs = self.get_terminal_cost(state_horizon[:, -1, :])
        total_cost = self.lib.mean(self.lib.concat([gamma * stage_costs, terminal_costs], 1), 1)  # Mean across the MPC horizon dimension
        return total_cost
