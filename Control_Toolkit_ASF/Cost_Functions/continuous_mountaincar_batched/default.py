import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.continuous_mountaincar_batched import continuous_mountaincar_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
altitude_weight = float(config["continuous_mountaincar_batched"]["default"]["altitude_weight"])
done_reward = float(config["continuous_mountaincar_batched"]["default"]["done_reward"])
control_penalty = float(config["continuous_mountaincar_batched"]["default"]["control_penalty"])


class default(cost_function_base):
    MAX_COST = altitude_weight + control_penalty
    
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        position, velocity = self.lib.unstack(states, 2, -1)
        force = inputs[..., 0]
        goal_position = self.variable_parameters.goal_position
        goal_velocity = self.variable_parameters.goal_velocity
        
        cost = (
            - altitude_weight * self.lib.sin(3 * position)
            - done_reward * self.lib.cast(continuous_mountaincar_batched.is_done(self.lib, states, goal_position, goal_velocity), self.lib.float32)  # This part is not differentiable
            + control_penalty * (force**2)
        )
        return cost
