import os
from SI_Toolkit.computation_library import TensorType

import yaml
from Control_Toolkit.Cost_Functions import cost_function_base
from Control_Toolkit.others.environment import EnvironmentBatched

config = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"), Loader=yaml.FullLoader)
config_default = config["GymEnvironment"]["default"]

class default(cost_function_base):
    """Uses as cost function the get_reward method of environment provided."""

    def __init__(self, env) -> None:
        self.env: EnvironmentBatched = env

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        return -self.env.get_reward(states, inputs)

    def get_trajectory_cost(self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None) -> TensorType:
        return self.env.lib.sum(
            self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input), 1
        ) + self.get_terminal_cost(state_horizon[:, -1, :])
