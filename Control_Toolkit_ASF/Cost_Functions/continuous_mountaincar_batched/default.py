import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.cartpole_simulator_batched import cartpole_simulator_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
altitude_weight = float(config["continuous_mountaincar_batched"]["default"]["altitude_weight"])
done_reward = float(config["continuous_mountaincar_batched"]["default"]["done_reward"])
control_penalty = float(config["continuous_mountaincar_batched"]["default"]["control_penalty"])


class default(cost_function_base):
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        position, velocity = self.lib.unstack(states, 2, -1)
        force = inputs[..., 0]
        
        cost = (
            - altitude_weight * self.lib.sin(3 * position)
            - done_reward * self.lib.cast(cartpole_simulator_batched.is_done(self.lib, states), self.lib.float32)  # This part is not differentiable
            + control_penalty * (force**2)
        )
        return cost

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
