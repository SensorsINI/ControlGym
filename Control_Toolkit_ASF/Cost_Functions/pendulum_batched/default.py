import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
altitude_weight = float(
    config["continuous_mountaincar_batched"]["default"]["altitude_weight"]
)
done_reward = float(config["continuous_mountaincar_batched"]["default"]["done_reward"])
control_penalty = float(
    config["continuous_mountaincar_batched"]["default"]["control_penalty"]
)


class default(cost_function_base):
    def _angle_normalize(self, x):
        _pi = self.lib.to_tensor(self.lib.pi, self.lib.float32)
        return ((x + _pi) % (2 * _pi)) - _pi
    
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        th, thdot, sinth, costh = self.lib.unstack(states, 4, -1)
        costs = (
            self._angle_normalize(th) ** 2
            + 0.1 * thdot**2
            + 0.001 * (inputs[:, 0] ** 2)
        )
        return costs

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
