import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.acrobot_batched import acrobot_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
angle_weight = float(config["acrobot_batched"]["default"]["angle_weight"])
angleD_weight = float(config["acrobot_batched"]["default"]["angleD_weight"])

class default(cost_function_base):
    MAX_COST = 2 * angle_weight + angleD_weight * (acrobot_batched.MAX_VEL_1 ** 2 + acrobot_batched.MAX_VEL_2 ** 2)
    
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        th1, th2, th1_vel, th2_vel = self.lib.unstack(states, 4, -1)
        cost = (
            angle_weight * (self.lib.cos(th1) + self.lib.cos(th2 + th1))
            + angleD_weight * (th1_vel ** 2 + th2_vel ** 2)
        )
        return cost
