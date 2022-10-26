import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.obstacle_avoidance_batched import obstacle_avoidance_batched


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
    def _distance_to_obstacle_cost(self, x: TensorType) -> TensorType:
        costs = None
        for obstacle_position in self.controller.obstacle_positions:
            _d = self.lib.sqrt(self.lib.sum((x - obstacle_position[:-1]) ** 2, -1))
            _c = 1.0 - (self.lib.min(1.0, _d / obstacle_position[-1])) ** 2
            _c = self.lib.unsqueeze(_c, -1)
            costs = _c if costs == None else self.lib.concat([costs, _c], -1)
        return self.lib.reduce_max(costs, -1)

    def get_distance(self, x1, x2):
        # Distance between points x1 and x2
        return self.lib.sqrt(self.lib.sum((x1 - x2) ** 2, -1))

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        target = self.lib.to_tensor(self.controller.target_point, self.lib.float32)
        num_dimensions = int(self.lib.shape(states)[-1] / 2)
        position = states[..., :num_dimensions]

        ld = self.get_distance(position, self.lib.unsqueeze(target, 0))

        car_in_bounds = obstacle_avoidance_batched._in_bounds(self.lib, position)
        car_at_target = obstacle_avoidance_batched._at_target(self.lib, position, target)

        reward = (
            self.lib.cast(car_in_bounds & car_at_target, self.lib.float32) * 10.0
            + self.lib.cast(car_in_bounds & (~car_at_target), self.lib.float32)
            * (-1.0 * (ld + 4 * self._distance_to_obstacle_cost(position)))
            + self.lib.cast(~car_in_bounds, self.lib.float32) * (-10.0)
        )

        return -reward

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
