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
    def _distance_to_obstacle_cost(self, x: TensorType, num_dimensions: int) -> TensorType:
        # x has shape batch_size x mpc_horizon x num_dimensions
        x_obs = self.controller.obstacle_positions[:, :num_dimensions]  # [num_obstacles, num_dimensions]
        x_obs = x_obs[:, self.lib.newaxis, self.lib.newaxis, :]
        radius = self.controller.obstacle_positions[:, -1:]  # [num_obstacles, 1]
        radius = radius[:, self.lib.newaxis, :]
        
        num_obstacles = self.lib.shape(x_obs)[0]
        # Repeat x and y to match the shape of the obstacle map
        x_repeated = self.lib.repeat(self.lib.unsqueeze(x, 0), num_obstacles, 0)
        d = self.lib.sqrt(self.lib.sum(
            (x_repeated - x_obs) ** 2, 3
        ))
        c = 1.0 - (self.lib.min(1.0, d / radius)) ** 2
        return self.lib.reduce_max(c, 0)

    def get_distance(self, x1, x2):
        # Distance between points x1 and x2
        return self.lib.sqrt(self.lib.sum((x1 - x2) ** 2, -1))

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        target = self.lib.to_tensor(self.controller.target_point, self.lib.float32)
        num_dimensions = self.controller.num_dimensions
        position = states[..., :num_dimensions]

        ld = self.get_distance(position, self.lib.unsqueeze(target, 0))

        car_in_bounds = obstacle_avoidance_batched._in_bounds(self.lib, position)
        car_at_target = obstacle_avoidance_batched._at_target(self.lib, position, target)

        reward = (
            - ld
            + self.lib.cast(car_in_bounds & car_at_target, self.lib.float32) * 10.0
            + self.lib.cast(car_in_bounds & (~car_at_target), self.lib.float32)
            * (-1.0 * (ld + 4 * self._distance_to_obstacle_cost(position, num_dimensions)))
            + self.lib.cast(~car_in_bounds, self.lib.float32) * (-10.0)
        )

        return -reward

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
