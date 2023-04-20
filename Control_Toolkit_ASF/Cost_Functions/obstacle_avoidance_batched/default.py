import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.obstacle_avoidance_batched import obstacle_avoidance_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
distance_to_target_weight = float(
    config["obstacle_avoidance_batched"]["default"]["distance_to_target_weight"]
)
distance_to_obstacle_weight = float(
    config["obstacle_avoidance_batched"]["default"]["distance_to_obstacle_weight"]
)
goal_reward = float(
    config["obstacle_avoidance_batched"]["default"]["goal_reward"]
)
out_of_bounds_cost = float(
    config["obstacle_avoidance_batched"]["default"]["out_of_bounds_cost"]
)

class default(cost_function_base):
    MAX_COST = max(12.0 * distance_to_target_weight + 1.0 * distance_to_obstacle_weight, out_of_bounds_cost)
    
    def _distance_to_obstacle_cost(self, x: TensorType, y: TensorType, z: TensorType) -> TensorType:
        # x/y/z each has shape batch_size x mpc_horizon
        x_obs, y_obs, z_obs, radius = self.lib.unstack(self.variable_parameters.obstacle_positions, 4, -1)
        x_obs = x_obs[:, self.lib.newaxis, self.lib.newaxis]
        y_obs = y_obs[:, self.lib.newaxis, self.lib.newaxis]
        z_obs = z_obs[:, self.lib.newaxis, self.lib.newaxis]
        radius = radius[:, self.lib.newaxis, self.lib.newaxis]
        
        num_obstacles = self.lib.shape(x_obs)[0]
        d = self.lib.sqrt(
            (self.lib.repeat(x[self.lib.newaxis, ...], num_obstacles, 0) - x_obs) ** 2
            + (self.lib.repeat(y[self.lib.newaxis, ...], num_obstacles, 0) - y_obs) ** 2
            + (self.lib.repeat(z[self.lib.newaxis, ...], num_obstacles, 0) - z_obs) ** 2
        )
        c = 1.0 - (self.lib.min(1.0, d / radius)) ** 2
        return self.lib.reduce_max(c, 0)

    def _get_distance(self, x1, x2):
        # Squared distance between points x1 and x2
        return (x1 - x2) ** 2

    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        target = self.lib.to_tensor(self.variable_parameters.target_point, self.lib.float32)
        pos_x, pos_y, pos_z, _, _, _ = self.lib.unstack(states, 6, -1)

        ld = (
            self._get_distance(pos_x, target[0])
            + self._get_distance(pos_y, target[1])
            + self._get_distance(pos_z, target[2])
        )

        car_in_bounds = obstacle_avoidance_batched._in_bounds(self.lib, pos_x, pos_y, pos_z)
        car_at_target = obstacle_avoidance_batched._at_target(self.lib, pos_x, pos_y, pos_z, target)

        cost = (
            - goal_reward * self.lib.sum(self.lib.cast(car_in_bounds & car_at_target, self.lib.float32), 1)[:, self.lib.newaxis]  # Sum number of horizon states at target -> optimizer will try to get as many horizon states into target area
            + self.lib.cast(car_in_bounds & (~car_at_target), self.lib.float32) * (
                distance_to_target_weight * ld
                + distance_to_obstacle_weight * self._distance_to_obstacle_cost(pos_x, pos_y, pos_z)
            )
            + out_of_bounds_cost * self.lib.cast(~car_in_bounds, self.lib.float32)
        )

        return cost
