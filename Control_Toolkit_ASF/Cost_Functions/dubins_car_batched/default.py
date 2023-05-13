import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.dubins_car_batched import dubins_car_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
distance_cost_weight = 0.125
x_cost_weight = 0.1
y_cost_weight = 0.1
obstacle_cost_weight = 5.0

class default(cost_function_base):
    MAX_COST = max(distance_cost_weight * (4.0 * x_cost_weight + 4.0 * y_cost_weight + 1.0 * distance_cost_weight), 1.0)
    
    def _distance_to_obstacle_cost(self, x: TensorType, y: TensorType) -> TensorType:
        # x/y each have shape batch_size x mpc_horizon
        x_obs, y_obs, radius = self.lib.unstack(self.controller.obstacle_positions[:, :, self.lib.newaxis, self.lib.newaxis], 3, 1)
        num_obstacles = self.lib.shape(x_obs)[0]
        # Repeat x and y to match the shape of the obstacle map
        x_repeated = self.lib.repeat(self.lib.unsqueeze(x, 0), num_obstacles, 0)
        y_repeated = self.lib.repeat(self.lib.unsqueeze(y, 0), num_obstacles, 0)
        d = self.lib.sqrt(
            (x_repeated - x_obs) ** 2 + (y_repeated - y_obs) ** 2
        )
        c = 1.0 - (self.lib.min(1.0, d / radius)) ** 2
        obstacle_cost = self.lib.reduce_max(c, 0)
        return obstacle_cost
        
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        x, y, yaw_car, steering_rate = self.lib.unstack(states, 4, -1)
        target = self.lib.to_tensor(self.controller.target_point, self.lib.float32)
        x_target, y_target, yaw_target = self.lib.unstack(target, 3, 0)

        head_to_target = dubins_car_batched.get_heading(self.lib, states, self.lib.unsqueeze(target, 0))
        alpha = head_to_target - yaw_car
        ld = dubins_car_batched.get_distance(self.lib, states, self.lib.unsqueeze(target, 0))
        crossTrackError = self.lib.sin(alpha) * ld

        car_in_bounds = dubins_car_batched._car_in_bounds(self.lib, x, y)
        car_at_target = dubins_car_batched._car_at_target(self.lib, x, y, x_target, y_target)

        cost = (
            # self.lib.cast(car_in_bounds & car_at_target, self.lib.float32) * (-10.0)
            self.lib.cast(car_in_bounds & (~car_at_target), self.lib.float32) * (
                distance_cost_weight * (
                    # 3 * crossTrackError**2
                    x_cost_weight * (x - x_target) ** 2
                    + y_cost_weight * (y - y_target) ** 2
                    # + 3 * (head_to_target - yaw_car)**2 / MAX_STEER
                    + obstacle_cost_weight * self._distance_to_obstacle_cost(x, y)
                )
            )
            + self.lib.cast(~car_in_bounds, self.lib.float32)
        )
        return cost
