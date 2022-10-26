import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.dubins_car_batched import dubins_car_batched


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
altitude_weight = float(config["continuous_mountaincar_batched"]["default"]["altitude_weight"])
done_reward = float(config["continuous_mountaincar_batched"]["default"]["done_reward"])
control_penalty = float(config["continuous_mountaincar_batched"]["default"]["control_penalty"])


class default(cost_function_base):
    def _distance_to_obstacle_cost(self, x: TensorType, y: TensorType) -> TensorType:
        costs = self.lib.unsqueeze(self.lib.zeros_like(x), -1)
        for obstacle_position in self.controller.obstacle_positions:
            x_obs, y_obs, radius = obstacle_position
            _d = self.lib.sqrt((x - x_obs) ** 2 + (y - y_obs) ** 2)
            _c = 1.0 - (self.lib.min(1.0, _d / radius)) ** 2
            _c = self.lib.unsqueeze(_c, -1)
            costs = self.lib.concat([costs, _c], -1)
        return self.lib.reduce_max(costs[..., 1:], -1)
        
    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        x, y, yaw_car, steering_rate = self.lib.unstack(states, 4, -1)
        target = self.lib.to_tensor(self.controller.target_point, self.lib.float32)
        x_target, y_target, yaw_target = self.lib.unstack(target, 3, 0)

        head_to_target = dubins_car_batched.get_heading(self.lib, states, self.lib.unsqueeze(target, 0))
        alpha = head_to_target - yaw_car
        ld = dubins_car_batched.get_distance(self.lib, states, self.lib.unsqueeze(target, 0))
        crossTrackError = self.lib.sin(alpha) * ld

        car_in_bounds = dubins_car_batched._car_in_bounds(self.lib, x, y)
        car_at_target = dubins_car_batched._car_at_target(self.lib, x, y, x_target, y_target)

        reward = (
            self.lib.cast(car_in_bounds & car_at_target, self.lib.float32) * 10.0
            + self.lib.cast(car_in_bounds & (~car_at_target), self.lib.float32)
            * (
                -0.125
                * (
                    # 3 * crossTrackError**2
                    0.1 * (x - x_target) ** 2
                    + 0.1 * (y - y_target) ** 2
                    # + 3 * (head_to_target - yaw_car)**2 / MAX_STEER
                    + 5 * self._distance_to_obstacle_cost(x, y)
                )
            )
            + self.lib.cast(~car_in_bounds, self.lib.float32) * (-1.0)
        )
        return -reward

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
