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
    def _distance_to_obstacle_cost(self, x: TensorType, y: TensorType, z: TensorType) -> TensorType:
        # x/y/z each has shape batch_size x mpc_horizon
        x_obs, y_obs, z_obs, radius = self.lib.unstack(self.controller.obstacle_positions, 4, -1)
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

    def get_distance(self, x1, x2):
        # Squared distance between points x1 and x2
        return (x1 - x2) ** 2

    def get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        target = self.lib.to_tensor(self.controller.target_point, self.lib.float32)
        pos_x, pos_y, pos_z, _, _, _ = self.lib.unstack(states, 6, -1)

        ld = (
            self.get_distance(pos_x, target[0])
            + self.get_distance(pos_y, target[1])
            + self.get_distance(pos_z, target[2])
        )

        car_in_bounds = obstacle_avoidance_batched._in_bounds(self.lib, pos_x, pos_y, pos_z)
        car_at_target = obstacle_avoidance_batched._at_target(self.lib, pos_x, pos_y, pos_z, target)

        reward = (
            - ld
            + self.lib.cast(car_in_bounds & car_at_target, self.lib.float32) * 10.0
            + self.lib.cast(car_in_bounds & (~car_at_target), self.lib.float32)
            * (-1.0 * (ld + 4 * self._distance_to_obstacle_cost(pos_x, pos_y, pos_z)))
            + self.lib.cast(~car_in_bounds, self.lib.float32) * (-1e4)
        )

        return -reward

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        return 0.0
