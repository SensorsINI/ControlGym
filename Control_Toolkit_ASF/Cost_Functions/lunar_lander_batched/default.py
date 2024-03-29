import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.lunar_lander_batched import lunar_lander_batched, GroundContactDetector


config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
pos_x_weight = float(config["lunar_lander_batched"]["default"]["pos_x_weight"])
pos_y_weight = float(config["lunar_lander_batched"]["default"]["pos_y_weight"])
vel_weight = float(config["lunar_lander_batched"]["default"]["vel_weight"])
angle_weight = float(config["lunar_lander_batched"]["default"]["angle_weight"])
vel_angle_weight = float(config["lunar_lander_batched"]["default"]["vel_angle_weight"])
discount_factor = float(config["lunar_lander_batched"]["default"]["discount_factor"])
ground_cost_weight = 1000.0
out_of_bounds_cost = 100.0


class default(cost_function_base):
    MAX_COST = (
        4.0 * pos_x_weight
        + (4.0 + ground_cost_weight) * pos_y_weight
        + 50.0 * vel_weight
        + angle_weight
        + 10.0 * vel_angle_weight
        + out_of_bounds_cost * 2.0
    )
    
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact = self.lib.unstack(states, 7, -1)
        throttle_main, throttle_lr = self.lib.unstack(inputs, 2, -1)
        target_point = self.lib.to_tensor(self.variable_parameters.target_point, self.lib.float32)
        ground_contact_detector: GroundContactDetector = self.variable_parameters.ground_contact_detector
        terminated_successfully = self.lib.cast(lunar_lander_batched.is_done(self.lib, states, target_point), self.lib.float32)
        
        cost = (
            pos_x_weight * ((pos_x - target_point[0, 0]) ** 2)
            + pos_y_weight * (
                ((pos_y - target_point[0, 1]) ** 2)
                + ground_cost_weight * self.lib.clip((ground_contact_detector.surface_y_at_point(pos_x) + 0.02 - pos_y), 0.0, 1.0) ** 2
            )
            + vel_weight * (
                vel_x ** 2
                + vel_y ** 2
            )
            + angle_weight * (self.lib.sin(angle) ** 2)
            + vel_angle_weight * (vel_angle ** 2)
            + out_of_bounds_cost * self.lib.clip(self.lib.clip(10.0 * (self.lib.abs(pos_x) - 0.9), 0.0, 1.0) ** 2, 0.0, 1.0)  # Out of bounds
            + out_of_bounds_cost * self.lib.clip(self.lib.clip(10.0 * (self.lib.abs(pos_y) - 0.9), 0.0, 1.0) ** 2, 0.0, 1.0)  # Out of bounds
            - terminated_successfully * 1e6
        )
        return cost

    def get_terminal_cost(self, terminal_states: TensorType) -> TensorType:
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact = self.lib.unstack(terminal_states, 7, -1)
        target_point = self.lib.to_tensor(self.variable_parameters.target_point, self.lib.float32)
        terminated_successfully = self.lib.cast(lunar_lander_batched.is_done(self.lib, terminal_states, target_point), self.lib.float32)
        return (
            (-100.0) * contact * terminated_successfully
        )

    def get_trajectory_cost(self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None) -> TensorType:
        stage_costs = self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input)  # Select all but last state of the horizon
        gamma = discount_factor * self.lib.ones_like(stage_costs)
        gamma = self.lib.cumprod(gamma, 1)

        total_cost = self.lib.sum(gamma * stage_costs, 1)  # Sum across the MPC horizon dimension
        total_cost = total_cost + self.get_terminal_cost(state_horizon[:, -1, :])
        return total_cost