"""
This script selects all combinations of MPC optimizers and environments from config.
It runs the controllers for a number of randomized trials using the default parametrization in controller / optimizer config.
You can additionally specify hyperparameters to sweep. Each hyperparameter combination will be run independently.
"""
# 1. Specify the following (can be an empty list)
# Example:
#   parameters_to_sweep = ["learning_rate", "batch_size"]
#   sweep_values = [[0.01, 0.05, 0.2], [1, 5, 10]]
#   -> Creates 3 runs with zipped parameter tuples (0.01, 1), (0.05, 5), (0.2, 10).

parameters_to_sweep = ["num_rollouts"]
sweep_values = [[16, 32]]  # One list per parameter above. All sublists need to have same length
controller_names = ["controller_mpc"]
optimizer_names = [
    "optimizer_cem_tf",
    "optimizer_mppi_tf",
    "optimizer_mppi_var_tf",
    "optimizer_mppi_optimize_tf",
    "optimizer_cem_grad_bharadhwaj_tf",
    "optimizer_gradient_tf",
    "optimizer_random_action_tf",
    "optimizer_rpgd_tf",
    "optimizer_rpgd_me_tf",
    "optimizer_rpgd_ml_tf"
]
environment_names = [
    "MountainCarContinuous-v0",
    "CartPoleSimulator-v0",
    "DubinsCar-v0",
    "Acrobot-v0",
    "Pendulum-v0",
    "CartPoleContinuous-v0",
    "BipedalWalkerBatched-v0",
    "ObstacleAvoidance-v0",
]

### ------------------------------------------------------------------------------------ ###
from datetime import datetime
import os
from itertools import product

import tensorflow as tf
from main import run_data_generator
from yaml import FullLoader, load

from Utilities.utils import CurrentRunMemory, OutputPath, get_logger

logger = get_logger(__name__)


config_controllers = load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"), "r"), Loader=FullLoader)
config_optimizers = load(open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml"), "r"), Loader=FullLoader)
config = load(open("config.yml", "r"), Loader=FullLoader)

if len(controller_names) != 1 or controller_names[0] != "controller_mpc":
    raise ValueError("This script is designed to run when the config.yml has controller_names=[controller_mpc].")
controller_name = controller_names[0]

if __name__ == "__main__":
    datetime_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Iterate over all optimizer-environment combinations:
    for optimizer_name, environment_name in product(optimizer_names, environment_names):
        optimizer_short_name = optimizer_name.replace("optimizer_", "").replace("_", "-")
        
        # Iterate over all zipped hyperparameter combinations:
        for sweep_value_tuple in zip(*sweep_values):
            OutputPath.collection_folder_name = os.path.join(
                f"{datetime_str}_sweep_{','.join(parameters_to_sweep)}",
                f"{datetime_str}_{controller_name}_{optimizer_name}_{environment_name}",
                f"{datetime_str}_{','.join(parameters_to_sweep)}={','.join(list(map(str, sweep_value_tuple)))}",
            )
            CurrentRunMemory.current_controller_name = controller_name
            CurrentRunMemory.current_optimizer_name = optimizer_name
            CurrentRunMemory.current_environment_name = environment_name
            
            for param_name, param_value in zip(parameters_to_sweep, sweep_value_tuple):
                if param_name not in config_optimizers[optimizer_short_name]:
                    raise ValueError(f"{param_name} is not used in {optimizer_name}")
                # Overwrite with sweep value:
                config_optimizers[optimizer_short_name][param_name] = param_value

            device_name = "/CPU:0"
            if config["use_gpu"]:
                if len(tf.config.list_physical_devices("GPU")) > 0:
                    device_name = "/GPU:0"
                else:
                    logger.info(
                        "GPU use specified in config but no device available. Using CPU instead."
                    )

            with tf.device(device_name):
                run_data_generator(
                    CurrentRunMemory.current_controller_name,
                    CurrentRunMemory.current_optimizer_name,
                    CurrentRunMemory.current_environment_name,
                    config
                )
