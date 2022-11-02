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

controller_names = config["1_data_generation"]["controller_names"]
if len(controller_names) != 1 or controller_names[0] != "controller_mpc":
    raise ValueError("This script is designed to run when the config.yml has controller_names=[controller_mpc].")

controller_name = controller_names[0]
optimizer_names = config["1_data_generation"]["optimizer_names"]
environment_names = config["1_data_generation"]["environment_names"]


if __name__ == "__main__":
    datetime_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Iterate over all optimizer-environment combinations:
    for optimizer_name, environment_name in product(optimizer_names, environment_names):
        optimizer_short_name = optimizer_name.replace("optimizer_", "").replace("_", "-")
        
        # Iterate over all zipped hyperparameter combinations:
        for sweep_value_tuple in zip(*sweep_values):
            OutputPath.collection_folder_name = os.path.join(f"{datetime_str}_sweep_controller_name", f"controller_name={controller_name}")
            CurrentRunMemory.current_controller_name = controller_name
            CurrentRunMemory.current_optimizer_name = optimizer_name
            CurrentRunMemory.current_environment_name = environment_name
            
            for param_name, param_value in zip(parameters_to_sweep, sweep_value_tuple):
                if param_name not in config_optimizers[optimizer_short_name]:
                    raise ValueError(f"{param_name} is not used in {optimizer_name}")
                # Overwrite with sweep value:
                config_optimizers[optimizer_short_name][param_name] = param_value

            device_name = "/CPU:0"
            if config["1_data_generation"]["use_gpu"]:
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
