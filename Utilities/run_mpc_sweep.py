"""
This script selects all combinations of MPC optimizers and environments from config.
It runs the controllers for a number of randomized trials using the default parametrization in controller / optimizer config.
You can additionally specify hyperparameters to sweep. Each hyperparameter combination will be run independently.
"""
# 1. Specify the following (can be an empty list)
# Example:
#   parameters_to_sweep = ["config_optimizers.rpgd-tf.learning_rate", "config_optimizers.rpgd-tf.batch_size"]
#   sweep_values = [[0.01, 0.05, 0.2], [1, 5, 10]]
#   -> Creates 3 runs with zipped parameter tuples (0.01, 1), (0.05, 5), (0.2, 10).
#   -> Hint: Make sure that the optimizer you sweep over is also set either here or directly in config

parameters_to_sweep = [
    "config_controllers.mpc.optimizer",  # syntax: <<config_name>>.<<config_entry>>.<<parameter_name>>
]
sweep_values = [
    ["rpgd-tf", "rpgd-me-tf"],
]  # One list per parameter above. All sublists need to have same length
controller_name = "controller_mpc"

### ------------------------------------------------------------------------------------ ###
from datetime import datetime
import os
from itertools import product

import tensorflow as tf
from main import run_data_generator
import ruamel.yaml

from Utilities.utils import ConfigManager, CurrentRunMemory, OutputPath, get_logger, nested_assignment_to_ordereddict

logger = get_logger(__name__)


config_manager = ConfigManager(".", "Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments")
environment_name = config_manager("config")["environment_name"]

if controller_name != "controller_mpc":
    raise ValueError("This script is designed to run when the config.yml has controller_names=[controller_mpc].")

if __name__ == "__main__":
    datetime_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Iterate over all zipped hyperparameter combinations:
    for sweep_value_tuple in zip(*sweep_values):
        OutputPath.collection_folder_name = os.path.join(
            f"{datetime_str}_sweep_{','.join(parameters_to_sweep)}",
            f"{datetime_str}_{controller_name}_{environment_name}",
            f"{datetime_str}_{','.join(parameters_to_sweep)}={','.join(list(map(str, sweep_value_tuple)))}",
        )
        CurrentRunMemory.current_controller_name = controller_name
        CurrentRunMemory.current_environment_name = environment_name
        
        for param_desc, param_value in zip(parameters_to_sweep, sweep_value_tuple):
            config_name, config_entry, param_name = param_desc.split(".")
            if param_name not in config_manager(config_name)[config_entry]:
                raise ValueError(f"{param_name} is not used in {config_name}")
            # Overwrite with sweep value:
            loader = config_manager.loaders[config_name]
            data: ruamel.yaml.comments.CommentedMap = loader.load()
            nested_assignment_to_ordereddict(data, {config_entry: {param_name: param_value}})
            loader.overwrite_config(data)

        device_name = "/CPU:0"
        if config_manager("config")["use_gpu"]:
            if len(tf.config.list_physical_devices("GPU")) > 0:
                device_name = "/GPU:0"
            else:
                logger.info(
                    "GPU use specified in config but no device available. Using CPU instead."
                )

        with tf.device(device_name):
            run_data_generator(
                CurrentRunMemory.current_controller_name,
                CurrentRunMemory.current_environment_name,
                config_manager,
            )
