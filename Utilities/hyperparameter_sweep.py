from datetime import datetime
import os

import tensorflow as tf
from main import run_data_generator
from yaml import FullLoader, load

from Utilities.utils import CurrentRunMemory, OutputPath, get_logger

CONTROLLER_TO_ANALYZE = "controller_dist_adam_resamp2_tf"
# PARAMETERS_TO_SWEEP = ["num_rollouts", "opt_keep_k"]
# SWEEP_VALUES = [[32, 32], [0, 32]]
# SWEEP_VALUES = [[1, 2, 4, 8, 16, 32, 64, 128, 256, 0], [0, 1, 1, 2, 4, 8, 16, 32, 64, 0]]
# SWEEP_VALUES = [[128], [32]]
# SWEEP_VALUES = [[16, 32, 64], [4, 8, 16]]
PARAMETERS_TO_SWEEP = ["interpolation_step"]
SWEEP_VALUES = [[1, 5, 10, 20, 50, 200]]

config = load(open("config.yml", "r"), Loader=FullLoader)
CONTROLLER_NAMES, ENVIRONMENT_NAMES, NUM_EXPERIMENTS = (
    config["1_data_generation"]["controller_names"],
    config["1_data_generation"]["environment_names"],
    config["1_data_generation"]["num_experiments"],
)
CONTROLLER_NAMES = (
    [CONTROLLER_NAMES] if isinstance(CONTROLLER_NAMES, str) else CONTROLLER_NAMES
)
ENVIRONMENT_NAMES = (
    [ENVIRONMENT_NAMES] if isinstance(ENVIRONMENT_NAMES, str) else ENVIRONMENT_NAMES
)
assert len(ENVIRONMENT_NAMES) == 1, "Can only sweep over one environment"

logger = get_logger(__name__)

if __name__ == "__main__":
    datetime_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    for sweep_value in zip(*SWEEP_VALUES):
        OutputPath.collection_folder_name = os.path.join(f"{datetime_str}_sweep_{','.join(PARAMETERS_TO_SWEEP)}_{CONTROLLER_TO_ANALYZE}", f"{','.join(PARAMETERS_TO_SWEEP)}={','.join(list(map(str, sweep_value)))}")
        CurrentRunMemory.current_controller_name = CONTROLLER_TO_ANALYZE
        CurrentRunMemory.current_environment_name = ENVIRONMENT_NAMES[0]

        for param_name, param_value in zip(PARAMETERS_TO_SWEEP, sweep_value):
            if param_name not in config["4_controllers"][CONTROLLER_TO_ANALYZE]:
                raise ValueError(f"{param_name} is not used in {CONTROLLER_TO_ANALYZE}")
            config["4_controllers"][CONTROLLER_TO_ANALYZE][param_name] = param_value

        device_name = "/CPU:0"
        if config["1_data_generation"]["use_gpu"]:
            if len(tf.config.list_physical_devices("GPU")) > 0:
                device_name = "/GPU:0"
            else:
                logger.info(
                    "GPU use specified in config but no device available. Using CPU instead."
                )

        with tf.device(device_name):
            run_data_generator(CurrentRunMemory.current_controller_name, CurrentRunMemory.current_environment_name, NUM_EXPERIMENTS, config)
