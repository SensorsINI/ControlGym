import os

import tensorflow as tf
from main import run_data_generator
from yaml import FullLoader, load

from Utilities.utils import CurrentRunMemory, OutputPath, get_logger

CONTROLLER_TO_ANALYZE = "controller_dist_adam_resamp2_tf"
PARAMETER_TO_SWEEP = "outer_its"
SWEEP_VALUES = [0, 1, 5, 10, 20]

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
    for sweep_value in SWEEP_VALUES:
        OutputPath.collection_folder_name = os.path.join(f"sweep_{PARAMETER_TO_SWEEP}_{CONTROLLER_TO_ANALYZE}", f"{PARAMETER_TO_SWEEP}={sweep_value}")
        CurrentRunMemory.current_controller_name = CONTROLLER_TO_ANALYZE
        CurrentRunMemory.current_environment_name = ENVIRONMENT_NAMES[0]

        if PARAMETER_TO_SWEEP not in config["4_controllers"][CONTROLLER_TO_ANALYZE]:
            raise ValueError(f"{PARAMETER_TO_SWEEP} is not used in {CONTROLLER_TO_ANALYZE}")
        config["4_controllers"][CONTROLLER_TO_ANALYZE][PARAMETER_TO_SWEEP] = sweep_value

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
