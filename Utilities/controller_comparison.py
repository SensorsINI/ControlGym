from datetime import datetime
import os
from itertools import product

import tensorflow as tf
from main import run_data_generator
from yaml import FullLoader, load

from Utilities.utils import CurrentRunMemory, OutputPath, get_logger

logger = get_logger(__name__)

# Load config
CONFIG = load(open("config.yml", "r"), Loader=FullLoader)
CONTROLLER_NAMES, ENVIRONMENT_NAMES, NUM_EXPERIMENTS = (
    CONFIG["1_data_generation"]["controller_names"],
    CONFIG["1_data_generation"]["environment_names"],
    CONFIG["1_data_generation"]["num_experiments"],
)
CONTROLLER_NAMES = (
    [CONTROLLER_NAMES] if isinstance(CONTROLLER_NAMES, str) else CONTROLLER_NAMES
)
ENVIRONMENT_NAMES = (
    [ENVIRONMENT_NAMES] if isinstance(ENVIRONMENT_NAMES, str) else ENVIRONMENT_NAMES
)

if __name__ == "__main__":
    for controller_name, environment_name in product(
        CONTROLLER_NAMES, ENVIRONMENT_NAMES
    ):
        OutputPath.collection_folder_name = os.path.join(f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_sweep_controllers", f"controller_name={controller_name}")
        CurrentRunMemory.current_controller_name = controller_name
        CurrentRunMemory.current_environment_name = environment_name

        device_name = "/CPU:0"
        if CONFIG["1_data_generation"]["use_gpu"]:
            if len(tf.config.list_physical_devices("GPU")) > 0:
                device_name = "/GPU:0"
            else:
                logger.info(
                    "GPU use specified in config but no device available. Using CPU instead."
                )

        with tf.device(device_name):
            run_data_generator(
                controller_name, environment_name, NUM_EXPERIMENTS, CONFIG
            )
