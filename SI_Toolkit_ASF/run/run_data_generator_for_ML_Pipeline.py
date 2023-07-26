from Utilities.utils import ConfigManager, CurrentRunMemory, get_logger, nested_assignment_to_ordereddict
from main import run_data_generator

import tensorflow as tf
import os
import ruamel.yaml

logger = get_logger(__name__)

if __name__ == '__main__':

    # Create a config manager which looks for '.yml' files within the list of folders specified.
    # Rationale: We want GUILD AI to be able to update values in configs that we include in this list.
    # We might intentionally want to exclude the path to a folder which does contain configs but should not be overwritten by GUILD.
    config_manager = ConfigManager(".", "Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments")

    # Scan for any custom parameters that should overwrite the toolkits' config files:
    submodule_configs = ConfigManager("Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments").loaders
    for base_name, loader in submodule_configs.items():
        if base_name in config_manager("config").get("custom_config_overwrites", {}):
            data: ruamel.yaml.comments.CommentedMap = loader.load()
            update_dict = config_manager("config")["custom_config_overwrites"][base_name]
            nested_assignment_to_ordereddict(data, update_dict)
            loader.overwrite_config(data)

    # Retrieve required parameters from config:
    CurrentRunMemory.current_controller_name = config_manager("config")["controller_name"]
    CurrentRunMemory.current_environment_name = config_manager("config")["environment_name"]
    config_SI = dict(config_manager("config_training"))


    def get_record_path():
        experiment_index = 1
        while True:
            record_path = f"Experiment-{experiment_index}"
            if os.path.exists(config_SI['paths']['PATH_TO_EXPERIMENT_FOLDERS'] + record_path):
                experiment_index += 1
            else:
                record_path = os.path.join(record_path, "Recordings")
                break

        record_path = config_SI['paths']['PATH_TO_EXPERIMENT_FOLDERS'] + record_path
        return record_path


    record_path = get_record_path()

    device_name = "/CPU:0"
    if config_manager("config")["use_gpu"]:
        if len(tf.config.list_physical_devices("GPU")) > 0:
            device_name = "/GPU:0"
        else:
            logger.info(
                "GPU use specified in config but no device available. Using CPU instead."
            )

    with tf.device(device_name):
        run_data_generator(controller_name=CurrentRunMemory.current_controller_name,
                           environment_name=CurrentRunMemory.current_environment_name,
                           config_manager=config_manager,
                           run_for_ML_Pipeline=True,
                           record_path=record_path)

