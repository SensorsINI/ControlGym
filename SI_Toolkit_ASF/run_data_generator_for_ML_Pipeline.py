from Utilities.utils import CurrentRunMemory, get_logger
from main import run_data_generator

# Automatically create new path to save everything in

import tensorflow as tf
import yaml, os
config_SI = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_training.yml')), Loader=yaml.FullLoader)
config_GymEnv = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

logger = get_logger(__name__)

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

if __name__ == '__main__':
    record_path = get_record_path()
    
    controller_names, environment_names, num_experiments = config_GymEnv["1_data_generation"]["controller_names"], config_GymEnv["1_data_generation"]["environment_names"], config_GymEnv["1_data_generation"]["num_experiments"]
    if isinstance(controller_names, list):
        if len(controller_names) > 1:
            logger.warning("Multiple controller names supplied. Only using the first one.")
        controller_name = controller_names[0]
    else:
        controller_name = controller_names
        
    if isinstance(environment_names, list):
        if len(environment_names) > 1:
            logger.warning("Multiple controller names supplied. Only using the first one.")
        environment_name = environment_names[0]
    else:
        environment_name = environment_names
        
    # Save copy of configs in experiment folder
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    yaml.dump(config_SI, open(record_path + "/SI_Toolkit_config_savefile.yml", "w"), default_flow_style=False)
    yaml.dump(config_GymEnv, open(record_path + "/GymEnv_config_savefile.yml", "w"), default_flow_style=False)

    # Run data generator
    CurrentRunMemory.current_controller_name = controller_name
    CurrentRunMemory.current_environment_name = environment_name

    device_name = "/CPU:0"
    if config_GymEnv["1_data_generation"]["use_gpu"]:
        if len(tf.config.list_physical_devices("GPU")) > 0:
            device_name = "/GPU:0"
        else:
            logger.info(
                "GPU use specified in config but no device available. Using CPU instead."
            )

    with tf.device(device_name):
        run_data_generator(controller_name, environment_name, num_experiments, config_GymEnv, run_for_ML_Pipeline=True, record_path=record_path)
    
