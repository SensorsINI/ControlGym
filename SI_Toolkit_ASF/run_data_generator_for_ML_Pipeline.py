from main import run_data_generator

# Automatically create new path to save everything in

import yaml, os
config_SI = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_training.yml')), Loader=yaml.FullLoader)
config_GymEnv = yaml.load(open('config.yml'), Loader=yaml.FullLoader)


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

    # Save copy of configs in experiment folder
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    yaml.dump(config_SI, open(record_path + "/SI_Toolkit_config_savefile.yml", "w"), default_flow_style=False)
    yaml.dump(config_GymEnv, open(record_path + "/GymEnv_config_savefile.yml", "w"), default_flow_style=False)

    # Run data generator
    run_data_generator(run_for_ML_Pipeline=True, record_path=record_path)
    
