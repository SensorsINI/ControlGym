from datetime import datetime
import numpy as np
from yaml import load, FullLoader
import os

from Utilities.utils import get_logger
from Visualizations.plot_box_plots import BoxPlotPlotter

logger = get_logger(__name__)

# Select a number of experiments

PATH_TO_EXPERIMENTS = "Output"
EXPERIMENTS_TO_PLOT = [
    "20220724-184558_controller_dist_adam_resamp2_CustomEnvironments_CartPoleContinuous_predictor_ODE_tf",
    "20220724-185337_controller_dist_adam_resamp2_CustomEnvironments_CartPoleContinuous_predictor_ODE_tf",
    "20220724-185649_controller_dist_adam_resamp2_CustomEnvironments_CartPoleContinuous_predictor_ODE_tf",
]
# Compare configs associated with the different experiments

# Generate box plot: Each box represents the total cost statistics for a specific controller

def generate_global_plots():
    all_average_cost_data = {}
    
    # Prepare average cost data
    for exp in EXPERIMENTS_TO_PLOT:
        all_average_cost_data[exp] = []
        
        path_to_experiment = os.path.join(PATH_TO_EXPERIMENTS, exp)
        config_filepaths = filter(lambda x: x.endswith(".yml"), os.listdir(path_to_experiment))
        config_filepaths = list(map(lambda x: os.path.join(path_to_experiment, x), config_filepaths))
        config = load(open(config_filepaths[0], "r"), FullLoader)
        
        cost_filepaths = filter(lambda x: "J_logged" in x and x.endswith(".npy"), os.listdir(path_to_experiment))
        cost_filepaths = map(lambda x: os.path.join(path_to_experiment, x), cost_filepaths)
        for cost_filepath in cost_filepaths:
            cost_log: np.ndarray = np.load(cost_filepath)
            mean_cost = float(np.squeeze(cost_log.mean()))
            all_average_cost_data[exp].append(mean_cost)
            
    logger.info("Generating box plot...")
    box_plot_plotter = BoxPlotPlotter(datetime.now().strftime("%Y%m%d-%H%M%S"), {
        "1_data_generation": {"controller_name": "", "environment_name": ""},
        "2_environments": {"": {"actuator_noise": 0}}
    })
    box_plot_plotter.plot(all_average_cost_data, True)
    logger.info("...done.")
    

if __name__ == "__main__":
    generate_global_plots()
    
