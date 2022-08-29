from datetime import datetime
from glob import glob
import numpy as np
from yaml import load, FullLoader
import os
from natsort import natsorted

from Utilities.utils import get_logger
from Visualizations.plot_cost_scatter_plots import CostScatterPlotPlotter
from pprint import pformat
import re

logger = get_logger(__name__)

# Select a number of experiments

## Option 1: Specify paths manually
# EXPERIMENTS_TO_PLOT = [
#     "Output/sweep_outer_its_controller_dist_adam_resamp2_tf/outer_its=0/20220824-113202_controller_dist_adam_resamp2_tf_CustomEnvironments_MountainCarContinuous-v0_predictor_ODE_tf",
#     "Output/sweep_outer_its_controller_dist_adam_resamp2_tf/outer_its=1/20220824-113445_controller_dist_adam_resamp2_tf_CustomEnvironments_MountainCarContinuous-v0_predictor_ODE_tf",
#     "Output/sweep_outer_its_controller_dist_adam_resamp2_tf/outer_its=5/20220824-113940_controller_dist_adam_resamp2_tf_CustomEnvironments_MountainCarContinuous-v0_predictor_ODE_tf",
#     "Output/sweep_outer_its_controller_dist_adam_resamp2_tf/outer_its=10/20220824-114927_controller_dist_adam_resamp2_tf_CustomEnvironments_MountainCarContinuous-v0_predictor_ODE_tf",
#     "Output/sweep_outer_its_controller_dist_adam_resamp2_tf/outer_its=20/20220824-120515_controller_dist_adam_resamp2_tf_CustomEnvironments_MountainCarContinuous-v0_predictor_ODE_tf",
# ]

## Option 2: Specify a top-level folder
EXPERIMENT_FOLDER = "20220828-185348_sweep_resamp_per_controller_dist_adam_resamp2_tf"
ENVIRONMENT_NAME = "CartPoleSimulator"
EXPERIMENTS_TO_PLOT = glob(f"Output/{EXPERIMENT_FOLDER}/**/*_controller_*{ENVIRONMENT_NAME}*", recursive="True")
EXPERIMENTS_TO_PLOT = natsorted(EXPERIMENTS_TO_PLOT)

# Specify what the sweeped value is (labeled on x-axis)
SWEEP_VALUE = "resamp_per"
sweep_values = {
    "Resamp every": list(map(
        lambda x: x.split("=")[1].split("/")[0].split("\\")[0],
        [re.search(f"{SWEEP_VALUE}=.*(/|\\\)", path).group() for path in EXPERIMENTS_TO_PLOT]
    ))
}

# Compare configs associated with the different experiments

# Generate box plot: Each box represents the total cost statistics for a specific controller


def generate_global_plots() -> None:
    all_total_cost_data: "dict[str, list[float]]" = {}

    # Prepare average cost data
    for exp in EXPERIMENTS_TO_PLOT:
        all_total_cost_data[exp] = []

        path_to_experiment = exp
        config_filepaths = filter(
            lambda x: x.endswith(".yml"), os.listdir(path_to_experiment)
        )
        config_filepaths = list(
            map(lambda x: os.path.join(path_to_experiment, x), config_filepaths)
        )
        config = load(open(config_filepaths[0], "r"), FullLoader)

        cost_filepaths = filter(
            lambda x: "realized_cost_logged" in x and x.endswith(".npy"),
            os.listdir(path_to_experiment),
        )
        cost_filepaths = map(
            lambda x: os.path.join(path_to_experiment, x), cost_filepaths
        )
        for cost_filepath in cost_filepaths:
            cost_log: np.ndarray = np.load(cost_filepath)
            total_cost = float(np.squeeze(cost_log.mean()))
            all_total_cost_data[exp].append(total_cost)

    logger.info("Generating box plot...")
    box_plot_plotter = CostScatterPlotPlotter(
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        {
            "1_data_generation": {"controller_name": "", "environment_name": ""},
            "2_environments": {"": {"actuator_noise": 0}},
        },
    )
    box_plot_plotter.plot(all_total_cost_data, sweep_values, True)
    logger.info(pformat({k: sorted(v) for k, v in all_total_cost_data.items()}))
    logger.info("...done.")


if __name__ == "__main__":
    generate_global_plots()
