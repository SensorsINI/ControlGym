"""
Script to generate plots which compare the effect of a hyperparameter sweep.
Folder structure
```
./Output/
  L <<datetime>>_sweep_<<sweep_variable>>/
    L controller_name=<<controller_1>>/
      L <<datetime>>_<<controller_name>>_<<environment_name>>_<<predictor_name>>/
        L <<configs and logfiles for a number of trials, e.g. 100 trials>>
    L controller_name=<<controller_2>>/
      L ...
    L ...
```

You need to provide the `<<datetime>>_sweep_<<sweep_variable>>/` entry to this script and it
will search for all suitable trials to compare within the subfolders.

"""
### ------------- 1. Specify the paths to the experiment ------------- ###
### Option 1: Specify paths manually, e.g.
# experiments_to_plot = [
#     "Output/20220905-151036_sweep_outer_its_controller_rpgd_tf/outer_its=0/20220905-151037_controller_rpgd_tf_CustomEnvironments_CartPoleSimulator-v0_predictor_ODE_tf",
#     "Output/20220905-151036_sweep_outer_its_controller_rpgd_tf/outer_its=1/20220905-153936_controller_rpgd_tf_CustomEnvironments_CartPoleSimulator-v0_predictor_ODE_tf",
#     ...
# ]

### Option 2: Specify a top-level folder
EXPERIMENT_FOLDER = "20221101-171103_sweep_controller_name"
ENVIRONMENT_NAME = "ObstacleAvoidance"
### ------------- Do not modify the following two lines ------------- ###
experiments_to_plot = glob(f"Output/{EXPERIMENT_FOLDER}/**/*_controller_*{ENVIRONMENT_NAME}*", recursive="True")
experiments_to_plot = natsorted(experiments_to_plot)
### ------------- ------------- ------------- ------------- ------------- ###


sweep_value = EXPERIMENT_FOLDER.split("sweep_")[1].split("_controller")[0]

### ------------- 2. Specify what the sweeped value is (labeled on x-axis) ------------- ###
sweep_values = {
    "description": "Controller Name",
    "xlabel": r"K\textsubscript{re}",
    "sweep_values": list(map(
        lambda x: x.split("=")[1].split("/")[0].split("\\")[0],
        [re.search(f"{sweep_value}=.*(/|\\\)", path).group() for path in experiments_to_plot]
    )),
}
sweep_values["boxcolors"] = ["white" for _ in range(len(sweep_values["sweep_values"]))]
# sweep_values["boxcolors"][-2:] = ["gray", "gray"]
sweep_values["ylabel"] = "Average Control Cost"
# sweep_values["ylabel"] = "Cost of Best Plan" if sweep_value == "resamp_per" else "Average Control Cost"


import os
import re
from datetime import datetime
from glob import glob
from pprint import pformat

import numpy as np
from natsort import natsorted
from Visualizations.plot_cost_scatter_plots import CostScatterPlotPlotter
from Visualizations.trajectory_age_plotter import TrajectoryAgePlotter
from yaml import FullLoader, load

from Utilities.utils import get_logger

logger = get_logger(__name__)


# Compare configs associated with the different experiments

# Generate box plot: Each box represents the total cost statistics for a specific controller


def generate_global_plots() -> None:
    all_total_cost_data: "dict[str, list[float]]" = {}
    all_trajectory_cost_data: "dict[str, list[float]]" = {}
    all_ages_data: "dict[str, list[float]]" = {}

    # Prepare average cost data
    for exp in experiments_to_plot:
        path_to_experiment = exp
        exp = exp.replace(".", "_")
        all_total_cost_data[exp], all_trajectory_cost_data[exp] = [], []
        all_ages_data[exp] = []

        config_filepaths = filter(
            lambda x: x.endswith(".yml"), os.listdir(path_to_experiment)
        )
        config_filepaths = list(
            map(lambda x: os.path.join(path_to_experiment, x), config_filepaths)
        )
        config = load(open(config_filepaths[0], "r"), FullLoader)

        realized_cost_filepaths = filter(
            lambda x: "realized_cost_logged" in x and x.endswith(".npy"),
            os.listdir(path_to_experiment),
        )
        realized_cost_filepaths = map(
            lambda x: os.path.join(path_to_experiment, x), realized_cost_filepaths
        )

        trajectory_cost_filepaths = filter(
            lambda x: "J_logged" in x and x.endswith(".npy"),
            os.listdir(path_to_experiment),
        )
        trajectory_cost_filepaths = map(
            lambda x: os.path.join(path_to_experiment, x), trajectory_cost_filepaths
        )

        trajectory_age_filepaths = filter(
            lambda x: "trajectory_ages_logged" in x and x.endswith(".npy"),
            os.listdir(path_to_experiment),
        )
        trajectory_age_filepaths = map(
            lambda x: os.path.join(path_to_experiment, x), trajectory_age_filepaths
        )

        for cost_filepath in realized_cost_filepaths:
            cost_log: np.ndarray = np.load(cost_filepath)
            total_cost = float(np.squeeze(cost_log.mean()))
            all_total_cost_data[exp].append(total_cost)

        for cost_filepath in trajectory_cost_filepaths:
            costs: np.ndarray = np.load(cost_filepath)
            average_best_trajectory = float(costs.min(axis=1).mean())
            all_trajectory_cost_data[exp].append(average_best_trajectory)

        for trajectory_age_filepath in trajectory_age_filepaths:
            try:
                trajectory_ages = np.load(trajectory_age_filepath)
            except:
                logger.info("No trajectory age file found. Skipping plot generation.")
            else:
                trajectory_age = list(trajectory_ages[:, 0])
                all_ages_data[exp].extend(trajectory_age)


    logger.info("Generating box plot...")
    box_plot_plotter = CostScatterPlotPlotter(
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        {
            "1_data_generation": {"controller_name": "", "environment_name": ""},
            "2_environments": {"": {"actuator_noise": 0}},
        },
    )
    if sweep_value == "resamp_per":
        box_plot_plotter.plot(all_trajectory_cost_data, sweep_values, True)
        c = np.array(list(all_trajectory_cost_data.values())).T
    else:
        box_plot_plotter.plot(all_total_cost_data, sweep_values, True)
        c = np.array(list(all_total_cost_data.values())).T
    logger.info(f"Average costs per experiment {np.around(np.mean(c, axis=0), 2)}")
    logger.info(f"Stdev costs per experiment {np.around(np.std(c, axis=0), 2)}")
    logger.info("...done.")

    # if all(len(v)>0 for v in all_ages_data.values()):
    #     logger.info("Generating trajectory age plot...")
    #     trajectory_age_plot = TrajectoryAgePlotter(
    #         datetime.now().strftime("%Y%m%d-%H%M%S"),
    #         {
    #             "1_data_generation": {"controller_name": "", "environment_name": ""},
    #             "2_environments": {"": {"actuator_noise": 0}},
    #         },
    #     )
    #     trajectory_age_plot.plot(all_ages_data, True)
    #     logger.info("...done.")


if __name__ == "__main__":
    generate_global_plots()
