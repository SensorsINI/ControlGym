from datetime import datetime

import numpy as np
from Controllers import Controller
from Visualizations.plot_environment import EnvironmentPlotter
from Visualizations.plot_horizon_costs import HorizonCostPlotter
from Visualizations.plot_input_plans import InputPlanPlotter
from yaml import dump

from Utilities.utils import get_logger, get_output_path

logger = get_logger(__name__)


def generate_plots(
    config: dict, controller: Controller, frames: list[np.ndarray] = None
):
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(get_output_path(timestamp_str, "config.yml"), "w") as f:
        dump(config, f)

    controller_output = controller.get_outputs()
    for n, a in controller_output.items():
        with open(
            get_output_path(timestamp_str, f"{n}.npy"),
            "wb",
        ) as f:
            np.save(f, a)

    logger.info("Creating horizon cost plot...")
    horizon_cost_plotter = HorizonCostPlotter(timestamp_str=timestamp_str)
    horizon_cost_plotter.plot(controller_output["J_logged"], save_to_image=True)
    logger.info("...done.")

    logger.info("Creating input plan animation...")
    input_plan_plotter = InputPlanPlotter(timestamp_str=timestamp_str)
    input_plan_plotter.plot(
        controller_output["Q_logged"],
        controller_output["J_logged"],
        save_to_video=True,
    )
    logger.info("...done.")

    if frames is not None:
        logger.info("Creating environment animation...")
        environment_plotter = EnvironmentPlotter(timestamp_str=timestamp_str)
        environment_plotter.plot(frames)
        logger.info("...done.")
    else:
        logger.info("Environment not rendered to file.")
