from datetime import datetime

import numpy as np
from Controllers import Controller
from Utilities.utils import get_output_path
from Visualizations.plot_horizon_costs import HorizonCostPlotter
from Visualizations.plot_input_plans import InputPlanPlotter
from Visualizations.plot_environment import EnvironmentPlotter
from yaml import dump


def generate_plots(config: dict, controller: Controller, frames: list[np.ndarray] = None):
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

    horizon_cost_plotter = HorizonCostPlotter(timestamp_str=timestamp_str)
    horizon_cost_plotter.plot(controller_output["J_logged"], save_to_image=True)

    input_plan_plotter = InputPlanPlotter(timestamp_str=timestamp_str)
    input_plan_plotter.plot(
        controller_output["Q_logged"],
        controller_output["J_logged"],
        save_to_video=True,
    )

    if frames is not None:
        environment_plotter = EnvironmentPlotter(timestamp_str=timestamp_str)
        environment_plotter.plot(frames)
