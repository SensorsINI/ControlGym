from typing import Optional
import numpy as np
from ControllersGym import Controller
from Visualizations.plot_horizon_costs import HorizonCostPlotter
from Visualizations.plot_input_plans import InputPlanPlotter

from Utilities.utils import get_logger, OutputPath
from Visualizations.plot_summary import SummaryPlotter

logger = get_logger(__name__)


def generate_experiment_plots(
    config: dict,
    controller_output: dict[str, np.ndarray],
    timestamp: str,
    frames: "list[np.ndarray]" = None,
):
    for n, a in controller_output.items():
        with open(
            OutputPath.get_output_path(timestamp, str(n), ".npy"),
            "wb",
        ) as f:
            np.save(f, a)

    if controller_output["s_logged"] is not None and controller_output["u_logged"] is not None:
        logger.info("Creating summary plot...")
        horizon_cost_plotter = SummaryPlotter(timestamp=timestamp, config=config)
        horizon_cost_plotter.plot(
            controller_output["s_logged"],
            controller_output["u_logged"],
            save_to_image=config["1_data_generation"]["save_plots_to_file"],
        )
        logger.info("...done.")
    else:
        logger.info("States and inputs were not saved in controller. Not generating plot.")

    if controller_output["J_logged"] is not None:
        logger.info("Creating horizon cost plot...")
        horizon_cost_plotter = HorizonCostPlotter(timestamp=timestamp, config=config)
        horizon_cost_plotter.plot(
            controller_output["J_logged"],
            save_to_image=config["1_data_generation"]["save_plots_to_file"],
        )
        logger.info("...done.")
    else:
        logger.info("Costs were not saved in controller. Not generating plot.")

    if controller_output["Q_logged"] is not None and controller_output["J_logged"] is not None:
        logger.info("Creating input plan animation...")
        input_plan_plotter = InputPlanPlotter(timestamp=timestamp, config=config)
        input_plan_plotter.plot(
            controller_output["Q_logged"],
            controller_output["J_logged"],
            frames,
            save_to_video=config["1_data_generation"]["save_plots_to_file"],
        )
        logger.info("...done.")
    else:
        logger.info("Input plans and costs were not saved in controller. Not generating plot.")
