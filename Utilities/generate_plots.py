from typing import Optional
import numpy as np
from Controllers import Controller
from Visualizations.plot_horizon_costs import HorizonCostPlotter
from Visualizations.plot_input_plans import InputPlanPlotter

from Utilities.utils import get_logger, OutputPath
from Visualizations.plot_summary import SummaryPlotter

logger = get_logger(__name__)


def generate_plots(
    config: dict,
    controller: Controller,
    timestamp: str,
    frames: list[np.ndarray] = None,
):
    controller_output = controller.get_outputs()
    for n, a in controller_output.items():
        with open(
            OutputPath.get_output_path(timestamp, str(n), ".npy"),
            "wb",
        ) as f:
            np.save(f, a)

    logger.info("Creating summary plot...")
    horizon_cost_plotter = SummaryPlotter(timestamp=timestamp, config=config)
    horizon_cost_plotter.plot(
        controller_output["s_logged"], controller_output["u_logged"], save_to_image=True
    )
    logger.info("...done.")

    logger.info("Creating horizon cost plot...")
    horizon_cost_plotter = HorizonCostPlotter(timestamp=timestamp, config=config)
    horizon_cost_plotter.plot(controller_output["J_logged"], save_to_image=True)
    logger.info("...done.")

    logger.info("Creating input plan animation...")
    input_plan_plotter = InputPlanPlotter(timestamp=timestamp, config=config)
    input_plan_plotter.plot(
        controller_output["Q_logged"],
        controller_output["J_logged"],
        frames,
        save_to_video=True,
    )
    logger.info("...done.")
