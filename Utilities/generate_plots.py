import numpy as np
from Visualizations.plot_horizon_costs import HorizonCostPlotter
from Visualizations.old.plot_input_plans import InputPlanPlotter

from Utilities.utils import OutputPath, get_logger
from Visualizations.plot_summary import SummaryPlotter

logger = get_logger(__name__)

def generate_experiment_plots(
    config: dict,
    environment_config: dict,
    controller_output: "dict[str, np.ndarray]",
    timestamp: str,
    frames: "list[np.ndarray]" = None,
):
    if frames is not None:
        # Save video of experiment renderings
        # Only if `save_plots_to_file` in `config.yml` was True
        from gymnasium.utils.save_video import save_video
        save_video(frames, OutputPath.get_output_path(timestamp, None), fps=20, name_prefix=f"recording_{OutputPath.RUN_NUM}")
    
    if (
        controller_output["s_logged"] is not None
        and controller_output["u_logged"] is not None
    ):
        # Plot the evolution of all state and input variables over time
        logger.info("Creating summary plot...")
        horizon_cost_plotter = SummaryPlotter(path=OutputPath.get_output_path(timestamp), run_config=config, environment_config=environment_config)
        horizon_cost_plotter.plot(
            controller_output["s_logged"],
            controller_output["u_logged"],
            save_to_image=config["save_plots_to_file"],
        )
        logger.info("...done.")
    else:
        logger.info(
            "States and inputs were not saved in controller. Not generating plot."
        )

    if controller_output["J_logged"] is not None:
        # Plot the distribution of rollout costs over time
        logger.info("Creating horizon cost plot...")
        horizon_cost_plotter = HorizonCostPlotter(path=OutputPath.get_output_path(timestamp), run_config=config, environment_config=environment_config)
        horizon_cost_plotter.plot(
            controller_output["J_logged"],
            save_to_image=config["save_plots_to_file"],
        )
        logger.info("...done.")
    else:
        logger.info("Costs were not saved in controller. Not generating plot.")

    ### Commented lines below because the input plan plots have not been of interest recently
    # if (
    #     controller_output["Q_logged"] is not None
    #     and controller_output["J_logged"] is not None
    # ):
    #     # Plot a histogram of the ages of the rollouts
    #     logger.info("Creating input plan animation...")
    #     input_plan_plotter = InputPlanPlotter(path=OutputPath.get_output_path(timestamp), run_config=config, environment_config=environment_config)
    #     input_plan_plotter.plot(
    #         controller_output["Q_logged"],
    #         controller_output["J_logged"],
    #         frames,
    #         save_to_video=config["save_plots_to_file"],
    #     )
    #     logger.info("...done.")
    # else:
    #     logger.info(
    #         "Input plans and costs were not saved in controller. Not generating plot."
    #     )
