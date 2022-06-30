import importlib
import time
from datetime import datetime

import gym
from yaml import FullLoader, load, dump

from Environments import *
from Visualizations.plot_horizon_costs import HorizonCostPlotter
from Visualizations.plot_input_plans import InputPlanPlotter
from Utilities.utils import get_output_path

config = load(open("config.yml", "r"), Loader=FullLoader)
CONTROLLER_NAME, ENVIRONMENT_NAME = (
    config["controller_name"],
    config["environment_name"],
)

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT_NAME)
    obs = env.reset(seed=config["environments"][ENVIRONMENT_NAME]["seed"])

    controller_module = importlib.import_module(f"Controllers.{CONTROLLER_NAME}")
    controller = getattr(controller_module, CONTROLLER_NAME)(
        env, **config["controllers"][CONTROLLER_NAME]
    )

    for step in range(config["num_iterations"]):
        action = controller.step(obs)
        new_obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.001)

        # If the epsiode is up, then start another one
        if done:
            env.reset()

        print(f"Step: {step}")
        print(obs)
        print(action)
        obs = new_obs

    # Close the env
    env.close()

    # Generate plots
    if config["controllers"][CONTROLLER_NAME]["controller_logging"]:
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
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

    with open(get_output_path(timestamp_str, "config.yml"), "w") as f:
        dump(config, f)