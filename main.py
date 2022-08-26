import importlib
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Any
import numpy as np

import gym
from numpy.random import SeedSequence
from yaml import dump

from ControllersGym import Controller
from Environments import register_envs
from Utilities.csv_helpers import save_to_csv
from Utilities.generate_plots import generate_experiment_plots
from Utilities.utils import (
    OutputPath,
    SeedMemory,
    get_logger,
    get_name_of_controller_module,
)

# Keep allowing absolute imports within CartPoleSimulation subgit
sys.path.append(os.path.join(os.path.abspath("."), "CartPoleSimulation"))

logger = get_logger(__name__)

# Gym API: Register custom environments
register_envs()


def run_data_generator(
    controller_name: str,
    environment_name: str,
    num_experiments: int,
    config: "dict[str, Any]",
    run_for_ML_Pipeline=False,
    record_path=None,
):
    # Generate seeds and set timestamp
    timestamp = datetime.now()
    seed_entropy = config["1_data_generation"]["seed_entropy"]
    if seed_entropy is None:
        seed_entropy = int(timestamp.timestamp())
        logger.info("No seed entropy specified. Setting to posix timestamp.")

    seed_sequences = SeedSequence(entropy=seed_entropy).spawn(num_experiments)
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")

    if run_for_ML_Pipeline:
        # Get training/validation split
        frac_train, frac_val = config["1_data_generation"]["split"]
        assert record_path is not None, "If ML mode is on, need to provide record_path."

    # Set current controller/env names in config which is saved later
    config["1_data_generation"].update(
        {"controller_name": controller_name, "environment_name": environment_name}
    )

    # Loop through independent experiments
    for i in range(num_experiments):
        # Generate new seeds for environment and controller
        seeds = seed_sequences[i].generate_state(3)
        SeedMemory.set_seeds(seeds)
        config["2_environments"][environment_name].update({"seed": int(seeds[0])})
        config["4_controllers"][controller_name].update({"seed": int(seeds[2])})

        # Flatten global controller config values
        config["4_controllers"][controller_name].update(
            {
                k: config["4_controllers"][k]
                for k in ["controller_logging", "mpc_horizon"]
            }
        )

        # Create environment and call reset
        render_mode = (
            "human"
            if config["1_data_generation"]["render_for_humans"]
            else "single_rgb_array"
        )

        import matplotlib

        if render_mode == "human":
            matplotlib.use("Qt5Agg")
        else:
            matplotlib.use("Agg")

        env = gym.make(
            environment_name,
            **config["2_environments"][environment_name],
            render_mode=render_mode,
        )
        obs = env.reset(seed=int(seeds[1]))

        # Instantiate controller
        controller_module_name = get_name_of_controller_module(controller_name)
        controller_module = importlib.import_module(
            f"ControllersGym.{controller_module_name}"
        )
        controller: Controller = getattr(controller_module, controller_module_name)(
            **{
                **{
                    "environment": deepcopy(env.unwrapped),
                    "controller_name": controller_name,
                },
                **config["4_controllers"][controller_name],
                **config["2_environments"][environment_name],
            },
        )

        frames = []
        start_time = time.time()
        num_iterations = config["1_data_generation"]["num_iterations"]
        for step in range(num_iterations):
            action = controller.step(obs)
            new_obs, reward, done, info = env.step(action)
            controller.realized_cost_logged = np.array([-reward])
            controller.update_logs()
            frames.append(env.render())

            time.sleep(0.001)

            # If the episode is up, start a new experiment
            if done:
                break

            logger.debug(
                f"\nStep          : {step+1}/{num_iterations}\nObservation   : {obs}\nPlanned Action: {action}\n"
            )
            obs = new_obs
        
        end_time = time.time()
        control_freq = num_iterations / (end_time - start_time)
        logger.debug(f"Achieved average control frequency of {round(control_freq, 2)}Hz ({round(1.0e3/control_freq, 2)}ms per iteration)")

        # Close the env
        env.close()

        # Generate plots
        OutputPath.RUN_NUM = i + 1
        controller_output = controller.get_outputs()

        if run_for_ML_Pipeline:
            # Save data as csv
            if i < int(frac_train * num_experiments):
                csv = os.path.join(record_path, "Train")
            elif i < int((frac_train + frac_val) * num_experiments):
                csv = os.path.join(record_path, "Validate")
            else:
                csv = os.path.join(record_path, "Test")
            os.makedirs(csv, exist_ok=True)
            save_to_csv(config, controller, environment_name, csv)
        elif config["4_controllers"]["controller_logging"]:
            if render_mode != "human":
                # Generate and save plots in default location
                generate_experiment_plots(
                    config=config,
                    controller_output=controller_output,
                    timestamp=timestamp_str,
                    frames=frames if len(frames) > 0 else None,
                )
            with open(
                OutputPath.get_output_path(timestamp_str, "config", ".yml"), "w"
            ) as f:
                dump(config, f)
