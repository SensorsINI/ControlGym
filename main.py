import importlib
import os
import sys
import time
from copy import deepcopy
from datetime import datetime

import gym
import pygame
from numpy.random import SeedSequence
from yaml import FullLoader, dump, load

from Environments import register_envs
from Utilities.csv_helpers import save_to_csv
from Utilities.generate_plots import generate_plots
from Utilities.utils import OutputPath, SeedMemory, get_logger, get_name_of_controller_module


# Keep allowing absolute imports within CartPoleSimulation subgit
sys.path.insert(0, os.path.join(os.path.abspath("."), "CartPoleSimulation"))

# Set dummy output device when machine is headless
if not pygame.display.get_init():
    os.environ["SDL_VIDEODRIVER"] = "dummy"

config = load(open("config.yml", "r"), Loader=FullLoader)
CONTROLLER_NAME, ENVIRONMENT_NAME, NUM_EXPERIMENTS = (
    config["data_generation"]["controller_name"],
    config["data_generation"]["environment_name"],
    config["data_generation"]["num_experiments"],
)
logger = get_logger(__name__)

register_envs()


def run_data_generator(run_for_ML_Pipeline=False, record_path=None):
    # Generate seeds and set timestamp
    timestamp = datetime.now()
    seed_entropy = config["data_generation"]["seed_entropy"]
    if seed_entropy is None:
        seed_entropy = int(timestamp.timestamp())
        logger.info("No seed entropy specified. Setting to posix timestamp.")

    seed_sequences = SeedSequence(entropy=seed_entropy).spawn(NUM_EXPERIMENTS)
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")

    if run_for_ML_Pipeline:
        # Get training/validation split
        frac_train, frac_val = config["data_generation"]["split"]
        assert record_path is not None, "If ML mode is on, need to provide record_path."

    # Loop through independent experiments
    for i in range(NUM_EXPERIMENTS):
        # Generate new seeds for environment and controller
        seeds = seed_sequences[i].generate_state(3)
        SeedMemory.set_seeds(seeds)
        config["environments"][ENVIRONMENT_NAME].update({"seed": int(seeds[0])})
        config["controllers"][CONTROLLER_NAME].update({"seed": int(seeds[2])})
        
        # Flatten global controller config values
        config["controllers"][CONTROLLER_NAME].update({
            k: config["controllers"][k]
            for k in ["controller_logging", "mpc_horizon", "dt"]
        })
        
        # Create environment and call reset
        render_mode = "human" if config["data_generation"]["render_for_humans"] else "single_rgb_array"
        env = gym.make(ENVIRONMENT_NAME, **config["environments"][ENVIRONMENT_NAME], render_mode=render_mode)
        obs = env.reset(seed=int(seeds[1]))
        
        # Instantiate controller
        controller_module_name = get_name_of_controller_module(CONTROLLER_NAME)
        controller_module = importlib.import_module(f"ControllersGym.{controller_module_name}")
        controller = getattr(controller_module, controller_module_name)(
            **{
                **{"environment": deepcopy(env), "controller_name": CONTROLLER_NAME},
                **config["controllers"][CONTROLLER_NAME],
                **config["environments"][ENVIRONMENT_NAME],
            },
        )

        frames = []
        for step in range(config["data_generation"]["num_iterations"]):
            action = controller.step(obs)
            new_obs, reward, done, info = env.step(action)
            if config["controllers"]["controller_logging"]:
                frames.append(env.render())

            time.sleep(0.001)

            # If the epsiode is up, then start another one
            if done:
                env.reset()

            logger.debug(
                f"\nStep       : {step+1}/{config['data_generation']['num_iterations']}\nObservation: {obs}\nAction     : {action}\n"
            )
            obs = new_obs

        # Close the env
        env.close()

        # Generate plots
        if NUM_EXPERIMENTS > 1:
            OutputPath.RUN_NUM = i + 1

        if run_for_ML_Pipeline:
            # Save data as csv
            if i < int(frac_train * NUM_EXPERIMENTS):
                csv = os.path.join(record_path, "Train")
            elif i < int((frac_train + frac_val) * NUM_EXPERIMENTS):
                csv = os.path.join(record_path, "Validate")
            else:
                csv = os.path.join(record_path, "Test")
            os.makedirs(csv, exist_ok=True)
            save_to_csv(config, controller, csv)
        elif config["controllers"]["controller_logging"]:
            # Generate and save plots in default location
            generate_plots(
                config=config,
                controller=controller,
                timestamp=timestamp_str,
                frames=frames if len(frames) > 0 else None,
            )
            with open(
                OutputPath.get_output_path(timestamp_str, "config", ".yml"), "w"
            ) as f:
                dump(config, f)


if __name__ == "__main__":
    run_data_generator()
