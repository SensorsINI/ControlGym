import importlib
import os
import time
from datetime import datetime
from copy import deepcopy

import sys
sys.path.insert(0, os.path.join(os.path.abspath("."), "CartPoleSimulation"))

import gym
import pygame
from numpy.random import SeedSequence
from yaml import FullLoader, dump, load

from Environments import register_envs
from Utilities.generate_plots import generate_plots
from Utilities.utils import OutputPath, SeedMemory, get_logger
from Utilities.csv_helpers import save_to_csv

if not pygame.display.get_init():
    # Set dummy output device when machine is headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"

config = load(open("config.yml", "r"), Loader=FullLoader)
CONTROLLER_NAME, ENVIRONMENT_NAME = (
    config["data_generation"]["controller_name"],
    config["data_generation"]["environment_name"],
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

    seed_sequences = SeedSequence(entropy=seed_entropy).spawn(config["data_generation"]["num_experiments"])
    timestamp = timestamp.strftime("%Y%m%d-%H%M%S")

    if run_for_ML_Pipeline:
        frac_train, frac_val = config["data_generation"]["split"]
        assert record_path is not None, "If ML mode is on, need to provide record_path."

    # Loop through independent experiments
    for i in range(config["data_generation"]["num_experiments"]):
        seeds = seed_sequences[i].generate_state(3)
        SeedMemory.seeds = seeds
        config["environments"][ENVIRONMENT_NAME].update({"seed": int(seeds[0])})
        env = gym.make(ENVIRONMENT_NAME, **config["environments"][ENVIRONMENT_NAME], render_mode="human" if config["data_generation"]["render_for_humans"] else "single_rgb_array")
        obs = env.reset(seed=int(seeds[1]))

        config["controllers"][CONTROLLER_NAME].update({"seed": int(seeds[2])})
        controller_module = importlib.import_module(f"ControllersGym.{CONTROLLER_NAME}")
        controller = getattr(controller_module, CONTROLLER_NAME)(
            **{
                **config["controllers"][CONTROLLER_NAME],
                **{
                    k: config["controllers"][k]
                    for k in ["controller_logging", "mpc_horizon", "dt", "predictor"]
                },
                **{"environment": deepcopy(env)},
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
        if config["data_generation"]["num_experiments"] > 1:
            OutputPath.RUN_NUM = i + 1

        if run_for_ML_Pipeline:
            # Save data as csv
            if i < int(frac_train * config["data_generation"]["num_experiments"]):
                csv = os.path.join(record_path, "Train")
            elif i < int((frac_train + frac_val) * config["data_generation"]["num_experiments"]):
                csv = os.path.join(record_path, "Validate")
            else:
                csv = os.path.join(record_path, "Test")
            os.makedirs(csv, exist_ok=True)
            csv = os.path.join(csv, "Experiment")
            save_to_csv(config, controller, csv)
        elif config["controllers"]["controller_logging"]:
            # Generate and save plots in default location
            generate_plots(
                config=config,
                controller=controller,
                timestamp=timestamp,
                frames=frames if len(frames) > 0 else None,
            )
            with open(
                OutputPath.get_output_path(timestamp, "config", ".yml"), "w"
            ) as f:
                dump(config, f)


if __name__ == "__main__":
    run_data_generator()