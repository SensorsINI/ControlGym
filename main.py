import importlib
import time
from datetime import datetime

import gym
from yaml import FullLoader, load, dump

from Environments import register_envs
from Utilities.generate_plots import generate_plots
from Utilities.utils import OutputPath, get_logger
from numpy.random import SeedSequence

config = load(open("config.yml", "r"), Loader=FullLoader)
CONTROLLER_NAME, ENVIRONMENT_NAME = (
    config["controller_name"],
    config["environment_name"],
)
logger = get_logger(__name__)

register_envs()

timestamp = datetime.now()
seed_entropy = config["seed_entropy"]
if seed_entropy is None:
    seed_entropy = int(timestamp.timestamp())
    logger.info("No seed entropy specified. Setting to posix timestamp.")

seeds = SeedSequence(entropy=seed_entropy).generate_state(config["num_experiments"])
timestamp = timestamp.strftime("%Y%m%d-%H%M%S")

if __name__ == "__main__":
    for i in range(config["num_experiments"]):
        config["environments"][ENVIRONMENT_NAME].update({"seed": int(seeds[i])})
        env = gym.make(ENVIRONMENT_NAME, **config["environments"][ENVIRONMENT_NAME])
        obs = env.reset(seed=int(seeds[i]))

        config["controllers"][CONTROLLER_NAME].update({"seed": int(seeds[i])})
        controller_module = importlib.import_module(f"Controllers.{CONTROLLER_NAME}")
        controller = getattr(controller_module, CONTROLLER_NAME)(
            env, **config["controllers"][CONTROLLER_NAME]
        )

        frames = []
        for step in range(config["num_iterations"]):
            action = controller.step(obs)
            new_obs, reward, done, info = env.step(action)
            if config["render_for_humans"]:
                env.render(mode="human")
            if config["controllers"][config["controller_name"]]["controller_logging"]:
                frames.append(env.render(mode="rgb_array"))

            time.sleep(0.001)

            # If the epsiode is up, then start another one
            if done:
                env.reset()

            logger.debug(
                f"\nStep       : {step}\nObservation: {obs}\nAction     : {action}\n"
            )
            obs = new_obs

        # Close the env
        env.close()

        # Generate plots
        if config["num_experiments"] > 1:
            OutputPath.RUN_NUM = i + 1

        if config["controllers"][CONTROLLER_NAME]["controller_logging"]:
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
