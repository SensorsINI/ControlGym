import importlib
import time

import gym
from yaml import FullLoader, load

from Environments import *
from Utilities.generate_plots import generate_plots
from Utilities.utils import get_logger

config = load(open("config.yml", "r"), Loader=FullLoader)
CONTROLLER_NAME, ENVIRONMENT_NAME = (
    config["controller_name"],
    config["environment_name"],
)
frames = []
logger = get_logger(__name__)

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT_NAME, **config["environments"][ENVIRONMENT_NAME])
    obs = env.reset(seed=config["environments"][ENVIRONMENT_NAME]["seed"])

    controller_module = importlib.import_module(f"Controllers.{CONTROLLER_NAME}")
    controller = getattr(controller_module, CONTROLLER_NAME)(
        env, **config["controllers"][CONTROLLER_NAME]
    )

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
    if config["controllers"][config["controller_name"]]["controller_logging"]:
        generate_plots(
            config=config,
            controller=controller,
            frames=frames if len(frames) > 0 else None,
        )
