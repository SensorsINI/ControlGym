import importlib
import os
import gym
import time
from yaml import load, FullLoader

GLOBAL_SEED = 1234

from Environments import *

config = load(open("config.yml", "r"), Loader=FullLoader)
CONTROLLER_NAME, ENVIRONMENT_NAME = config["controller_name"], config["environment_name"]

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT_NAME)
    obs = env.reset(seed=GLOBAL_SEED)

    controller_module = importlib.import_module(f"Controllers.{CONTROLLER_NAME}")
    controller = getattr(controller_module, CONTROLLER_NAME)(env, **config["controllers"][CONTROLLER_NAME])

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
