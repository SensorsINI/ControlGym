import importlib
import os
import gym
import time

GLOBAL_SEED = 1234

from Configs.cem_gradient_default import (
    ENV_NAME,
    NUM_ITERATIONS,
    CONTROLLER_NAME,
    CONTROLLER_CONFIG,
)
import Environments


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    obs = env.reset(seed=GLOBAL_SEED)

    controller_module = importlib.import_module(f"Controllers.{CONTROLLER_NAME}")
    controller = getattr(controller_module, CONTROLLER_NAME)(env, **CONTROLLER_CONFIG)

    for step in range(NUM_ITERATIONS):
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
