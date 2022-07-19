from importlib import import_module

import numpy as np
from Environments import ENV_REGISTRY, NumpyLibrary
from yaml import FullLoader, load
import gym

from Utilities.utils import SeedMemory


config = load(open("config.yml", "r"), Loader=FullLoader)
environment_module = ENV_REGISTRY[config['environment_name']].split(":")[0].split(".")[-1]
Environment = getattr(import_module(f"Environments.{environment_module}"), environment_module)


STATE_VARIABLES = np.array([f"x{i}" for i in range(Environment.num_states)])
STATE_INDICES = {x: np.where(STATE_VARIABLES == x)[0][0] for x in STATE_VARIABLES}
CONTROL_INPUTS = np.array([f"Q{i}" for i in range(Environment.num_actions)])
CONTROL_INDICES = {x: np.where(CONTROL_INPUTS == x)[0][0] for x in CONTROL_INPUTS}


config = load(open("config.yml", "r"), Loader=FullLoader)


class next_state_predictor_ODE():

    def __init__(self, dt, intermediate_steps, batch_size, **kwargs):
        
        self.s = None
        env_name = config["environment_name"]
        
        env_config = {**config["environments"][env_name].copy(), **{"seed": SeedMemory.seeds[0]}}
        planning_env_config = {**env_config, **{"computation_lib": NumpyLibrary}}
        self.env = gym.make(env_name, **env_config).unwrapped.__class__(
            batch_size=batch_size, **planning_env_config
        )

        self.intermediate_steps = intermediate_steps
        self.t_step = np.float32(dt / float(self.intermediate_steps))

    def step(self, s, Q, params):
        self.env.reset(s.copy())
        next_state, _, _, _ = self.env.step(Q)
        return next_state


def augment_predictor_output(output_array, net_info):
    pass
    return output_array
