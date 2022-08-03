from importlib import import_module

import numpy as np
from Environments import ENV_REGISTRY, register_envs
from Control_Toolkit.others.environment import NumpyLibrary

from Utilities.utils import CurrentRunMemory, SeedMemory


environment_module = (
    ENV_REGISTRY[CurrentRunMemory.current_environment_name]
    .split(":")[0]
    .split(".")[-1]
)
Environment = getattr(
    import_module(f"Environments.{environment_module}"), environment_module
)


STATE_VARIABLES = np.array([f"x_{i}" for i in range(1, Environment.num_states)])
STATE_INDICES = {x: np.where(STATE_VARIABLES == x)[0][0] for x in STATE_VARIABLES}
CONTROL_INPUTS = np.array([f"u_{i}" for i in range(Environment.num_actions)])
CONTROL_INDICES = {x: np.where(CONTROL_INPUTS == x)[0][0] for x in CONTROL_INPUTS}


register_envs()


class next_state_predictor_ODE:
    def __init__(self, dt, intermediate_steps, batch_size, **kwargs):
        self.s = None
        env_name = CurrentRunMemory.current_environment_name

        planning_env_config = {
            **CurrentRunMemory.controller_specific_params,
            **{"seed": SeedMemory.get_seeds()[0]},
            **{"computation_lib": NumpyLibrary},
        }
        EnvClass, EnvName = ENV_REGISTRY[env_name].split(":")
        self.env = getattr(import_module(EnvClass), EnvName)(
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
