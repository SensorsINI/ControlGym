from importlib import import_module
from typing import Callable

import numpy as np
from Environments import ENV_REGISTRY, register_envs
from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import NumpyLibrary, TensorType

from Utilities.utils import CurrentRunMemory, SeedMemory


environment_module = (
    ENV_REGISTRY[CurrentRunMemory.current_environment_name]
    .split(":")[0]
    .split(".")[-1]
)
Environment: "type[EnvironmentBatched]" = getattr(
    import_module(f"Environments.{environment_module}"), environment_module
)


STATE_VARIABLES = np.array([f"x_{i}" for i in range(1, Environment.num_states)])
STATE_INDICES = {x: np.where(STATE_VARIABLES == x)[0][0] for x in STATE_VARIABLES}
CONTROL_INPUTS = np.array([f"u_{i}" for i in range(Environment.num_actions)])
CONTROL_INDICES = {x: np.where(CONTROL_INPUTS == x)[0][0] for x in CONTROL_INPUTS}


register_envs()


class next_state_predictor_ODE:
    def __init__(self, dt: float, intermediate_steps: int, batch_size: int, **kwargs):
        self.s = None

        self.step_fun = CurrentRunMemory.current_environment.step_dynamics

        self.intermediate_steps = intermediate_steps
        self.t_step = dt / float(self.intermediate_steps)

    def step(self, s, Q):
        for _ in range(self.intermediate_steps):
            next_state = self.step_fun(s, Q, self.t_step)
            s = next_state
        return next_state


def augment_predictor_output(output_array, net_info):
    pass
    return output_array
