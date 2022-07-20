from importlib import import_module
import numpy as np
import torch
from Environments import EnvironmentBatched, NumpyLibrary, PyTorchLibrary, TensorFlowLibrary

from ControllersGym import Controller
from Utilities.utils import SeedMemory

from yaml import load, FullLoader

config = load(open("config.yml", "r"), FullLoader)


class ControllerCartPoleSimulationImport(Controller):
    def __init__(self, environment: EnvironmentBatched, **controller_config) -> None:
        super().__init__(environment, **controller_config)
        controller_name = controller_config["controller_name"]
        
        env_name = config["data_generation"]["environment_name"]
        planning_env_config = {
            **config["environments"][env_name].copy(),
            **{"seed": SeedMemory.get_seeds()[0]},
            **{"computation_lib": TensorFlowLibrary},
        }
        batch_size = controller_config.get("num_rollouts", controller_config.get("cem_rollouts", 1))
        env_mock = environment.__class__(batch_size=batch_size, **planning_env_config)
        env_mock.set_computation_library(TensorFlowLibrary if controller_name[-2:] == "tf" else NumpyLibrary)

        self._controller = getattr(
            import_module(
                f"CartPoleSimulation.Controllers.{controller_name}"
            ),
            controller_name,
        )(**{**controller_config, **{"environment": env_mock, "num_control_inputs": self._m}})

    def step(self, s: np.ndarray) -> np.ndarray:
        # self._predictor_environment.reset(s)

        self.u = np.array(self._controller.step(s))
        if self.u.ndim == 0:
            self.u = self.u[np.newaxis]
        self.Q = self._controller.Q.copy()
        self.J = self._controller.J.copy()

        # Q: (batch_size x horizon_length x action_space)
        # J: (batch_size)
        self.s = s.copy()
        self._update_logs()
        return self.u
