from importlib import import_module
import numpy as np
import torch
from Environments import EnvironmentBatched, NumpyLibrary, PyTorchLibrary, TensorFlowLibrary

from Controllers import Controller


class ControllerCartPoleSimulationImport(Controller):
    def __init__(self, environment: EnvironmentBatched, **controller_config) -> None:
        super().__init__(environment, **controller_config)
        controller_name = controller_config["controller"]
        
        environment.set_computation_library(TensorFlowLibrary if controller_name[-2:] == "tf" else NumpyLibrary)

        controller_full_name = f"controller_{controller_name.replace('-', '_')}"
        self._controller = getattr(
            import_module(
                f"CartPoleSimulation.Controllers.{controller_full_name}"
            ),
            controller_full_name,
        )(**{**controller_config[controller_name], **{"environment": environment, "num_control_inputs": self._m}})

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
