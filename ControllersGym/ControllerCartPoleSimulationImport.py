from importlib import import_module
import numpy as np
import torch
from Environments import EnvironmentBatched, NumpyLibrary, PyTorchLibrary, TensorFlowLibrary

from ControllersGym import Controller
from Utilities.utils import CurrentRunMemory, SeedMemory

class ControllerCartPoleSimulationImport(Controller):
    def __init__(self, environment: EnvironmentBatched, **controller_config) -> None:
        super().__init__(environment, **controller_config)
        controller_name = controller_config["controller_name"]
        
        planning_env_config = environment.unwrapped.config.copy()
        planning_env_config.update({"computation_lib": TensorFlowLibrary})
        
        batch_size = controller_config.get("num_rollouts", controller_config.get("cem_rollouts", 1))
        env_mock = environment.__class__(batch_size=batch_size, **planning_env_config)
        env_mock.set_computation_library(TensorFlowLibrary)

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
        self.u_logged = self.u
        self.Q_logged = self._controller.Q_logged.copy()
        self.J_logged = self._controller.J_logged.copy()
        
        self.rollout_trajectories_logged = getattr(self._controller, "rollout_trajectories_logged", None)
        if self.rollout_trajectories_logged is not None:
            self.rollout_trajectories_logged = self.rollout_trajectories_logged.copy()
            l = getattr(CurrentRunMemory, "controller_logs", {})
            l.update({"rollout_trajectories_logged": self.rollout_trajectories_logged})
            setattr(CurrentRunMemory, "controller_logs", l)
        # Q: (batch_size x horizon_length x action_space)
        # J: (batch_size)
        self.s_logged = s.copy()
        return self.u
