from importlib import import_module

import gym
import numpy as np
from Control_Toolkit.others.environment import (EnvironmentBatched,
                                                TensorFlowLibrary)
from Utilities.utils import CurrentRunMemory, SeedMemory

from ControllersGym import Controller


class ControllerCartPoleSimulationImport(Controller):
    def __init__(self, environment: EnvironmentBatched, **controller_config) -> None:
        super().__init__(environment, **controller_config)
        controller_name = controller_config["controller_name"]
        
        for attr in ["num_rollouts", "num_rollouts", "mpc_rollouts"]:
            batch_size = controller_config.get(attr, None)
            if batch_size is not None:
                break
        if batch_size is None:
            raise ValueError("Controller needs one of num_rollouts, num_rollouts, mpc_rollouts to be set in config")

        self._controller = getattr(
            import_module(
                f"Control_Toolkit.Controllers.{controller_name}"
            ),
            controller_name,
        )(**{**controller_config, **{"environment": environment, "num_control_inputs": self._m}})

    def step(self, s: np.ndarray) -> np.ndarray:
        # self._predictor_environment.reset(s)

        self.u = np.array(self._controller.step(s))
        if self.u.ndim == 0:
            self.u = self.u[np.newaxis]
        self.u_logged = self.u.copy()
        self.Q_logged = self._controller.Q_logged.copy()
        self.J_logged = self._controller.J_logged.copy()
        
        self.rollout_trajectories_logged = getattr(self._controller, "rollout_trajectories_logged", None)
        if self.rollout_trajectories_logged is not None:
            self.rollout_trajectories_logged = self.rollout_trajectories_logged.copy()
            l = getattr(CurrentRunMemory, "controller_logs", {})
            l.update({"rollout_trajectories_logged": self.rollout_trajectories_logged})
            setattr(CurrentRunMemory, "controller_logs", l)
        self.trajectory_ages_logged = getattr(self._controller, "trajectory_ages_logged", None)
        if self.trajectory_ages_logged is not None:
            self.trajectory_ages_logged = self.trajectory_ages_logged.copy()
            l = getattr(CurrentRunMemory, "controller_logs", {})
            l.update({"trajectory_ages_logged": self.trajectory_ages_logged})
            setattr(CurrentRunMemory, "controller_logs", l)
        # Q: (batch_size x horizon_length x action_space)
        # J: (batch_size)
        self.s_logged = s.copy()
        return self.u
