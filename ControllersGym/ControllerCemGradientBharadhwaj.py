from importlib import import_module
import numpy as np
import torch
from Control_Toolkit.others.environment import EnvironmentBatched, PyTorchLibrary

# Import original paper's code
from ControllersGym.External.gradcem import GradCEMPlan

from ControllersGym import Controller


class ControllerCemGradientBharadhwaj(Controller):
    def __init__(self, environment: EnvironmentBatched, **controller_config) -> None:
        super().__init__(environment, **controller_config)

        self._num_rollouts = controller_config["num_rollouts"]
        self._horizon_steps: int = controller_config["mpc_horizon"]
        self._opt_iters = controller_config["cem_outer_it"]
        self._select_best_k = controller_config["cem_best_k"]

        _planning_env_config = environment.unwrapped.config.copy()
        _planning_env_config.update({"computation_lib": PyTorchLibrary})
        self._predictor_environment = getattr(
            import_module(f"Predictors.{controller_config['predictor_name']}"),
            controller_config["predictor_name"],
        )(
            environment.unwrapped.__class__(
                batch_size=self._num_rollouts, **_planning_env_config
            ),
            controller_config["seed"],
        )

        self._controller = GradCEMPlan(
            planning_horizon=self._horizon_steps,
            opt_iters=self._opt_iters,
            samples=self._num_rollouts,
            top_samples=self._select_best_k,
            env=self._predictor_environment,
            device=torch.device("cpu"),
            grad_clip=True,
            learning_rate=controller_config["grad_learning_rate"],
            grad_max=controller_config["grad_max"],
            sgd_momentum=controller_config["grad_sgd_momentum"],
            grad_epsilon=controller_config["grad_epsilon"],
        )

    def step(self, s: np.ndarray) -> np.ndarray:
        # self._predictor_environment.reset(s)

        self.Q, self.J = self._controller.forward(
            s=s,
            batch_size=1,
            return_plan=False,
            return_plan_each_iter=True,
        )
        # Select last optim iteration result of Q
        self.Q = self.Q[-1].swapaxes(0, 1)

        # Q: (batch_size x horizon_length x action_space)
        # J: (batch_size)
        self.u = self.Q[np.argmin(self.J), 0, :]
        self.s_logged = s.copy()

        self.u_logged = self.u.copy()
        self.J_logged, self.Q_logged = self.J.copy(), self.Q.copy()

        return self.u
