import numpy as np
import torch
from Environments import EnvironmentBatched

# Import original paper's code
from mpc.gradcem import GradCEMPlan

from Controllers import Controller


class ControllerCemGradientBharadhwaj(Controller):
    def __init__(self, environment: EnvironmentBatched, **controller_config) -> None:
        super().__init__(environment, **controller_config)

        self._num_rollouts = controller_config["cem_rollouts"]
        self._horizon_steps = int(
            controller_config["mpc_horizon"] / controller_config["dt"]
        )
        self._opt_iters = controller_config["cem_outer_it"]
        self._select_best_k = controller_config["cem_best_k"]
        self._max_grad = controller_config["max_grad"]

        _planning_env_config = environment.unwrapped.config.copy()
        _planning_env_config.update({"computation_lib": "pytorch"})
        self._planning_env = environment.unwrapped.__class__(
            batch_size=self._num_rollouts, **_planning_env_config
        )

        self._controller = GradCEMPlan(
            planning_horizon=self._horizon_steps,
            opt_iters=self._opt_iters,
            samples=self._num_rollouts,
            top_samples=self._select_best_k,
            env=self._planning_env,
            device=torch.device("cpu"),
            grad_clip=True,
        )

    def step(self, s: np.ndarray) -> np.ndarray:
        # self._planning_env.reset(s)

        self.Q, self.J = self._controller.forward(
            s=s,
            batch_size=1,
            return_plan=False,
            return_plan_each_iter=True,
        )
        # Select last optim iteration result of Q
        self.Q, self.J = (
            self.Q[-1].detach().numpy().swapaxes(0, 1),
            self.J.detach().numpy(),
        )
        # Q: (batch_size x horizon_length x action_space)
        # J: (batch_size)
        self.u = self.Q[np.argmin(self.J), 0, :]
        self.s = s.copy()
        self._update_logs()
        return self.u
