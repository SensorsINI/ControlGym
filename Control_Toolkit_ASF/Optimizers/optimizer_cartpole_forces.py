"""
This is a linear-quadratic optimizer
The Jacobian of the model needs to be provided
"""

from typing import Tuple
from SI_Toolkit.computation_library import ComputationLibrary, TensorFlowLibrary

import numpy as np
import tensorflow as tf
import scipy

from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.Optimizers import template_optimizer
from Control_Toolkit.others.globals_and_utils import CompileTF
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper

from CartPoleSimulation.CartPole.state_utilities import (ANGLE_IDX, ANGLED_IDX, POSITION_IDX,
                                      POSITIOND_IDX)
from Control_Toolkit.others.globals_and_utils import create_rng


#Forces
from forces import forcespro
import numpy as np
from forces import get_userid

class optimizer_cartpole_forces(template_optimizer):
    supported_computation_libraries = {TensorFlowLibrary}

    def __init__(
            self,
            predictor: PredictorWrapper,
            cost_function: CostFunctionWrapper,
            num_states: int,
            num_control_inputs: int,
            control_limits: "Tuple[np.ndarray, np.ndarray]",
            computation_library: "type[ComputationLibrary]",
            seed: int,
            mpc_horizon: int,
            num_rollouts: int,
            optimizer_logging: bool,
            jacobian_path: str,
            action_max: float,
            state_max: list[float],
            Q: float,
            R: float
    ):
        super().__init__(
            predictor=predictor,
            cost_function=cost_function,
            num_states=num_states,
            num_control_inputs=num_control_inputs,
            control_limits=control_limits,
            optimizer_logging=optimizer_logging,
            seed=seed,
            num_rollouts=1,
            mpc_horizon=mpc_horizon,
            computation_library=computation_library,
        )

        #dynamically import jacobian module
        module_path, jacobian_method = jacobian_path.rsplit('.', 1)
        self.module = __import__(module_path, fromlist=["cartpole_jacobian"])
        self.jacobian = getattr(self.module, jacobian_method)

        self.action_low = -action_max
        self.action_high = +action_max
        self.Q = Q
        self.R = R

        self.optimizer_reset()



        #lower and upper bounds
        constrained_idx = [i for i,x in enumerate(state_max) if x!='inf']

        xmax = np.array([state_max[i] for i in constrained_idx])
        xmin = -xmax

        umin = np.array([self.action_low])
        umax = np.array([self.action_high])

        ubidx = [1] + [i+2 for i in constrained_idx]
        lbidx = ubidx

        self.nxc = len(constrained_idx)
        self.nx = len(state_max)
        self.nu = 1

        # Cost matrices for LQR controller
        self.Q = np.diag([self.Q] * self.nx)  # How much to punish x
        self.R = np.diag([self.R] * self.nu)  # How much to punish u

        # MPC setup
        N = self.mpc_horizon
        Q = self.Q
        R = self.R
        # terminal weight obtained from discrete-time Riccati equation
        P = Q


        # FORCESPRO multistage form
        # assume variable ordering zi = [u{i-1}, x{i}] for i=1...N
        # forcespro._set_forces_dir(forcespro.forces_dir)
        self.stages = forcespro.MultistageProblem(N)

        # for readability
        stages = self.stages
        nxc = self.nxc
        nx = self.nx
        nu = self.nu


        s = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
        jacobian = self.jacobian(s, 0.0)  # linearize around u=0.0
        A = jacobian[:, :-1]
        B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * self.action_high
        self.A = A
        self.B = B

        for i in range(N):
            # equality constraints
            if (i < N - 1):
                stages.eq[i]['C'] = np.hstack((np.zeros((nx, nu)), A))
            if (i > 0):
                stages.eq[i]['c'] = np.zeros((nx, 1))
            stages.eq[i]['D'] = np.hstack((B, -np.eye(nx)))


        for i in range(N):

            # dimensions
            stages.dims[i]['n'] = nx + nu  # number of stage variables
            stages.dims[i]['r'] = nx  # number of equality constraints
            stages.dims[i]['l'] = nxc + nu  # number of lower bounds
            stages.dims[i]['u'] = nxc + nu  # number of upper bounds

            # cost
            if (i == N - 1):
                stages.cost[i]['H'] = np.vstack(
                    (np.hstack((R, np.zeros((nu, nx)))), np.hstack((np.zeros((nx, nu)), P))))
            else:
                stages.cost[i]['H'] = np.vstack(
                    (np.hstack((R, np.zeros((nu, nx)))), np.hstack((np.zeros((nx, nu)), Q))))
            stages.cost[i]['f'] = np.zeros((nx + nu, 1))

            # lower bounds
            stages.ineq[i]['b']['lbidx'] = lbidx # lower bound acts on these indices
            stages.ineq[i]['b']['lb'] = np.concatenate((umin, xmin), 0)  # lower bound for this stage variable

            # upper bounds
            stages.ineq[i]['b']['ubidx'] = ubidx  # upper bound acts on these indices
            stages.ineq[i]['b']['ub'] = np.concatenate((umax, xmax), 0)  # upper bound for this stage variable

        # solver settings
        stages.codeoptions['name'] = 'myMPC_FORCESPRO'
        stages.codeoptions['printlevel'] = 0
        stages.codeoptions['floattype'] = 'float'

        stages.newParam('minusA_times_x0', [1], 'eq.c')  # RHS of first eq. constr. is a parameter: z1=-A*x0

        # define output of the solver
        stages.newOutput('u0', 1, list(range(1, nu + 1)))

        # generate code
        stages.generateCode(get_userid.userid)
        import myMPC_FORCESPRO_py
        self.myMPC_FORCESPRO_py = myMPC_FORCESPRO_py
        self.problem = myMPC_FORCESPRO_py.myMPC_FORCESPRO_params


    def cartpole_order2jacobian_order(self, s: np.ndarray):
        #Jacobian does not match the state order, permutation is needed
        new_s = np.ndarray(4)
        new_s[0] = s[POSITION_IDX]
        new_s[1] = s[POSITIOND_IDX]
        new_s[2] = s[ANGLE_IDX]
        new_s[3] = s[ANGLED_IDX]
        return new_s

    def jacobian_order2cartpole_order(self, s: np.ndarray):
        # Jacobian does not match the state order, permutation is needed
        new_s = np.ndarray(6)
        new_s[POSITION_IDX] = s[0]
        new_s[POSITIOND_IDX] = s[1]
        new_s[2] = np.cos(new_s[0])
        new_s[3] = np.sin(new_s[0])
        new_s[ANGLE_IDX] = s[2]
        new_s[ANGLED_IDX] = s[3]
        return new_s

    def step(self, s: np.ndarray, time=None):

        s = self.cartpole_order2jacobian_order(s)
        self.problem['minusA_times_x0'] = -np.dot(self.A, s)
        [solverout, exitflag, info] = self.myMPC_FORCESPRO_py.myMPC_FORCESPRO_solve(self.problem)
        if (exitflag == 1):
            u = solverout['u0']
            print('Problem solved in %5.3f milliseconds (%d iterations).' % (1000.0 * info.solvetime, info.it))
        else:
            print(info)
            raise RuntimeError('Some problem in solver')

        return u.astype(np.float32)

    def optimizer_reset(self):
        pass
        # state = np.array(
        #     [[s[POSITION_IDX] - self.env_mock.target_position], [s[POSITIOND_IDX]], [s[ANGLE_IDX]], [s[ANGLED_IDX]]])
        #
        # Q = np.dot(-self.K, state).item()
        #
        # Q = np.float32(Q * (1 + self.p_Q * self.rng_lqr.uniform(self.action_low, self.action_high)))
        # # Q = self.rng_lqr.uniform(-1.0, 1.0)
        #
        # # Clip Q
        # if Q > 1.0:
        #     Q = 1.0
        # elif Q < -1.0:
        #     Q = -1.0
        # else:
        #     pass

        # return Q
