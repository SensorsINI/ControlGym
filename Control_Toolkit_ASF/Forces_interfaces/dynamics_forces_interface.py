import numpy as np
import casadi
from CartPoleSimulation.CartPole.cartpole_model import _cartpole_ode
from CartPoleSimulation.CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX, ANGLE_SIN_IDX,
                                                             ANGLED_IDX, POSITION_IDX, POSITIOND_IDX)
from CartPoleSimulation.CartPole.cartpole_jacobian import cartpole_jacobian
from Environments.acrobot_batched import acrobot_batched
"""
Forces requires a function of the dynamics in the form
dx/dt = f(x,u,p)
to derive equality constraints
"""


def cartpole_linear_dynamics(s, u, p):
    # calculate dx/dt evaluating f(x,u) = A(x,u)*x + B(x,u)*u
    action_high = 2.62
    jacobian = cartpole_jacobian(s, 0.0)  # linearize around u=0.0
    A = jacobian[:, :-1]
    B = np.reshape(jacobian[:, -1], newshape=(4, 1)) * action_high
    return A @ s + B @ u

def cartpole_non_linear_dynamics(s, u, p: 0):
    u_max = 2.62
    ca, sa, angleD, positionD = np.cos(s[0]), np.sin(s[0]), s[1], s[3]
    angleDD, positionDD = _cartpole_ode(ca, sa, angleD, positionD, u*u_max)
    sD = casadi.SX.sym('sD', 4, 1)
    sD[0] = angleD
    sD[1] = angleDD
    sD[2] = positionD
    sD[3] = positionDD
    return sD

def pendulum_dynamics(s, u, p):
    # th, thD, sth, cth = s[0], s[1], s[2], s[3]
    g = 10.0
    l = 1.0
    m = 1.0
    sD = casadi.SX.sym('sD', 2, 1)
    sD[0] = s[1]
    sD[1] = 3 * g / (2 * l) * np.sin(s[0]) + 3.0 / (m * l ** 2) * u
    return sD

def acrobot_dynamics(s, u, p):
    # PDIP non linear solver are not suitable for this environment,
    # since it has continuous dynamics but discrete action space
    # Mixed integer non linear solver (MINLP) is required
    # https://forces.embotech.com/Documentation/examples/minlp_f8_aircraft/index.html?highlight=mixed%20integer

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    m1 = LINK_MASS_1
    m2 = LINK_MASS_2
    l1 = LINK_LENGTH_1
    lc1 = LINK_COM_POS_1
    lc2 = LINK_COM_POS_2
    I1 = LINK_MOI
    I2 = LINK_MOI
    g = 9.8

    # theta1, theta2, dtheta1, dtheta2 = tuple(list(s))
    # theta1, theta2, dtheta1, dtheta2 = np.unstack(s, 4, 1)
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]

    a = u
    d1 = (
        m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * casadi.cos(theta2))
        + I1
        + I2
    )
    d2 = m2 * (lc2**2 + l1 * lc2 * casadi.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * casadi.cos(theta1 + theta2 - casadi.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * casadi.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * casadi.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * casadi.cos(theta1 - casadi.pi / 2)
        + phi2
    )

    # the following line is consistent with the java implementation and the
    # book
    ddtheta2 = (
        a
        + d2 / d1 * phi1
        - m2 * l1 * lc2 * dtheta1**2 * casadi.sin(theta2)
        - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

    sD = casadi.SX.sym('sD', 4, 1)
    sD[0] = dtheta1
    sD[1] = dtheta2
    sD[2] = ddtheta1
    sD[3] = ddtheta2

    return sD

def continuous_mountaincar(s,u,p):
    power = 0.0015
    force = u
    min_position = -1.2

    position = s[0]
    velocity = s[1]
    sD = casadi.SX.sym('sD', 2, 1)

    sD[0] = s[1] * casadi.logic_not(casadi.logic_and((position <= min_position), (velocity < 0)))
    sD[1] = force * power - 0.0025 * casadi.cos(3 * position)

    return sD