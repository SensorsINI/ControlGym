import casadi
import yaml
import os
"""
Forces requires a function of the cost in the form
objective = f(z,p)
to derive equality constraints
"""

config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)


def pendulum(z, p):
    return -casadi.cos(z[1])

def continuous_mountaincar(z, p):
    return -casadi.sin(3 * z[1]) / ((z[1] - 1.25) ** 2)

def continuous_mountaincar_approximated(z, p):
    return -1.27*(z[1] + 0.4)**3 - 1.56 * (z[1] + 0.4)**2 + 0.00326758 * (z[1] + 0.4) + 0.322505

def cartpole_weights():
    cartpole_angle_weight = float(config["cartpole_simulator_batched"]["default"]["angle_weight"])
    cartpole_position_weight = float(config["cartpole_simulator_batched"]["default"]["position_weight"])
    return cartpole_angle_weight, cartpole_position_weight

def cartpole_simulator1(z, p):
    cartpole_angle_weight, cartpole_position_weight = cartpole_weights()
    return -cartpole_angle_weight*casadi.cos(z[1]) + cartpole_position_weight*(z[3] - p[3])**2

def cartpole_simulator2(z, p):
    cartpole_angle_weight, cartpole_position_weight = cartpole_weights()
    return -cartpole_angle_weight*(z[1]**2) + cartpole_position_weight*(z[3] - p[3])**2