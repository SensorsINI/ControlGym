################################################
# ForcesPRO requires the cost function to be
# fixed at compile time. In order to change
# target at runtime it is necessary to provide it
# as parameter
################################################

import numpy as np

def standard_target(controller, parameters_map):
    return np.zeros(parameters_map['nz'])

def cartpole_target(controller, parameters_map):
    target = np.zeros(parameters_map['nz'])
    target[3] = controller.target_position.numpy()
    return target

def dubins_car_target(controller, parameters_map):
    target = np.zeros(parameters_map['nz'])
    nx = controller.target_point.shape[0]
    target[-nx:] = controller.target_point
    return target