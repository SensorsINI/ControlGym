import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.armbot_batched import armbot_batched
import tensorflow as tf
import numpy as np

costoption = 1
costoption2 = 1
anglecost_fact = 1
# option 0 (costoption=1 and costoption2=1): only quadratic cost of end effector to target position
# option 1: add some ultra reward when reach to target
# option 2: add cost on angle changes between segments
config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
config2 = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml"), "r"),
    Loader=yaml.FullLoader,
)
discount_factor = float(config["armbot_batched"]["discounted_horizon"]["discount_factor"])
distance_factor = float(config["armbot_batched"]["discounted_horizon"]["distance_factor"])
mpc_horizon = int(config2["rpgd-tf"]["mpc_horizon"])
xtarget = armbot_batched.xtarget
ytarget = armbot_batched.ytarget
xobs = armbot_batched.xobs
yobs = armbot_batched.yobs
robs = armbot_batched.robs
useobs = armbot_batched.useobs


class discounted_horizon(cost_function_base):
    def get_distance_cost(self, states_tuple: TensorType):
        tuple2 = states_tuple
        theta = tuple2[0]
        xees = []
        yees = []
        xee = tf.cos(theta)
        yee = tf.cos(theta)
        xees.append(xee)
        yees.append(yee)
        for i in range(armbot_batched.num_states):
            if i > 0:
                theta += tuple2[i]
                xee += tf.cos(theta)
                yee += tf.sin(theta)
                xees.append(xee)
                yees.append(yee)
        cost = distance_factor * (
                (xee - xtarget) ** 2 + (yee - ytarget) ** 2
        )
        return cost, xees, yees, xee, yee
        
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        tuple2 = self.lib.unstack(states, armbot_batched.num_states, -1)
        cost, xees, yees, xee, yee = self.get_distance_cost(tuple2)
        
        if costoption2 == 1:
            cost2 = tf.zeros_like(tuple2[0])
            for i in range(len(tuple2) - 1):
                cost2 += tf.abs(tuple2[i + 1] - tuple2[i])
            cost += cost2 * anglecost_fact
        if useobs > 0:
            for i in range(len(xees)):
                cost_obs = (xees[i] - xobs) ** 2 + (yees[i] - yobs) ** 2
                cost2 = tf.where(tf.less_equal(cost_obs, robs ** 2), 1e6, 0.0)
                cost += cost2
        #crossing constaints, at least ee should not cross with other segments:
        for i in range(len(xees)-2):
            dist1 = (xees[i] - xee) ** 2 + (yees[i] - yee) ** 2
            dist2 = (xees[i+1] - xee) ** 2 + (yees[i+1] - yee) ** 2
            dist3 = (xees[i]-xees[i+1])**2 + (yees[i]-yees[i+1])**2
            diff=dist1+dist2-dist3
            cost2 = tf.where(tf.less_equal(tf.abs(diff), 0.0025), 1e4, 0.0)
            cost += cost2
        return cost

    def get_terminal_cost(self, terminal_states: TensorType):
        tuple2 = self.lib.unstack(terminal_states, armbot_batched.num_states, -1)
        cost, _, _, _, _ = self.get_distance_cost(tuple2)
        if costoption == 1:
            cost = tf.where(tf.less_equal(cost, 0.2), -1e6, 0.0)
        return self.lib.unsqueeze(cost, 1)

    # discounted cost adapted from existing acrobot discount horizon implementation
    def get_trajectory_cost(self, state_horizon: TensorType, inputs: TensorType,
                            previous_input: TensorType = None) -> TensorType:
        stage_costs = self.get_stage_cost(state_horizon[:, :-1, :], inputs,
                                          previous_input)  # Select all but last state of the horizon
        gamma = discount_factor * self.lib.ones_like(stage_costs)
        gamma = self.lib.cumprod(gamma, 1)

        terminal_costs = self.get_terminal_cost(state_horizon[:, -1, :])
        total_cost = self.lib.mean(self.lib.concat([gamma * stage_costs, terminal_costs], 1),
                                   1)  # Mean across the MPC horizon dimension
        return total_cost
