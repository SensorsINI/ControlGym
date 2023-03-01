import yaml
import os

from SI_Toolkit.computation_library import TensorType
from Control_Toolkit.Cost_Functions import cost_function_base
from Environments.armbot_batched import armbot_batched
import tensorflow as tf
import numpy as np

config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)
config2 = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml"), "r"),
    Loader=yaml.FullLoader,
)
# 'n_horizon': self.config_controller["mpc_horizon"]
discount_factor = float(config["armbot_batched"]["discounted_horizon"]["discount_factor"])
mpc_horizon=int(config2["rpgd-tf"]["mpc_horizon"])
xtarget = armbot_batched.xtarget
ytarget = armbot_batched.ytarget

class discounted_horizon(cost_function_base):
    def _get_stage_cost(self, states: TensorType, inputs: TensorType, previous_input: TensorType) -> TensorType:
        tuple2 = self.lib.unstack(states, armbot_batched.num_states, -1)
        theta = tuple2[0]
        xee = tf.cos(theta)
        yee = tf.cos(theta)
        for i in range(armbot_batched.num_states):
            if i > 0:
                theta += tuple2[i]
                xee += tf.cos(theta)
                yee += tf.sin(theta)
        cost = (
                (xee - xtarget) ** 2 + (yee - ytarget) ** 2
        )
        cost2 = tf.where(tf.less_equal(cost, 0.01), -1000.0, 0)
        cost+=cost2
        # tuple3 = self.lib.unstack(cost, mpc_horizon, -1)
        # for i in range(len(tuple3)):
        #     tmp=tuple3[i].numpy()
        #     tmp[tmp<0.01]=-1000
        #     tmp=tf.Variable(tmp)
        #     tuple3[i]+=tmp
        # cost = self.lib.stack(tuple3, 1)
        # costnp=cost.numpy()
        # costnp[costnp<0.01]=-1000
        # cost=cost+costnp
        # cost=tf.convert_to_tensor(costnp)
        # cost2=tf.Variable(cost)
        return cost
    #discounted cost adapted from existing discount horizon implementation
    def get_trajectory_cost(self, state_horizon: TensorType, inputs: TensorType, previous_input: TensorType = None) -> TensorType:
        stage_costs = self.get_stage_cost(state_horizon[:, :-1, :], inputs, previous_input)  # Select all but last state of the horizon
        gamma = discount_factor * self.lib.ones_like(stage_costs)
        gamma = self.lib.cumprod(gamma, 1)

        terminal_costs = self.get_terminal_cost(state_horizon[:, -1, :])
        total_cost = self.lib.mean(self.lib.concat([gamma * stage_costs, terminal_costs], 1), 1)  # Mean across the MPC horizon dimension
        return total_cost
