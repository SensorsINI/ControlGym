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

xtarget = armbot_batched.xtarget
ytarget = armbot_batched.ytarget


class default(cost_function_base):
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
        return cost
