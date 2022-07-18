import tensorflow as tf
import gym

from Utilities.utils import SeedMemory
from yaml import load, FullLoader

from SI_Toolkit.TF.TF_Functions.Compile import Compile
import numpy as np
from Environments import TensorFlowLibrary

from SI_Toolkit_ASF_global.predictors_customization import STATE_INDICES

STATE_INDICES_TF = tf.lookup.StaticHashTable(  # TF style dictionary
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(STATE_INDICES.keys())), values=tf.constant(list(STATE_INDICES.values()))),
    default_value=-100, name=None
)
config = load(open("config.yml", "r"), Loader=FullLoader)

class next_state_predictor_ODE_tf():

    def __init__(self, dt, intermediate_steps, batch_size, **kwargs):
        self.s = None
        
        env_name = config["environment_name"]
        env_config = {**config["environments"][env_name].copy(), **{"seed": SeedMemory.seeds[0]}}
        planning_env_config = {**env_config, **{"computation_lib": TensorFlowLibrary}}
        self.env = gym.make(env_name, **env_config).unwrapped.__class__(
            batch_size=batch_size, **planning_env_config
        )

        self.intermediate_steps = tf.convert_to_tensor(intermediate_steps, dtype=tf.int32)
        self.t_step = tf.convert_to_tensor(dt / float(self.intermediate_steps), dtype=tf.float32)

    @Compile
    def step(self, s, Q, params):
        self.env.reset(s)
        next_state = self.env.step(Q)[0]
        self.env.lib.set_shape(next_state, self.env.lib.shape(self.env.state))
        return next_state


class predictor_output_augmentation_tf:
    def __init__(self, net_info):
        self.net_output_indices = {key: value for value, key in enumerate(net_info.outputs)}
        indices_augmentation = []
        features_augmentation = []
        if 'sin(x)' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['sin(x)'])
            features_augmentation.append('sin(x)')

        self.indices_augmentation = indices_augmentation
        self.features_augmentation = features_augmentation
        self.augmentation_len = len(self.indices_augmentation)

        if 'x' in net_info.outputs:
            self.index_x = tf.convert_to_tensor(self.net_output_indices['x'])

    def get_indices_augmentation(self):
        return self.indices_augmentation

    def get_features_augmentation(self):
        return self.features_augmentation

    @Compile
    def augment(self, net_output):

        output = net_output
        # if 'sin(x)' in self.features_augmentation:
        #     sin_x = tf.math.sin(net_output[..., self.index_x])[:, :, tf.newaxis]
        #     output = tf.concat([output, sin_x], axis=-1)

        return output
