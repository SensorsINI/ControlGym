from importlib import import_module
import tensorflow as tf
import gym

from Utilities.utils import SeedMemory
from yaml import load, FullLoader

from SI_Toolkit.TF.TF_Functions.Compile import Compile
import numpy as np
from Environments import ENV_REGISTRY, TensorFlowLibrary

from SI_Toolkit_ASF.predictors_customization import STATE_INDICES

STATE_INDICES_TF = tf.lookup.StaticHashTable(  # TF style dictionary
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(STATE_INDICES.keys())),
        values=tf.constant(list(STATE_INDICES.values())),
    ),
    default_value=-100,
    name=None,
)
config = load(open("config.yml", "r"), Loader=FullLoader)


class next_state_predictor_ODE_tf:
    def __init__(self, dt, intermediate_steps, batch_size, **kwargs):
        self.s = None

        env_name = config["1_data_generation"]["environment_name"]
        planning_env_config = {
            **config["2_environments"][env_name].copy(),
            **{"seed": SeedMemory.get_seeds()[0]},
            **{"computation_lib": TensorFlowLibrary},
        }
        EnvClass, EnvName = ENV_REGISTRY[env_name].split(":")
        self.env = getattr(import_module(EnvClass), EnvName)(
            batch_size=batch_size, **planning_env_config
        )

        self.intermediate_steps = tf.convert_to_tensor(
            intermediate_steps, dtype=tf.int32
        )
        self.t_step = tf.convert_to_tensor(
            dt / float(self.intermediate_steps), dtype=tf.float32
        )

    def step(self, s, Q, params):
        self.env.reset(s)
        next_state = self.env.step_tf(s, Q)
        return next_state


class predictor_output_augmentation_tf:
    def __init__(self, net_info, differential_network=False):
        self.net_output_indices = {
            key: value for value, key in enumerate(net_info.outputs)
        }
        indices_augmentation = []
        features_augmentation = []

        self.indices_augmentation = indices_augmentation
        self.features_augmentation = features_augmentation
        self.augmentation_len = len(self.indices_augmentation)

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
