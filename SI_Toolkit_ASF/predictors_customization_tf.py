import tensorflow as tf
from SI_Toolkit.Functions.TF.Compile import CompileTF

from SI_Toolkit_ASF.predictors_customization import (STATE_INDICES, STATE_VARIABLES, next_state_predictor_ODE, CONTROL_INPUTS, CONTROL_INDICES)

STATE_INDICES_TF = tf.lookup.StaticHashTable(  # TF style dictionary
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(STATE_INDICES.keys())),
        values=tf.constant(list(STATE_INDICES.values())),
    ),
    default_value=-100,
    name=None,
)


next_state_predictor_ODE_tf = next_state_predictor_ODE


class predictor_output_augmentation_tf:
    def __init__(self, net_info, lib, differential_network=False):
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

    @CompileTF
    def augment(self, net_output):

        output = net_output
        # if 'sin(x)' in self.features_augmentation:
        #     sin_x = tf.math.sin(net_output[..., self.index_x])[:, :, tf.newaxis]
        #     output = tf.concat([output, sin_x], axis=-1)

        return output
