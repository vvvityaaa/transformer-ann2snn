import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking import data_structures

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import _caching_device
# from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell
import matplotlib.pyplot as plt
import numpy as np


class Linear(tf.keras.layers.Layer):
    """
    Unchanged example from https://www.tensorflow.org/guide/keras/custom_layers_and_models
    Basically a Dense-Layer, nothing new here.
    """
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class DenseRNN(tf.keras.layers.Layer):
    """
    Same as the dense layer above, but for use in an RNN (= with internal state, although unused)
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(DenseRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True)
        #self.built = True

    def call(self, input_at_t, states_at_t):
        output_at_t = tf.matmul(input_at_t, self.w) + self.b
        states_at_t_plus_1 = output_at_t  # unused
        return output_at_t, states_at_t_plus_1


class IF(tf.keras.layers.Layer):
    """
    IF layer. Adds input*weight+bias to the internal state.
    Generates spikes as output when state>threshold is reached.
    Reset by substraction.
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(IF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True)
        #self.built = True

    def call(self, input_at_t, states_at_t):
        potential = states_at_t[0] + (tf.matmul(input_at_t, self.w) + self.b)
        output_at_t = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class SpikingReLU(tf.keras.layers.Layer):
    """
    Roughly the same as the IF above, but without matmul.
    So a standard dense layer without activation has to be used before.
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingReLU, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # only to suppress warning
    def call(self, input_at_t, states_at_t, training):  # training as keyword only needed with the decorator
        potential = states_at_t[0] + input_at_t
        output_at_t = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class SpikingSigmoid(tf.keras.layers.Layer):
    """
    Works like the SpikingReLU but is shiftet by 0.5 to the left.
    An neuron with spike adaptation might result in less conversion loss
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingSigmoid, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # only to suppress warning
    def call(self, input_at_t, states_at_t, training):  # training as keyword only needed with the decorator
        potential = states_at_t[0] + (input_at_t + 0.5)
        output_at_t = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class SpikingTanh(tf.keras.layers.Layer):
    """
    Roughly the same as the IF above, but without matmul.
    So a standard dense layer without activation has to be used before.
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingTanh, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # only to suppress warning
    def call(self, input_at_t, states_at_t, training):  # training as keyword only needed with the decorator
        potential = states_at_t[0] + (input_at_t)
        excitatory = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        inhibitory = -1 * tf.cast(tf.math.less(potential, -1), dtype=tf.float32)
        output_at_t = excitatory + inhibitory
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class Accumulate(tf.keras.layers.Layer):
    """
    Accumulates all input as state for use with a softmax layer.
    """
    # ToDo: include softmax layer directly here?
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(Accumulate, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # only to suppress warning
    def call(self, input_at_t, states_at_t, training):  # training keyword only needed with the decorator
        output_at_t = states_at_t[0] + input_at_t
        states_at_t_plus_1 = output_at_t
        return output_at_t, states_at_t_plus_1
