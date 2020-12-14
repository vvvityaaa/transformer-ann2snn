import tensorflow as tf
import numpy as np
from typing import Tuple, Callable

"""
All credits to Daniel A. Code is used from a tutorial
"""



@tf.custom_gradient
def spike_function(v_to_threshold: tf.Tensor) -> tuple:
    """
    A custom gradient for networks of spiking neurons.

    @param v_to_threshold: The difference between current and threshold voltage of the neuron.
    @type v_to_threshold: tf.float32
    @return: Activation z and gradient grad.
    @rtype: tuple
    """
    z = tf.cast(tf.greater(v_to_threshold, 1.), dtype=tf.float32)

    def grad(dy: tf.Tensor) -> tf.Tensor:
        """
        The gradient function for calculating the derivative of the spike-function.

        The return value is determined as follows:

        # @negative: v_to_threshold < 0 -> dy*0
        # @rest: v_to_threshold = 0 -> dy*0+
        # @thresh: v_to_threshold = 1 -> dy*1
        # @+thresh: v_to_threshold > 1 -> dy*1-
        # @2thresh: v_to_threshold > 2 -> dy*0
        #
        #         /\
        #        /  \
        # ______/    \______
        # -1   0   1  2   3  v_to_threshold

        @param dy: The previous upstream gradient.
        @return: The calculated gradient of this stage of the network
        """
        return [dy * tf.maximum(1 - tf.abs(v_to_threshold - 1), 0)]

    return z, grad


class IntegratorNeuronCell(tf.keras.layers.Layer):
    """
    A simple spiking neuron layer that integrates (sums up) the outputs of the previous layer.
    """

    def __init__(self, n_in, n_neurons, **kwargs):
        """
        Initialization function of the IntegratorNeuronCell.

        @param n_in: Number of inputs, i.e. outputs of previous layer.
        @param n_neurons: Number of neurons, i.e. outputs of this layer.
        @param kwargs: Additional parameters, forwarded to standard Layer init function of tf.
        """
        super(IntegratorNeuronCell, self).__init__(**kwargs)
        self.n_in = n_in
        self.n_neurons = n_neurons

        self.w_in = None

    def build(self, input_shape):
        """
        Creates the variables of this layer, i.e. creates and initializes the weights
        for all neurons within this layer.

        @param input_shape: Not needed for this layer.
        @type input_shape:
        """
        del input_shape  # Unused

        w_in = tf.random.normal((self.n_in, self.n_neurons), dtype=self.dtype)
        self.w_in = tf.Variable(initial_value=w_in / np.sqrt(self.n_in), trainable=True)

    @property
    def state_size(self) -> Tuple[int, int]:
        """
        Returns the state size depicted of cell and hidden state  as a tuple of number of neurons, number of neurons.
        @return:
        """
        return self.n_neurons, self.n_neurons

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """

        @param inputs:
        @param batch_size:
        @param dtype:
        @return:
        """
        del inputs  # Unused

        zeros = tf.zeros((batch_size, self.n_neurons), dtype=dtype)
        return zeros, zeros

    def call(self, input_at_t, states_at_t):
        """

        @param input_at_t:
        @param states_at_t:
        @return:
        """
        old_v, old_z = states_at_t

        i_t = tf.matmul(input_at_t, self.w_in)

        new_v = old_v + i_t
        new_z = tf.nn.softmax(new_v)

        return new_z, (new_v, new_z)


class LifNeuronCell(IntegratorNeuronCell):
    """
    A more advanced spiking tf layer building upon the IntegratorNeuronCell,
    but augmenting it with a leaky and fire functionality.
    """

    def __init__(self, n_in: int, n_neurons: int, tau: float = 20., threshold: float = 0.1,
                 activation_function: Callable[[tf.Tensor], tuple] = spike_function, **kwargs):
        """
        Initializes a (Recurrent)LifNeuronCell.

        @param n_in: Number of inputs, i.e. outputs of previous layer.
        @param n_neurons: Number of neurons, i.e. outputs of this layer.
        @param tau: The time constant tau.
        @param threshold: The threshold for the neurons in this layer.
        @param activation_function: The activation function for the LIF-Neuron, defaults to a simple spike-function.
        @param kwargs: Additional parameters, forwarded to standard Layer init function of tf.
        """
        super(LifNeuronCell, self).__init__(n_in, n_neurons, **kwargs)
        self.tau = tau
        self.decay = tf.exp(-1 / tau)
        self.threshold = threshold

        self.activation_function = activation_function

    def call(self, input_at_t, states_at_t):
        old_v, old_z = states_at_t

        i_t = tf.matmul(input_at_t, self.w_in)
        i_reset = old_z * self.threshold

        new_v = self.decay * old_v + (1.0 - self.decay) * i_t - i_reset
        new_z = self.activation_function(new_v / self.threshold)

        return new_z, (new_v, new_z)


class RecurrentLifNeuronCell(LifNeuronCell):
    """
    A recurrent spiking layer implementing a recurrent layer of LIF-Neurons.
    Each neuron has a connection to the previous/next layer as well recurrent
    connection to itself.
    """

    def build(self, input_shape):
        del input_shape  # Unused

        w_in = tf.random.normal((self.n_in, self.n_neurons), dtype=self.dtype)
        self.w_in = tf.Variable(initial_value=w_in / np.sqrt(self.n_in), trainable=True)

        w_rec = tf.random.normal((self.n_neurons, self.n_neurons), dtype=self.dtype)
        w_rec = tf.linalg.set_diag(w_rec, np.zeros(self.n_neurons))
        self.w_rec = tf.Variable(initial_value=w_rec / np.sqrt(self.n_neurons), trainable=True)

    def call(self, input_at_t, states_at_t):
        old_v, old_z = states_at_t

        i_t = tf.matmul(input_at_t, self.w_in) + tf.matmul(old_z, self.w_rec)
        i_reset = old_z * self.threshold

        new_v = self.decay * old_v + (1.0 - self.decay) * i_t - i_reset
        new_z = self.activation_function(new_v / self.threshold)

        return new_z, (new_v, new_z)
