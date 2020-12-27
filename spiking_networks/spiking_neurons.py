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


class SuperSpike(tf.keras.layers.Layer):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        # ctx.save_for_backward(input)
        out = tf.zeros_like(input)
        out[input > 0] = 1.0
        return out

    # @staticmethod
    # def backward(ctx, grad_output):
    #     """
    #     In the backward pass we receive a Tensor we need to compute the
    #     surrogate gradient of the loss with respect to the input.
    #     Here we use the normalized negative part of a fast sigmoid
    #     as this was done in Zenke & Ganguli (2018).
    #     """
    #     input, = ctx.saved_tensors
    #     grad_input = grad_output.clone()
    #     grad = grad_input / (SuperSpike.scale * tf.math.abs(input) + 1.0) ** 2
    #     return grad


# TODO: Adapt code from PyTorch to TensorFlow
class SQN(tf.keras.Model):

    def __init__(self, network_shape, alpha, beta, weight_scale=1, encoding='constant', decoding='potential',
                 threshold=1, simulation_time=100, reset='subtraction', two_input_neurons=False,
                 add_bias_as_observation=False):
        """Args:
            network_shape: shape of the network given in the form [5,17,17,2] for 5 input neurons, two hidden layers with 17 neurons and 2 output neurons
            device: device for the torch tensors
            alpha: synapse decay
            beta: membrane decay
            weight_scale: determines the standard deviaiton of the random weights
            encoding: 'constant','equidistant' or 'poisson' for the three possible input methods
            decoding: 'potential' or 'spikes' for the two different output methods, also 'first_spike' is possible which returns once the fiirst output neuron spikes
            threshold: 'threshold' when a spike occurs
            simulation_time: number of time steps to be simulated
            reset: either 'subtraction' or 'zero'
            two_input_neurons: for input methods 'equidistant' and 'poisson' if one output neuron is to be used for negative and positive inputs each
            add_bias_as_observation: this option is for training using SpyTorch as SpyTorch allows no hidden layer biases. If true a 1 is added as constant input"""
        self.alpha = alpha
        self.beta = beta
        self.encoding = encoding
        self.decoding = decoding
        self.threshold = threshold
        self.simulation_time = simulation_time
        self.input_size = network_shape[0]
        self.reset = reset
        self.add_bias_as_observation = add_bias_as_observation
        self.two_input_neurons = two_input_neurons

        if self.add_bias_as_observation:
            # add one more neuron to the architecture at the input, because the bias acts as an additional input
            network_shape[0] += 1

        self.weights = []
        self.bias = []
        for i in range(0, len(network_shape)-1):
            self.weights.append(tf.raw_ops.Empty((network_shape[i], network_shape[i+1]), dtype=tf.float32))
            initializer = tf.random_normal_initializer(mean=0.0, stddev=weight_scale / np.sqrt(network_shape[i]))

            self.weights[i] = tf.Variable(initializer(self.weights[i].shape, dtype=tf.float32))
            # initialize all biases with None, SpyTorch does not support biases.
            self.bias.append(None)

        self.spike_fn = SuperSpike.apply

        if self.two_input_neurons:
            self.weights[0] = tf.concat([self.weights[0], (-1)*(self.weights[0])])
            self.input_size *= 2

    def forward(self, input_data):
        """Simulates the network for the number of time steps specified in self.simulation_time
        Args:
            input_data: tensor

        Returns:
             potentials (or spikes depending on the output method) of the output neurons"""

        # if two input neurons is True, double the input size, where the first set of values represents the positive inputs and the second set the negative inputs.
        if self.two_input_neurons:
            input_data = tf.concat.cat([input_data * (input_data > 0), (-1)*(input_data * (input_data < 0))])

        # reshape input such that it is in the form (batch_size, input_dimenstion) and not (input_dimension,) or (input_dimension)
        if input_data.shape == (self.input_size,) or input_data.dim() == 1:
            input_data = input_data.reshape(1, self.input_size)
        batch_size = input_data.shape[0]

        if self.add_bias_as_observation:
            bias = tf.ones((batch_size, 1), dtype=tf.float32)
            input_data = tf.concat((input_data, bias), axis=1)

        # reset the array for membrane potential and synaptic variables
        syn = []
        mem = []
        for l in range(0, len(self.weights)):
            syn.append(tf.zeros((batch_size, self.weights[l].shape[1]), dtype=tf.float32))
            mem.append(tf.zeros((batch_size, self.weights[l].shape[1]), dtype=tf.float32))

        # Here we define two lists which we use to record the membrane potentials and output spikes
        mem_rec = []
        spk_rec = []
        # Additionally we define a list that counts the spikes in the output layer, if the output method 'spikes' is used
        if self.decoding == 'spikes':
            spk_count = tf.zeros((batch_size, self.weights[-1].shape[1]), dtype=tf.float64)

        if self.encoding == 'equidistant':
            # spike counter is used to count the number of spike for each input neuron so far
            spike_counter = tf.ones_like(input_data)
            fixed_distance = 1 / input_data

        # Here we loop over time
        for t in range(self.simulation_time):
            # append the new timestep to mem_rec and spk_rec
            mem_rec.append([])
            spk_rec.append([])

            if self.encoding == 'constant':
                input = input_data.detach().clone()
            elif self.encoding == 'poisson':
                #generate poisson distributed input
                spike_snapshot = tf.Tensor(np.random.uniform(low=0, high=1, size=input_data.shape))
                input = (spike_snapshot <= input_data).float()
            elif self.encoding == 'equidistant':
                # generate fixed number of equidistant spikes
                input = (tf.ones_like(input_data)*t == tf.math.round(fixed_distance*spike_counter)).float()
                spike_counter += input
            else:
                raise Exception('Encoding Method '+str(self.encoding)+' not implemented')

            # loop over layers
            for l in range(len(self.weights)):
                # define impulse
                if l == 0:
                    h = tf.einsum("ab,bc->ac", [input, self.weights[0]])
                else:
                    h = tf.einsum("ab,bc->ac", [spk_rec[len(spk_rec) - 1][l - 1], self.weights[l]])
                # add bias
                if self.bias[l] is not None:
                    h += self.bias[l]

                # calculate the spikes for all layers (decoding='spikes' or 'first_spike') or for all but the last layer (decoding='potential')
                if self.decoding == 'spikes' or self.decoding == 'first_spike' or l < len(self.weights)-1:
                    mthr = mem[l] - self.threshold
                    out = self.spike_fn(mthr)
                    rst = tf.zeros_like(mem[l], device=self.device)
                    c = (mthr > 0)
                    rst[c] = tf.ones_like(mem[l], device=self.device)[c]
                    # count the spikes in the output layer
                    if self.decoding == 'spikes' and l == len(self.weights)-1:
                        spk_count = tf.add(spk_count, out)
                else:
                    # else reset is 0 (= no reset)
                    c = tf.zeros_like(mem[l], dtype=tf.bool, device=self.device)
                    rst = tf.zeros_like(mem[l], device=self.device)

                # calculate the new synapse potential
                new_syn = self.alpha * syn[l] + h
                # calculate new membrane potential
                if self.reset == 'subtraction':
                    new_mem = self.beta * mem[l] + syn[l] - rst
                elif self.reset == 'zero':
                    new_mem = self.beta * mem[l] + syn[l]
                    new_mem[c] = 0

                mem[l] = new_mem
                syn[l] = new_syn

                mem_rec[len(mem_rec) - 1].append(mem[l])
                spk_rec[len(spk_rec) - 1].append(out)

                if self.decoding == 'first_spike' and l == len(self.weights)-1:
                    if tf.reduce_sum(out) > 0:
                        return out
        if self.decoding == 'potential':
            # return the final recorded membrane potential (len(mem_rec)-1) in the output layer (-1)
            return mem_rec[len(mem_rec)-1][-1]
        if self.decoding == 'spikes':
            # return the sum over the spikes in the output layer
            return spk_count
        else:
            raise Exception('Decoding Method '+str(self.decoding)+' not implemented')
