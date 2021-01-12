# -*- coding: utf-8 -*-
# Credits: SNN-Toolbox https://github.com/NeuromorphicProcessorProject/snn_toolbox
# and Deep-Spiking-Q-Networks https://github.com/vhris/Deep-Spiking-Q-Networks
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np


def weight_conversion_model(weights, bias):
    """
    Simple model-based conversion model proposed by Diehl et al.
    :param weights: weights of the network.
    :param bias: bias of the network.
    :return: rescaled weights.
    """
    # Get weights from trained network
    converted_weights = weights
    converted_bias = bias

    # model based normalization
    previous_factor = 1
    for l in range(len(converted_weights)):
        max_pos_input = 0
        # Find maximum input for this layer
        for o in range(converted_weights[l].shape[0]):
            input_sum = 0
            for i in range(converted_weights[l].shape[1]):
                input_sum += tf.math.maximum(0, converted_weights[l][o, i])
            if converted_bias is not None and converted_bias[l] is not None:
                input_sum += tf.math.maximum(0, converted_bias[l][o])
            max_pos_input = tf.math.maximum(max_pos_input, input_sum)

        # get the maximum weight in the layer, in case all weights are negative, max_pos_input would be zero, so we use the max weight to rescale instead
        max_wt = tf.math.reduce_max(converted_weights[l])
        if converted_bias is not None and converted_bias[l] is not None:
            max_bias = tf.math.reduce_max(converted_bias[l])
            max_wt = tf.math.maximum(max_wt, max_bias)
        scale_factor = tf.math.maximum(max_wt, max_pos_input)
        # Rescale all weights
        applied_factor = scale_factor / previous_factor
        converted_weights[l] = converted_weights[l] / applied_factor
        if converted_bias is not None and converted_bias[l] is not None:
            converted_bias[l] = converted_bias[l] / scale_factor
        previous_factor = scale_factor
        print(f"Scale factor for this layer is {previous_factor}")

    return converted_weights, converted_bias


def weight_conversion_robust_and_data_based(weights, bias, model, data, normalization_method='robust',
                                            ppercentile=0.99):

    """
    Two methods proposed by Diehl et al and Rueckauer et al. Both methods are data-based, so they use weights and activations to
    find the best scaling factor.
    :param weights: weights of the network.
    :param bias: bias of the network.
    :param model: ann model.
    :param data: dataset to determine activations.
    :param normalization_method: type of normalization - robust (Rueckauer) or data (Diehl).
    :param ppercentile: percentile of the activation, which is taken from maximal activation.
    :return: rescaled weights.
    """
    if normalization_method == 'data':
        ppercentile = 1.0

    # Get weights from trained network
    converted_weights = weights
    converted_bias = bias

    # use training set to find max_act for each neuron

    activations = []
    for l in range(0, len(converted_weights)):
        activation = get_activations_layer(model.input, model.layers[l].output, data)
        activation_per_neuron = [np.max(activation[:, i]) for i in range(activation.shape[1])]
        activations.append(activation_per_neuron)

    previous_factor = 1
    for l in range(len(converted_weights)):
        # get the p-percentile of the activation
        pos_inputs = activations[l]
        pos_inputs.sort()
        max_act = pos_inputs[int(ppercentile * (len(pos_inputs) - 1))]
        # get the maximum weight in the layer
        max_wt = tf.math.reduce_max(converted_weights[l])
        if converted_bias is not None and converted_bias[l] is not None:
            max_bias = tf.math.reduce_max(converted_bias[l])
            max_wt = tf.math.maximum(max_wt, max_bias)
        scale_factor = tf.math.maximum(max_wt, max_act)

        applied_factor = scale_factor / previous_factor
        # rescale weights
        converted_weights[l] = converted_weights[l] / applied_factor

        # rescale bias
        if converted_bias is not None and converted_bias[l] is not None:
            converted_bias[l] = converted_bias[l] / scale_factor
        previous_factor = scale_factor
        print(f"Scale factor for this layer is {previous_factor}")

    return converted_weights, converted_bias


def get_activations_layer(layer_in, layer_out, data, batch_size=32):

    """
    Getting activation for specific layer of neural network.
    :param layer_in: input layer of a model. For sequential models first layer, for functional model.layers[0].input can
    be used.
    :param layer_out: layer for which activations should be computed. For functional model.layers[i].output can be used.
    :param data: dataset.
    :param batch_size: batch_size of batches in which dataset should be divided.
    :return: activations for a specific layer for all
    """

    if len(data) % batch_size != 0:
        data = data[: -(len(data) % batch_size)]

    return Model(layer_in, layer_out).predict(data, batch_size)




