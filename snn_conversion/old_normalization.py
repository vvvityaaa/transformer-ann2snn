import tensorflow as tf
import numpy as np


def get_normalized_weights(model, x_test, percentile=100):
    x_test = x_test[::10]
    max_activation = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.ReLU):
            activation = tf.keras.Model(inputs=model.inputs, outputs=layer.output)(x_test).numpy()
            if np.percentile(activation, percentile) > max_activation:
                max_activation = np.percentile(activation, percentile)
        elif isinstance(layer, tf.keras.layers.Dense):
            if layer.activation.__name__ == 'relu':
                activation = tf.keras.Model(inputs=model.inputs, outputs=layer.output)(x_test).numpy()
                if np.percentile(activation, percentile) > max_activation:
                    max_activation = np.percentile(activation, percentile)

    weights = model.get_weights()
    if max_activation == 0:
        print("\n" + "-" * 32 + "\nNo normalization\n" + "-" * 32)
    else:
        print("\n" + "-" * 32 + "\nNormalizing by", max_activation, "\n" + "-" * 32)
        for i in range(len(weights)):
            weights[i] /= (max_activation)
    return weights