import tensorflow as tf


def robust_weight_normalization(model, x_test, ppercentile=1):

    """
    Robust weight normalization proposed by Rueckauer et al
    """

    x_test = x_test[::10]
    prev_factor = 1
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.ReLU) or (isinstance(layer, tf.keras.layers.Dense) and
                                                       layer.activation.__name__ == 'relu'):
            activation = tf.keras.Model(inputs=model.inputs, outputs=layer.output)(x_test).numpy()
            # calculating max over different dimensions
            for _ in range(activation.ndim - 1):
                activation = tf.math.reduce_max(activation, axis=0)
            activation = tf.sort(activation)
            max_act = activation[int(ppercentile * (len(activation) - 1))]

            weights, bias = layer.get_weights()
            max_wt = max(0, tf.math.reduce_max(weights))
            max_bias = tf.math.reduce_max(bias)

            max_wt_bias = max(max_bias, max_wt)

            scale_factor = max(max_act, max_wt_bias)
            applied_factor = scale_factor / prev_factor

            weights = weights / applied_factor
            bias = bias / scale_factor

            prev_factor = scale_factor
            layer.set_weights([weights, bias])
            print(f"Scale factor for layer {layer}")
            print(f"{applied_factor}")

    return model