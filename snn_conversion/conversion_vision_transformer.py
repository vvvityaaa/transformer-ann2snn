import tensorflow as tf
import numpy as np
import keras
from spiking_models import SpikingReLU, Accumulate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from operations_layers import SqueezeLayer, ExpandLayer, ExtractPatchesLayer, PositionalEncodingLayer
from weight_normalization import robust_weight_normalization
from utils import evaluate_conversion, evaluate_conversion_and_save_data
from multi_head_self_attention import multi_head_self_attention


def create_and_train_ann():

    """
    Definition and training of artificial neural network with defined architecture in a keras functional API way.

    :return: trained artificial neural network
    """

    inputs = tf.keras.layers.Input(shape=(28, 28, 1))

    patches = ExtractPatchesLayer()(inputs)
    x = tf.keras.layers.Dense(d_model)(patches)
    x = PositionalEncodingLayer(d_model, num_patches)(x)

    out = x
    for _ in range(num_multi_head_attention_modules):
        out = multi_head_self_attention(out, num_heads, projection_dim, d_model)
        x = tf.keras.layers.Reshape([-1, d_model])(x)
        add = tf.keras.layers.Add()([out, x])

        # feedforward mlp
        out = tf.keras.layers.Dense(mlp_dim, activation="relu")(add)
        out = tf.keras.layers.Dense(d_model)(out)

        out = tf.keras.layers.Add()([out, add])

    x = tf.keras.layers.Flatten()(out)
    x = tf.keras.layers.Dense(mlp_dim, activation="relu")(x)
    # --------------------------------------------------
    x = tf.keras.layers.Dense(num_classes)(x)
    x = tf.keras.layers.Softmax()(x)

    ann = tf.keras.models.Model(inputs=inputs, outputs=x)

    ann.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    ann.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs)
    return ann


def create_and_train_snn(weights, y_test):

    """
    Definition of spiking neural network. It copies ann network up to the dense layers with relu activation functions,
    which are translated into rnn layers with SpikingReLU cells (neurons). This network is not trained, it's weights
    are filled with normalized weights of artificial neural network.
    :param weights: normalized weights from ann
    :param y_test:
    :return:
    """

    inputs = tf.keras.layers.Input(shape=(28, 28, 1), batch_size=y_test.shape[0])

    patches = ExtractPatchesLayer()(inputs)
    x = tf.keras.layers.Dense(d_model)(patches)
    x = PositionalEncodingLayer(d_model, num_patches)(x)
    out = x
    for _ in range(num_multi_head_attention_modules):
        out = multi_head_self_attention(out, num_heads, projection_dim, d_model)
        x = tf.keras.layers.Reshape([-1, d_model])(x)
        add = tf.keras.layers.Add()([out, x])

        # feedforward mlp
        out = tf.keras.layers.Dense(mlp_dim)(add)
        out = tf.keras.layers.Reshape([1, l * mlp_dim])(out)
        out = tf.keras.layers.RNN(SpikingReLU(l * mlp_dim), return_sequences=True, return_state=False,
                                  stateful=True)(out)
        out = tf.keras.layers.Reshape([-1, mlp_dim])(out)
        out = tf.keras.layers.Dense(d_model)(out)

        out = tf.keras.layers.Add()([out, add])

    x = tf.keras.layers.Flatten()(out)
    x = ExpandLayer()(x)
    x = tf.keras.layers.Dense(mlp_dim)(x)
    x = tf.keras.layers.RNN(SpikingReLU(mlp_dim), return_sequences=True, return_state=False,
                            stateful=True)(x)
    # --------------------------------------------------
    x = tf.keras.layers.Dense(num_classes)(x)

    x = tf.keras.layers.RNN(Accumulate(num_classes), return_sequences=True, return_state=False, stateful=True)(x)
    x = tf.keras.layers.Softmax()(x)

    x = SqueezeLayer()(x)

    spiking = tf.keras.models.Model(inputs=inputs, outputs=x)

    print("-" * 32 + "\n")
    spiking.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    print(spiking.summary())
    spiking.set_weights(weights)
    return spiking


if __name__ == "__main__":
    tf.random.set_seed(1234)
    batch_size = 64
    epochs = 2
    d_model = 64
    mlp_dim = 128
    l = 50
    num_heads = 4
    num_classes = 10
    channels = 1
    image_size = 28
    patch_size = 4
    num_patches = (image_size // patch_size) ** 2
    patch_dim = channels * patch_size ** 2
    projection_dim = d_model // num_heads
    num_multi_head_attention_modules = 4
    timesteps = 50

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize input so we can train ANN with it.
    # Will be converted back to integers for SNN layer.
    x_train = x_train / 255
    x_test = x_test / 255

    # One-hot encode target vectors.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Analog model
    ann = create_and_train_ann()
    print(ann.summary())

    _, testacc = ann.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    model_normalized = robust_weight_normalization(ann, x_test, ppercentile=0.99)
    weights = model_normalized.get_weights()

    # Preprocessing for RNN
    # Add a channel dimension.
    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    x_train_expanded = np.expand_dims(x_train, axis)
    x_test_expanded = np.expand_dims(x_test, axis)

    # Conversion to spiking model
    snn = create_and_train_snn(weights, y_test)

    print("Simulating network")
    evaluate_conversion(snn, x_test_expanded, y_test, testacc, y_test.shape[0], timesteps)
