import tensorflow as tf
import numpy as np
from multi_head_self_attention import multi_head_self_attention
from spiking_models import SpikingReLU, Accumulate
from tensorflow.keras.utils import to_categorical
from operations_layers import SqueezeLayer, ExpandLayer, Tokpos
from weight_normalization import robust_weight_normalization
from utils import evaluate_conversion, evaluate_conversion_and_save_data


def create_and_train_ann():

    """
    Definition and training of artificial neural network with defined architecture in a keras functional API way.

    :return: trained artificial neural network
    """

    inputs = tf.keras.layers.Input(shape=(maxlen,))
    x = Tokpos(maxlen, vocab_size, d_model)(inputs)
    out = x
    for _ in range(num_multi_head_attention_modules):
        out = multi_head_self_attention(out, num_heads, d_model, d_model)
        x = tf.keras.layers.Reshape([-1, d_model])(x)
        add = tf.keras.layers.Add()([out, x])

        # feedforward mlp
        out = tf.keras.layers.Dense(mlp_dim, activation="relu")(add)
        out = tf.keras.layers.Dense(d_model)(out)
        out = tf.keras.layers.Add()([out, add])

    x = tf.keras.layers.GlobalAveragePooling1D()(out)
    x = tf.keras.layers.Dense(d_model, activation="relu")(x)
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.Dense(mlp_dim, activation="relu")(x)

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
    :param y_test: gives information about batch_size, it is a size of test labels
    :return: spiking model with weight set with normalized weights of ann
    """

    inputs = tf.keras.layers.Input(shape=(1, maxlen,), batch_size=y_test.shape[0])
    x = Tokpos(maxlen, vocab_size, d_model)(inputs)
    out = x
    for _ in range(num_multi_head_attention_modules):
        out = multi_head_self_attention(out, num_heads, d_model, d_model)
        x = tf.keras.layers.Reshape([-1, d_model])(x)
        add = tf.keras.layers.Add()([out, x])

        # feedforward mlp
        out = tf.keras.layers.Dense(mlp_dim)(add)
        n_dimension = np.prod(out.shape[1:])
        out = tf.keras.layers.Reshape([1, n_dimension])(out)
        out = tf.keras.layers.RNN(SpikingReLU(n_dimension), return_sequences=True, return_state=False,
                                  stateful=True)(out)
        out = tf.keras.layers.Reshape([-1, mlp_dim])(out)
        out = tf.keras.layers.Dense(d_model)(out)

        out = tf.keras.layers.Add()([out, add])

    x = tf.keras.layers.GlobalAveragePooling1D()(out)
    x = ExpandLayer()(x)
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.RNN(SpikingReLU(d_model), return_sequences=True, return_state=False,
                            stateful=True)(x)
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.Dense(mlp_dim)(x)
    x = tf.keras.layers.RNN(SpikingReLU(mlp_dim), return_sequences=True, return_state=False,
                            stateful=True)(x)

    x = tf.keras.layers.Dense(num_classes)(x)

    x = tf.keras.layers.RNN(Accumulate(num_classes), return_sequences=True, return_state=False, stateful=True)(x)
    x = tf.keras.layers.Softmax()(x)

    x = SqueezeLayer()(x)

    spiking = tf.keras.models.Model(inputs=inputs, outputs=x)

    print("-" * 32 + "\n")
    spiking.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print(spiking.summary())
    spiking.set_weights(weights)
    return spiking


if __name__ == "__main__":
    tf.random.set_seed(1234)
    batch_size = 64
    epochs = 2
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    d_model = 32
    mlp_dim = 64
    num_heads = 2
    l = maxlen // num_heads
    num_multi_head_attention_modules = 2
    num_classes = 2
    timesteps = 50

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    x_train_expanded = np.expand_dims(x_train, axis=1)
    x_test_expanded = np.expand_dims(x_test, axis=1)

    # Analog model
    ann = create_and_train_ann()
    print(ann.summary())

    _, testacc = ann.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    # weights = ann.get_weights()

    model_normalized = robust_weight_normalization(ann, x_test, ppercentile=0.99)
    weights = model_normalized.get_weights()

    snn = create_and_train_snn(weights, y_test)
    evaluate_conversion(snn, x_test_expanded, y_test, testacc, y_test.shape[0], timesteps)
