import tensorflow as tf
import numpy as np
from spiking_models import DenseRNN, SpikingReLU, SpikingSigmoid, SpikingTanh, Accumulate


def convert(model, weights, x_test, y_test):
    print("Converted model:\n" + "-"*32)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            print("Input Layer")
            inputs = tf.keras.Input(shape=(1, model.layers[0].input_shape[0][1]), batch_size=y_test.shape[0])
            x = inputs
        elif isinstance(layer, tf.keras.layers.Dense):
            x = tf.keras.layers.Dense(layer.output_shape[1])(x)
            # x = tf.keras.layers.RNN(DenseRNN(layer.output_shape[1]), return_sequences=True, return_state=False, stateful=True)(x)
            if layer.activation.__name__ == 'linear':
                print("Dense Layer w/o activation")
                pass
            elif layer.activation.__name__ == 'relu':
                print("Dense Layer with SpikingReLU")
                x = tf.keras.layers.RNN(SpikingReLU(layer.output_shape[1]), return_sequences=True, return_state=False, stateful=True)(x)
            elif layer.activation.__name__ == 'sigmoid':
                print("Dense Layer with SpikingSigmoid")
                x = tf.keras.layers.RNN(SpikingSigmoid(layer.output_shape[1]), return_sequences=True, return_state=False, stateful=True)(x)
            elif layer.activation.__name__ == 'tanh':
                print("Dense Layer with SpikingTanh")
                x = tf.keras.layers.RNN(SpikingTanh(layer.output_shape[1]), return_sequences=True, return_state=False, stateful=True)(x)
            else:
                print('[Info] Activation type', layer.activation.__name__, 'not implemented')
        elif isinstance(layer, tf.keras.layers.ReLU):
            print("SpikingReLU Layer")
            x = tf.keras.layers.RNN(SpikingReLU(layer.output_shape[1]), return_sequences=True, return_state=False, stateful=True)(x)
        elif isinstance(layer, tf.keras.layers.Softmax):
            print("Accumulate + Softmax Layer")
            print(layer.output_shape[1])
            x = tf.keras.layers.RNN(Accumulate(layer.output_shape[1]), return_sequences=True, return_state=False, stateful=True)(x)
            x = tf.keras.layers.Softmax()(x)
        else:
            print("[Info] Layer type ", layer, "not implemented")
    spiking = tf.keras.models.Model(inputs=inputs, outputs=x)
    print("-"*32 + "\n")

    spiking.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["sparse_categorical_accuracy"],)

    spiking.set_weights(weights)
    return spiking


def get_normalized_weights(model, x_test, percentile=100):
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
        print("\n" + "-"*32 + "\nNo normalization\n" + "-"*32)
    else:
        print("\n" + "-"*32 + "\nNormalizing by", max_activation, "\n" + "-"*32)
        for i in range(len(weights)):
            weights[i] /= (max_activation)
    return weights


def evaluate_conversion(converted_model, original_model, x_test, y_test, testacc, timesteps=50):
    for i in range(1, timesteps+1):
        _, acc = converted_model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=0)
        print(
            "Timesteps", str(i) + "/" + str(timesteps) + " -",
            "acc spiking (orig): %.2f%% (%.2f%%)" % (acc*100, testacc*100),
            "- conv loss: %+.2f%%" % ((-(1 - acc/testacc)*100)))


tf.random.set_seed(1234)
batch_size=512
epochs = 5
act='relu'


def create_ann():
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(500, activation=act)(inputs)
    #x = tf.keras.layers.ReLU()(x)  # max_value=1
    x = tf.keras.layers.Dense(100, activation=act)(x)
    #x = tf.keras.layers.Activation(tf.nn.relu)(x)  # not implemented yet
    x = tf.keras.layers.Dense(10, activation=act)(x)
    x = tf.keras.layers.Softmax()(x)
    ann = tf.keras.Model(inputs=inputs, outputs=x)

    ann.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    ann.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs)
    return ann


if __name__ == "__main__":
    ##################################################
    # Import Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))

    # Analog model
    ann = create_ann()

    _, testacc = ann.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    #weights = ann.get_weights()
    weights = get_normalized_weights(ann, x_train, percentile=85)

    ##################################################
    # Preprocessing for RNN
    x_train = np.expand_dims(x_train, axis=1)  # (60000, 784) -> (60000, 1, 784)
    x_test = np.expand_dims(x_test, axis=1)
    #x_rnn = np.tile(x_train, (1, 1, 1))
    #y_rnn = y_train  # np.tile(x_test, (1, timesteps, 1))

    ##################################################
    # Conversion to spiking model
    snn = convert(ann, weights, x_test, y_test)
    evaluate_conversion(snn, ann, x_test, y_test, testacc, timesteps=50)
