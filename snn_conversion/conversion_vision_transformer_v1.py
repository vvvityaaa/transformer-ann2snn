import tensorflow as tf
import numpy as np
from spiking_models import DenseRNN, SpikingReLU, SpikingSigmoid, SpikingTanh, Accumulate
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import mnist
from operations_layers import SqueezeLayer, ExpandLayer, MatMulLayer, MatMulLayerTranspose, TransposeLayer, \
    ExtractPatchesLayer, PositionalEncodingLayer


# %%

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


def evaluate_conversion(converted_model, original_model, x_test, y_test, testacc, timesteps=50):
    for i in range(1, timesteps + 1):
        _, acc = converted_model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=0)
        print(
            "Timesteps", str(i) + "/" + str(timesteps) + " -",
                         "acc spiking (orig): %.2f%% (%.2f%%)" % (acc * 100, testacc * 100),
                         "- conv loss: %+.2f%%" % ((-(1 - acc / testacc) * 100)))


# %%

tf.random.set_seed(1238)
batch_size = 128
epochs = 2


def multi_head_attention(x):
    # ================== Multi Head Self Attention ===============
    v2 = tf.keras.layers.Dense(embed_dim)(x)
    q2 = tf.keras.layers.Dense(embed_dim)(x)
    k2 = tf.keras.layers.Dense(embed_dim)(x)

    v = tf.keras.layers.Reshape([l, num_heads, projection_dim])(v2)
    v = TransposeLayer()(v)
    q = tf.keras.layers.Reshape([l, num_heads, projection_dim])(q2)
    q = TransposeLayer()(q)
    k = tf.keras.layers.Reshape([l, num_heads, projection_dim])(k2)
    k = TransposeLayer()(k)

    # =============== Scaled dot-product attention =================
    # QK^T
    att = MatMulLayerTranspose()([q, k])
    # softmax(QK^T)
    att = tf.keras.layers.Softmax(axis=-1)(att)
    # softmax(QK^T)*V
    out = MatMulLayer()([att, v])

    att = TransposeLayer()(out)
    out = tf.keras.layers.Reshape([-1, l, embed_dim])(att)
    out = tf.keras.layers.Dense(embed_dim)(out)
    # out = tf.keras.layers.Reshape([l, d_model, 1])(out)
    x = tf.keras.layers.Reshape([-1, l, embed_dim])(x)
    # ============== End of Multi Head Self Attention =============
    # Concat Layer
    add = tf.keras.layers.Add()([out, x])
    # ================== End of Transformer =======================
    return out, add


def create_ann_approved_version():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))

    patches = ExtractPatchesLayer()(inputs)
    x = tf.keras.layers.Dense(d_model)(patches)
    x = PositionalEncodingLayer()(x)

    out = x
    for i in range(1):
        out, add = multi_head_attention(out)
        out = tf.keras.layers.Dense(mlp_dim, activation="relu")(add)
        out = tf.keras.layers.Dense(embed_dim)(out)
        out = tf.keras.layers.Add()([out, add])

    x = tf.keras.layers.Flatten()(out)
    x = tf.keras.layers.Dense(embed_dim, activation="relu")(x)
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


def convert_tailored_approved_version(weights, y_test):
    inputs = tf.keras.layers.Input(shape=(28, 28, 1), batch_size=y_test.shape[0])

    patches = ExtractPatchesLayer()(inputs)
    x = tf.keras.layers.Dense(d_model)(patches)
    x = PositionalEncodingLayer()(x)
    out = x
    for i in range(1):
        out, add = multi_head_attention(out)
        out = tf.keras.layers.Dense(mlp_dim)(add)
        print(out.shape)
        out = tf.keras.layers.Reshape([1, l * mlp_dim])(out)
        out = tf.keras.layers.RNN(SpikingReLU(l * mlp_dim), return_sequences=True, return_state=False,
                                  stateful=True)(out)
        out = tf.keras.layers.Reshape([1, l, mlp_dim])(out)

        out = tf.keras.layers.Dense(embed_dim)(out)
        out = tf.keras.layers.Add()([out, add])

    x = tf.keras.layers.Flatten()(out)
    x = ExpandLayer()(x)
    x = tf.keras.layers.Dense(embed_dim)(x)
    x = tf.keras.layers.RNN(SpikingReLU(embed_dim), return_sequences=True, return_state=False,
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

    dv = 24
    dout = 32
    nv = 8
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    embed_dim = d_model = 64  # Embedding size for each token
    mlp_dim = 128
    l = 50
    num_heads = 4
    num_classes = 10
    image_size = 28
    patch_size = 4
    num_patches = (image_size // patch_size) ** 2
    channels = 1
    patch_dim = channels * patch_size ** 2
    projection_dim = embed_dim // num_heads

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize input so we can train ANN with it.
    # Will be converted back to integers for SNN layer.
    x_train = x_train / 255
    x_test = x_test / 255

    # Add a channel dimension.
    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    x_train = np.expand_dims(x_train, axis)
    x_test = np.expand_dims(x_test, axis)

    # One-hot encode target vectors.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Analog model
    ann = create_ann_approved_version()
    print(ann.summary())

    _, testacc = ann.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    # weights = ann.get_weights()
    weights = get_normalized_weights(ann, x_train, percentile=85)

    ##################################################
    # Preprocessing for RNN
    # x_train = np.expand_dims(x_train, axis=1)  # (60000, 784) -> (60000, 1, 784)
    # x_test = np.expand_dims(x_test, axis=1)

    ##################################################
    # Conversion to spiking model
    # snn = convert(ann, weights, x_test, y_test)
    snn = convert_tailored_approved_version(weights, y_test)
    evaluate_conversion(snn, ann, x_test, y_test, testacc, timesteps=50)
