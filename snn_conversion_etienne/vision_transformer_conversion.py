import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    RNN
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import numpy as np
from spiking_models import DenseRNN, SpikingReLU, SpikingSigmoid, SpikingTanh, Accumulate
import keras
from tensorflow.keras.utils import to_categorical


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                Dense(embed_dim),
                Dropout(dropout),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class SpikingMultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class SpikingTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = SpikingMultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                Dense(mlp_dim),
                RNN(SpikingReLU(mlp_dim), return_sequences=True, return_state=False, stateful=True),
                Dropout(dropout),
                Dense(embed_dim),
                Dropout(dropout),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


def convert(model, weights, x_test, y_test, d_model, num_heads, mlp_dim, dropout):
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
        elif isinstance(layer, TransformerBlock):
            x = SpikingTransformerBlock(d_model, num_heads, mlp_dim, dropout)(x)
        else:
            print("[Info] Layer type ", layer, "not implemented, so we use it's keras version")
            x = layer(x)
    spiking = tf.keras.models.Model(inputs=inputs, outputs=x)
    print("-"*32 + "\n")

    spiking.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["sparse_categorical_accuracy"],)

    spiking.set_weights(weights)
    return spiking


def extract_patches(images, patch_dim, patch_size):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patches = tf.reshape(patches, [batch_size, -1, patch_dim])
    return patches


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


if __name__ == "__main__":
    # Variable definition
    logdir = "logs"
    image_size = 28
    patch_size = 4
    num_layers = 4
    d_model = 64
    num_heads = 4
    mlp_dim = 128
    lr = 3e-4
    weight_decay = 1e-4
    epochs = 5
    batch_size = 50
    dropout = 0.1
    num_classes = 10
    channels = 1
    patch_dim = channels * patch_size ** 2
    num_patches = (image_size // patch_size) ** 2

    # Dataset
    # the data, shuffled and split between tran and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize input so we can train ANN with it.
    # Will be converted back to integers for SNN layer.
    # x_train = x_train / 255
    # x_test = x_test / 255

    # Add a channel dimension.
    axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    x_train = np.expand_dims(x_train, axis)
    x_test = np.expand_dims(x_test, axis)

    # One-hot encode target vectors.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Model
    input_shape = x_train.shape[1:]

    inp = tf.keras.layers.Input(shape=(input_shape))
    x = Rescaling(1.0 / 255)(inp)

    #  VISION PART
    # patching, positional embedding and class embedding
    patches = extract_patches(x, patch_dim, patch_size)
    x = Dense(d_model)(patches)

    pos_emb = tf.Variable(initial_value=tf.random.uniform(shape=(1, num_patches + 1, d_model)),
                          name="pos_emb", validate_shape=(1, num_patches + 1, d_model), trainable=True)
    class_emb = tf.Variable(initial_value=tf.random.uniform(shape=(1, 1, d_model)), name="class_emb",
                            validate_shape=(1, 1, d_model), trainable=True)

    class_emb = tf.broadcast_to(class_emb, [batch_size, 1, d_model])

    x = tf.concat([class_emb, x], axis=1)
    x = x + pos_emb

    # Transformer Blocks
    x = TransformerBlock(d_model, num_heads, mlp_dim, dropout)(x)

    #  MLP HEAD
    x = Dense(mlp_dim, activation=tf.nn.relu)(x[:, 0])
    x = Dense(num_classes)(x)

    #  Model compilation and training
    model = tf.keras.models.Model(inputs=inp, outputs=x)

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"],
    )

    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))

    ### =================== CONVERSION ========================
    _, testacc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    weights = get_normalized_weights(model, x_train, percentile=85)

    ##################################################
    # Preprocessing for RNN
    x_train = np.expand_dims(x_train, axis=1)  # (60000, 784) -> (60000, 1, 784)
    x_test = np.expand_dims(x_test, axis=1)
    # x_rnn = np.tile(x_train, (1, 1, 1))
    # y_rnn = y_train  # np.tile(x_test, (1, timesteps, 1))

    ##################################################
    # Conversion to spiking model
    snn = convert(model, weights, x_test, y_test, d_model, num_heads, mlp_dim, dropout)
    evaluate_conversion(snn, model, x_test, y_test, testacc, timesteps=50)