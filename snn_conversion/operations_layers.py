import tensorflow as tf
import numpy as np


class SqueezeLayer(tf.keras.layers.Layer):

    """
    Layer that encapsulates a squeezing operation for the axis 1.
    """

    def __init__(self):
        super(SqueezeLayer, self).__init__()

    def call(self, inputs):
        return tf.squeeze(inputs, axis=1)


class ExpandLayer(tf.keras.layers.Layer):
    """
    Layer that encapsulates expand operation for the axis 1.
    """

    def __init__(self):
        super(ExpandLayer, self).__init__()

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)


class Tokpos(tf.keras.layers.Layer):

    """
    Positional and Token embedding for NLP Transformer. Positional embedding is created as an embedding over a range
    from 0 to the max length.
    """

    def __init__(self, maxlen, vocab_size, embed_dim):
        super(Tokpos, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim, name="positional")
        self.token_emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, name="token")

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class Tokposangles(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(Tokposangles, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # self.pos_emb = tf.keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim, name="positional")
        self.token_emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, name="token")
        self.positions = self.positional_encoding(self.maxlen, self.embed_dim)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.token_emb(x)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        return x + self.positions[:, :seq_len, :]


class MatMulLayer(tf.keras.layers.Layer):

    """
    Layer for multiplication of inputs.
    """

    def __init__(self):
        super(MatMulLayer, self).__init__()

    def call(self, inputs):
        return tf.matmul(inputs[0], inputs[1])


class MatMulLayerTranspose(tf.keras.layers.Layer):
    def __init__(self):
        super(MatMulLayerTranspose, self).__init__()

    def call(self, inputs):
        return tf.matmul(inputs[0], inputs[1], transpose_b=True) / np.sqrt(inputs[0].shape[-1])


class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TransposeLayer, self).__init__()

    def call(self, inputs):
        return tf.transpose(inputs, perm=[0, 2, 1, 3])


class ExtractPatchesLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ExtractPatchesLayer, self).__init__()
        self.patch_size = 4
        self.patch_dim = 16

    def extract_patches(self, images, patch_size, patch_dim):
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

    def call(self, inputs):
        return self.extract_patches(inputs, self.patch_size, self.patch_dim)


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_patches):
        super(PositionalEncodingLayer, self).__init__()
        self.num_patches = num_patches
        self.d_model = d_model
        self.pos_emb = self.add_weight("pos_emb", shape=(1, self.num_patches + 1, self.d_model))
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.d_model))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        x = tf.concat([class_emb, inputs], axis=1)
        return x + self.pos_emb