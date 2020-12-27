import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.utils.io_utils import path_to_string
try:
    import h5py
except ImportError:
    h5py = None


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
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
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
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        # TODO: adapt MLP architecture in the TransformerBlock
        self.mlp = tf.keras.Sequential(
            [
                # TODO: check potential problems of using relu instead of gelu
                Dense(mlp_dim, activation=tf.nn.relu),
                Dense(embed_dim),
            ]
        )

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        out1 = attn_output + inputs

        mlp_output = self.mlp(out1)
        return mlp_output + out1


class VisionTransformer(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_layers, num_classes, d_model, num_heads, mlp_dim, channels=1):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.rescale = Rescaling(1.0 / 255)
        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches + 1, d_model)
        )
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model))
        self.patch_proj = Dense(d_model)
        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim)
            for _ in range(num_layers)
        ]

        # TODO: adapt MLP architecture after TransformerBlock
        self.mlp_head = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)

        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        x = tf.concat([class_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x

    def load(self, weights, biases, transpose=True):

        """
        Analog to previous method, but weights and biases are passed directly
        it is used for converting networks. For some reason, when converting, weights are saved in transposed form, so
        we need to transpose them back to the correct shape. Whether to transpose or not can be controlled with the
        parameter transpose
        """
        # TODO: adapt to be working
        for i in range(0, len(weights)):
            to_add = weights[i]
            if transpose:
                to_add = to_add.transpose(0, 1)
            self.weights[i] = to_add
            if biases[i] is not None:
                self.bias[i] = biases[i]
            else:
                self.bias[i] = biases[i]

    def load_weights(self, filepath):
        """Loads all layer weights, either from a TensorFlow or an HDF5 weight file.
        Arguments:
            filepath: String, path to the weights file to load. For weight files in
                TensorFlow format, this is the file prefix (the same as was passed
                to `save_weights`).
        Returns:
            When loading a weight file in TensorFlow format, returns the same status
            object as `tf.train.Checkpoint.restore`. When graph building, restore
            ops are run automatically as soon as the network is built (on first call
            for user-defined classes inheriting from `Model`, immediately if it is
            already built).
            When loading weights in HDF5 format, returns `None`.
        Raises:
            ImportError: If h5py is not available and the weight file is in HDF5
                format.
            ValueError: If `skip_mismatch` is set to `True` when `by_name` is
              `False`.
        """

        filepath = path_to_string(filepath)
        if h5py is None:
            raise ImportError(
                '`load_weights` requires h5py when loading weights from HDF5.')
        if not self._is_graph_network and not self.built:
            raise ValueError(
                'Unable to load weights saved in HDF5 format into a subclassed '
                'Model which has not created its variables yet. Call the Model '
                'first, then load the weights.')
        with h5py.File(filepath, 'r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            else:
                hdf5_format.load_weights_from_hdf5_group(f, self.layers)
