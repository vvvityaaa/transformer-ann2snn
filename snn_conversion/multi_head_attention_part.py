import tensorflow as tf
from operations_layers import MatMulLayer, MatMulLayerTranspose, TransposeLayer


def multi_head_self_attention(x, num_heads, projection_dim, d_model):

    """ Functional Multi Head Self Attention. Performs self attention for the input tensor x. In this form it can be
    used for conversion in other conversion frameworks.

    Parameters
    ----------
    x : Input tensor on which self attention is performed.
    num_heads : Dimension
    projection_dim : Dimension
    d_model: Dimension


    Returns
    -------
    sum : `float`
       Sum of ``values``.
    """

    # ---------------------- Multi Head Self Attention ---------------------
    v2 = tf.keras.layers.Dense(d_model)(x)
    q2 = tf.keras.layers.Dense(d_model)(x)
    k2 = tf.keras.layers.Dense(d_model)(x)

    # reshaping and transposing is equivalent to split_head function
    v = tf.keras.layers.Reshape([-1, num_heads, projection_dim])(v2)
    v = TransposeLayer()(v)
    q = tf.keras.layers.Reshape([-1, num_heads, projection_dim])(q2)
    q = TransposeLayer()(q)
    k = tf.keras.layers.Reshape([-1, num_heads, projection_dim])(k2)
    k = TransposeLayer()(k)

    # ------------------- Scaled dot-product attention ----------------------
    # QK^T
    att = MatMulLayerTranspose()([q, k])
    # softmax(QK^T)
    att = tf.keras.layers.Softmax(axis=-1)(att)
    # softmax(QK^T)*V
    att = MatMulLayer()([att, v])

    att = TransposeLayer()(att)
    att = tf.keras.layers.Reshape([-1, d_model])(att)
    att = tf.keras.layers.Dense(d_model)(att)
    # ------------------- End of Multi Head Self Attention -------------------

    return att