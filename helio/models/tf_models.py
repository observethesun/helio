"""Conv block and U-net model."""
import tensorflow as tf


def conv_block(x, layout, filters=None, transpose=False, rate=0.1, activation=None, is_training=True):
    """Convolutional block.

    Parameters
    ----------
    x : tensor
        Input tensor.
    layout : str
        Layout of layers. Can contain "c" for convolution, "a" for activation, "n" for batchnorm,
        "p" for maxpooling, "d" for dropout. E.g. of layout: "ccnapd".
    filters : int, list or None
        Number of filters for convolutions. Can be a single number (all convolutions will
        have the same number of filters), a list of the same length as a count of letters "c"
        in the layout, or None if the layout contains no "c".
    transpose : bool
        If true, transposed convolutions are used.
    rate : float
        Dropout rate parameter. Default to 0.1.
    activation : function
        Activation function. If not specified activation is tf.nn.elu.
    is_training: bool
        Phase of training for batchnorm.
        Default to True.

    Returns
    -------
    x : tensor
        Output tensor.
    """
    i = 0
    try:
        iter(filters)
    except TypeError:
        filters = list([filters] * layout.count('c'))

    for s in layout:
        if s == 'c':
            if transpose:
                x = tf.layers.conv2d_transpose(x, filters[i], (3, 3), strides=(2, 2), padding='same')
            else:
                x = tf.layers.conv2d(x, filters[i], (3, 3), padding='same')
            i += 1
        elif s == 'a':
            if activation is None:
                activation = tf.nn.elu
            x = activation(x)
        elif s == 'p':
            x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))
        elif s == 'n':
            x = tf.layers.batch_normalization(x, training=is_training, momentum=0.9)
        elif s == 'd':
            if rate is None:
                rate = 0.1
            x = tf.layers.dropout(x, rate=rate, training=is_training)
        else:
            raise KeyError('unknown letter {0}'.format(s))
    return x


def u_net(images, depth, init_filters, output=None, is_training=True, verbose=0):
    """U-net implementation.

    Parameters
    ----------
    images : 4d tensor
        Input tensor.
    depth : int
        Depth of the U-net.
    init_filters : int
        Number of filters in the first conv block.
    output : dict
        Output conv block config.
    is_training: bool
        Phase of training for batchnorm. Default to True.
    verbose : int
        Level for information messages. Default to 0.

    Returns
    -------
    up : 4d tensor
        Output tensor.
    """
    verboseprint = print if verbose == 0 else lambda *args, **kwargs: None

    conv_d = []

    conv = images
    verboseprint('input', conv.get_shape())

    for d in range(depth):
        conv = conv_block(conv, 'caca', init_filters * (2 ** d), is_training=is_training)
        verboseprint('conv_block_{0}'.format(d), conv.get_shape())
        conv_d.append(conv)
        conv = conv_block(conv, 'pd')
        verboseprint('pool_{0}'.format(d), conv.get_shape())

    conv = conv_block(conv, 'cacad', init_filters * (2 ** depth), is_training=is_training)
    verboseprint('bottom_conv_block_{0}'.format(depth), conv.get_shape())

    for d in range(depth, 0, -1):
        conv = conv_block(conv, 'cad', init_filters * (2 ** d), transpose=True, is_training=is_training)
        verboseprint('up_{0}'.format(d - 1), conv.get_shape())
        conv = tf.concat([conv, conv_d[d - 1]], axis=-1)
        verboseprint('concat_{0}'.format(d), conv.get_shape())
        conv = conv_block(conv, 'cacad', init_filters * (2 ** (d - 1)), is_training=is_training)
        verboseprint('up_conv_block_{0}'.format(d), conv.get_shape())

    if output is not None:
        out = conv_block(conv, is_training=is_training, **output)
    else:
        out = conv

    verboseprint('output shape', out.get_shape())
    return out
