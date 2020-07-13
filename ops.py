import tensorflow as tf

w_initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
# w_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
b_initializer = tf.constant_initializer(0.0)


def conv(x, channels, kernel_size=3, stride=1, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels, strides=stride,
                             kernel_size=kernel_size, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer,
                             use_bias=use_bias, padding=padding)
        return x


def deconv(x, channels, kernel_size=3, stride=2, padding='SAME', reuse=False, use_bias=True, scope='deconv_0'):
    with tf.variable_scope(scope):
        x = tf.compat.v1.layers.Conv2DTranspose(filters=channels, kernel_size=kernel_size, strides=stride,
                                            padding=padding, use_bias=True, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, trainable=(not reuse), name=None)(x)
        return x

def denseBlock(x, growth_rate=16, kernel_size=3, stride=1, padding='SAME', scope='denseBlock_0'):
    with tf.variable_scope(scope):
        x_in = []
        for i in range(8):
            if i == 0:
                x = conv(x, channels=growth_rate, kernel_size=kernel_size, stride=stride,
                        padding=padding, use_bias=True, scope='conv_'+str(i))
                x = relu(x)
                x_in.append(x)
            else:
                x = concatenation(x_in)
                x = conv(x, channels=growth_rate, kernel_size=kernel_size, stride=stride, 
                        padding=padding, use_bias=True, scope='conv_'+str(i))
                x = relu(x)
                x_in.append(x)

        x = concatenation(x_in)
        return x


def concatenation(x):
    x = tf.concat(x, axis=3)
    return x


def bottleneck(x, channels=256, kernel_size=1, stride=1, padding='SAME', use_bias=True, scope='bottleneck_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels, strides=stride,
                             kernel_size=kernel_size, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer,
                             use_bias=use_bias, padding=padding)
        return x


def relu(x):
    return tf.nn.relu(x)