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

def deconv(x, channels, kernel_size=3, stride=2, padding='SAME', use_bias=True, scope='deconv_0'):
    bs, h, w, c = x.shape
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels, strides=stride,
                             kernel_size=kernel_size, kernel_initializer=w_initializer,
                             bias_initializer=b_initializer,
                             use_bias=use_bias, padding=padding)
        x = x[:,:h*2,:w*2,:]
        return x

def dense_block(x, growth_rate=16, kernel_size=3, stride=1, padding='SAME', scope='dense_block_0'):
    with tf.variable_scope(scope):
        x = conv(x, channels=growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, use_bias=True, scope='conv_0')
        x = relu(x)
        x_init = x
        for i in range(1, 8):
            x = conv(x, channels=growth_rate, kernel_size=kernel_size, stride=stride, padding=padding, use_bias=True, scope='conv_'+str(i))
            x = relu(x)
            x_init = tf.concat([x_init, x], axis=3)
            x = x_init
        
        return x


def skip_connection(x):
    return x

def bottleneck_layer(x, scope='bottleneck_layer_0'):
    with tf.variable_scope(scope):
        x = conv(x, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_0')
        return x

def relu(x):
    return tf.nn.relu(x)