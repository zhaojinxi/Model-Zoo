import tensorflow

regularizer = tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7)

def MobileNetV2Block(x, stride, t, output_channel, layer, training):
    with tensorflow.variable_scope('MobileNetV2Block%s' % layer):
        with tensorflow.variable_scope('conv1'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[1, 1, x.shape.as_list()[-1], int(t * x.shape.as_list()[-1])],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=regularizer)
            bias = tensorflow.get_variable(
                name='bias',
                shape=int(t * x.shape.as_list()[-1]),
                initializer=tensorflow.initializers.zeros(),
                regularizer=regularizer)
            y = tensorflow.nn.conv2d(
                input=x,
                filter=weight,
                strides=[1, 1, 1, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)
            y = tensorflow.nn.relu6(y)

        with tensorflow.variable_scope('conv2'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[3, 3, y.shape.as_list()[-1], 1],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=regularizer)
            bias = tensorflow.get_variable(
                name='bias',
                shape=y.shape.as_list()[-1],
                initializer=tensorflow.initializers.zeros(),
                regularizer=regularizer)
            y = tensorflow.nn.depthwise_conv2d(
                input=y,
                filter=weight,
                strides=[1, stride, stride, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)
            y = tensorflow.nn.relu6(y)

        with tensorflow.variable_scope('conv3'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[1, 1, y.shape.as_list()[-1], output_channel],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=regularizer)
            bias = tensorflow.get_variable(
                name='bias',
                shape=output_channel,
                initializer=tensorflow.initializers.zeros(),
                regularizer=regularizer)
            y = tensorflow.nn.conv2d(
                input=y,
                filter=weight,
                strides=[1, 1, 1, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)

        if x.shape.as_list() == y.shape.as_list():
            y = x + y
    return y

def MobileNetV2(x, width, num_classes, training):
    with tensorflow.variable_scope('MobileNetV2', reuse=tensorflow.AUTO_REUSE):
        with tensorflow.variable_scope('layer1'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[3, 3, x.shape.as_list()[-1], int(32 * width)],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=regularizer)
            bias = tensorflow.get_variable(
                name='bias',
                shape=int(32 * width),
                initializer=tensorflow.initializers.zeros(),
                regularizer=regularizer)
            y = tensorflow.nn.conv2d(
                input=x,
                filter=weight,
                strides=[1, 2, 2, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)
            y = tensorflow.nn.relu6(y)

        y = MobileNetV2Block(y, 1, 1, int(16 * width), 2, training)

        y = MobileNetV2Block(y, 2, 6, int(24 * width), 3, training)
        y = MobileNetV2Block(y, 1, 6, int(24 * width), 4, training)

        y = MobileNetV2Block(y, 2, 6, int(32 * width), 5, training)
        y = MobileNetV2Block(y, 1, 6, int(32 * width), 6, training)
        y = MobileNetV2Block(y, 1, 6, int(32 * width), 7, training)

        y = MobileNetV2Block(y, 2, 6, int(64 * width), 8, training)
        y = MobileNetV2Block(y, 1, 6, int(64 * width), 9, training)
        y = MobileNetV2Block(y, 1, 6, int(64 * width), 10, training)
        y = MobileNetV2Block(y, 1, 6, int(64 * width), 11, training)

        y = MobileNetV2Block(y, 1, 6, int(96 * width), 12, training)
        y = MobileNetV2Block(y, 1, 6, int(96 * width), 13, training)
        y = MobileNetV2Block(y, 1, 6, int(96 * width), 14, training)

        y = MobileNetV2Block(y, 2, 6, int(160 * width), 15, training)
        y = MobileNetV2Block(y, 1, 6, int(160 * width), 16, training)
        y = MobileNetV2Block(y, 1, 6, int(160 * width), 17, training)

        y = MobileNetV2Block(y, 1, 6, int(320 * width), 18, training)

        with tensorflow.variable_scope('layer19'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[1, 1, y.shape.as_list()[-1], int(1280 * width)],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=regularizer)
            bias = tensorflow.get_variable(
                name='bias',
                shape=int(1280 * width),
                initializer=tensorflow.initializers.zeros(),
                regularizer=regularizer)
            y = tensorflow.nn.conv2d(
                input=y,
                filter=weight,
                strides=[1, 1, 1, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)
            y = tensorflow.nn.relu6(y)
            y = tensorflow.nn.avg_pool2d(y, 7, 1, 'VALID')

        with tensorflow.variable_scope('layer20'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[1, 1, y.shape.as_list()[-1], num_classes],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=regularizer)
            bias = tensorflow.get_variable(
                name='bias',
                shape=num_classes,
                initializer=tensorflow.initializers.zeros(),
                regularizer=regularizer)
            y = tensorflow.nn.conv2d(
                input=y,
                filter=weight,
                strides=[1, 1, 1, 1],
                padding='SAME') + bias
    return y

if __name__ == '__main__':
    tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True), inter_op_parallelism_threads=0, intra_op_parallelism_threads=0))
    data = tensorflow.random.uniform([128, 224, 224, 3])
    y = MobileNetV2(data, 1.4, 10, True)
    print()